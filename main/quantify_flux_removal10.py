import os
import csv
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

# --- Lightweight GUI for file selection & dialogs ---
import tkinter as tk
from tkinter import filedialog, messagebox

# ROI drawing
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ==============================
# Utility Functions
# ==============================
def load_image_rgb(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0

def save_image_uint8(path, arr):
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def to_gray(img_rgb):
    r, g, b = img_rgb[...,0], img_rgb[...,1], img_rgb[...,2]
    return 0.2126*r + 0.7152*g + 0.0722*b

def hanning_window(h, w):
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx)

def phase_correlation_shift(ref_gray, mov_gray):
    h, w = ref_gray.shape
    win = hanning_window(h, w)
    F1 = np.fft.fft2(ref_gray * win)
    F2 = np.fft.fft2(mov_gray * win)
    R = F1 * np.conj(F2)
    denom = np.abs(R); denom[denom == 0] = 1e-9
    R /= denom
    r = np.fft.ifft2(R)
    r = np.fft.fftshift(r)
    peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    cy, cx = r.shape[0]//2, r.shape[1]//2
    dy = peak[0] - cy
    dx = peak[1] - cx
    return int(dy), int(dx)

def apply_translation(img, dy, dx):
    """
    Apply integer translation with edge replication (avoids black borders).
    """
    h, w, c = img.shape
    out = np.empty_like(img)
    # Start with a copy so we can fill everything and then overwrite the valid region
    # Fill by replicating edges
    out[:] = img[
        np.clip(np.arange(h)[:,None], 0, h-1),
        np.clip(np.arange(w)[None,:], 0, w-1)
    ][:]  # broadcasted indexing trick to pre-fill with edge values

    # Now paste the translated valid region
    y_src_start = max(0, -dy); y_src_end = min(h, h - dy)
    x_src_start = max(0, -dx); x_src_end = min(w, w - dx)
    y_dst_start = max(0, dy);  y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx);  x_dst_end = x_dst_start + (x_src_end - x_src_start)

    if (y_src_end > y_src_start) and (x_src_end > x_src_start):
        out[y_dst_start:y_dst_end, x_dst_start:x_dst_end, :] = img[y_src_start:y_src_end, x_src_start:x_src_end, :]
    return out

def per_channel_normalize_to_ref(img, ref):
    out = img.copy()
    eps = 1e-6
    for ch in range(3):
        m_ref = ref[..., ch].mean()
        s_ref = ref[..., ch].std() + eps
        m_img = img[..., ch].mean()
        s_img = img[..., ch].std() + eps
        out[..., ch] = (img[..., ch] - m_img) * (s_ref / s_img) + m_ref
    return np.clip(out, 0.0, 1.0)

def compute_normalized_red(img):
    eps = 1e-6
    s = img.sum(axis=2) + eps
    return img[..., 0] / s

def otsu_threshold(data):
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(data, bins=256)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-9)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-9))[::-1]
    between_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
    if between_var.size == 0 or not np.isfinite(between_var).any():
        return float(data.mean())
    idx = np.argmax(between_var)
    return float((bin_centers[idx] + bin_centers[idx+1]) / 2.0) if idx+1 < len(bin_centers) else float(bin_centers[idx])

def overlay_heatmap_on_base(base_rgb, delta, alpha=0.55):
    """
    Overlay positive delta as red on a given base image without darkening
    the background. No multiplicative darkening.
    """
    base = np.clip(base_rgb, 0, 1)
    d = delta.copy()
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, 0, None)

    # Normalize to P95 to avoid over-saturation
    scale = np.percentile(d[d > 0], 95) if np.any(d > 0) else 1.0
    scale = max(scale, 1e-6)
    norm = np.clip(d / scale, 0, 1)

    red = np.zeros_like(base)
    red[..., 0] = norm

    # Pure alpha blend without darkening
    out = base * (1.0) + alpha * red
    return np.clip(out, 0, 1)


# ==============================
# Interactive helpers (Tk)
# ==============================
def select_image_dialog(title_text, info_text):
    root = tk.Tk(); root.withdraw(); root.update()
    messagebox.showinfo(title=title_text, message=info_text)
    filetypes = [("Image files","*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp"), ("All files","*.*")]
    path = filedialog.askopenfilename(title=title_text, filetypes=filetypes)
    root.destroy()
    if not path: raise FileNotFoundError("No file selected.")
    return path

def ask_yes_no(title, msg):
    root = tk.Tk(); root.withdraw()
    result = messagebox.askyesno(title=title, message=msg)
    root.destroy()
    return result

def pick_output_folder_dialog(default_name="FluxCleaning"):
    """
    Ask user to choose a parent folder, then create a timestamped subfolder.
    Returns absolute output folder path.
    """
    root = tk.Tk(); root.withdraw(); root.update()
    messagebox.showinfo("Select output location", "Choose a folder to save results.")
    parent = filedialog.askdirectory(title="Choose output folder")
    root.destroy()
    if not parent:
        raise FileNotFoundError("No output folder selected.")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(parent) / f"{default_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir.resolve())

# ==============================
# ROI selection via mouse drag
# ==============================
def pick_roi_with_mouse(image_rgb):
    fig, ax = plt.subplots(num="Select ROI - drag to draw, press ENTER when done")
    ax.imshow(image_rgb)
    ax.set_title("Drag to select ROI, press ENTER to confirm. Press 'r' to reset.")
    plt.tight_layout()

    roi = {'x1': None, 'y1': None, 'x2': None, 'y2': None}

    def onselect(eclick, erelease):
        roi['x1'], roi['y1'] = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        roi['x2'], roi['y2'] = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))

    def toggle_selector(event):
        if event.key in ['R','r']:
            rs.to_draw.set_visible(False)
            fig.canvas.draw_idle()
            roi['x1']=roi['y1']=roi['x2']=roi['y2']=None

    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)

    done = {'flag': False}
    def on_key(event):
        if event.key == 'enter':
            done['flag'] = True
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    if not done['flag'] or None in roi.values():
        return None
    x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
    x, y = max(0, x1), max(0, y1)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return (x, y, w, h)

# ==============================
# Core Analysis
# ==============================
def analyze(pre_path, post_flux_path, post_clean_path, roi=None, align=True, normalize_color=True, outdir='output'):
    os.makedirs(outdir, exist_ok=True)

    # 1) Load & size harmonization
    pre = load_image_rgb(pre_path)
    pf  = load_image_rgb(post_flux_path)
    pc  = load_image_rgb(post_clean_path)
    h = min(pre.shape[0], pf.shape[0], pc.shape[0])
    w = min(pre.shape[1], pf.shape[1], pc.shape[1])
    pre, pf, pc = pre[:h,:w,:], pf[:h,:w,:], pc[:h,:w,:]

    # 2) Alignment (translation only)
    shifts = {'pf': (0,0), 'pc': (0,0)}
    if align:
        ref_gray = to_gray(pre)
        dy_pf, dx_pf = phase_correlation_shift(ref_gray, to_gray(pf))
        dy_pc, dx_pc = phase_correlation_shift(ref_gray, to_gray(pc))
        pf = apply_translation(pf, dy_pf, dx_pf)
        pc = apply_translation(pc, dy_pc, dx_pc)
        shifts = {'pf': (int(dy_pf), int(dx_pf)), 'pc': (int(dy_pc), int(dx_pc))}

    # 3) Color normalization
    if normalize_color:
        pf = per_channel_normalize_to_ref(pf, pre)
        pc = per_channel_normalize_to_ref(pc, pre)

    # 4) ROI
    if roi is not None:
        x, y, rw, rh = roi
        pre = pre[y:y+rh, x:x+rw, :]
        pf  = pf [y:y+rh, x:x+rw, :]
        pc  = pc [y:y+rh, x:x+rw, :]

    # 5) Δ(redness)
    nR_pre, nR_pf, nR_pc = compute_normalized_red(pre), compute_normalized_red(pf), compute_normalized_red(pc)
    d_pf, d_pc = nR_pf - nR_pre, nR_pc - nR_pre

    # 6) Threshold from post‑flux positives
    pos_pf = d_pf[d_pf > 0]
    if pos_pf.size > 100:
        thr = otsu_threshold(pos_pf)
    else:
        thr = max(0.0, float(d_pf.mean() + d_pf.std()))

    mask_pf = d_pf > thr
    mask_pc = d_pc > thr

    # 7) Metrics
    def metrics(delta, mask):
        total = delta.size
        area_frac = float(mask.sum()) / float(total) if total > 0 else 0.0
        positive = delta[mask]
        mean_pos = float(positive.mean()) if positive.size > 0 else 0.0
        integrated = float(np.clip(delta, 0, None).sum())
        return area_frac, mean_pos, integrated

    area_pre, mean_pre, integ_pre = 0.0, 0.0, 0.0
    area_pf, mean_pf, integ_pf = metrics(d_pf, mask_pf)
    area_pc, mean_pc, integ_pc = metrics(d_pc, mask_pc)

    def efficiency(m_post, m_clean, m_pre):
        denom = (m_post - m_pre)
        if abs(denom) < 1e-9:
            return 1.0 if abs(m_clean - m_pre) < 1e-9 else 0.0
        return float(np.clip((m_post - m_clean) / denom, 0.0, 1.0))

    eff_area = efficiency(area_pf, area_pc, area_pre)
    eff_mean = efficiency(mean_pf, mean_pc, mean_pre)
    eff_integ = efficiency(integ_pf, integ_pc, integ_pre)

    # 8) Overlays
    d_pf_pos = np.clip(d_pf, 0, None)
    d_pc_pos = np.clip(d_pc, 0, None)
    # Use PRE as base so background stays gray
    pf_overlay = overlay_heatmap_on_base(pre, d_pf_pos, alpha=0.55)
    pc_overlay = overlay_heatmap_on_base(pre, d_pc_pos, alpha=0.55)
    save_image_uint8(os.path.join(outdir, 'post_flux_overlay.png'), pf_overlay)
    save_image_uint8(os.path.join(outdir, 'post_clean_overlay.png'), pc_overlay)

    # 9) CSV
    csv_path = os.path.join(outdir, 'cleaning_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric','PreFlux','PostFlux','PostClean','CleaningEfficiency(0-1)','Notes'])
        writer.writerow(['AreaFraction(>thr)', area_pre, area_pf, area_pc, eff_area, f'thr={thr:.6f}'])
        writer.writerow(['MeanDeltaOnStain', mean_pre, mean_pf, mean_pc, eff_mean, ''])
        writer.writerow(['IntegratedDelta', integ_pre, integ_pf, integ_pc, eff_integ, ''])

    return {
        'threshold': float(thr),
        'area': {'pre': area_pre, 'post_flux': area_pf, 'post_clean': area_pc, 'efficiency': eff_area},
        'mean_delta': {'pre': mean_pre, 'post_flux': mean_pf, 'post_clean': mean_pc, 'efficiency': eff_mean},
        'integrated_delta': {'pre': integ_pre, 'post_flux': integ_pf, 'post_clean': integ_pc, 'efficiency': eff_integ},
        'shifts': shifts,
        'outputs': [os.path.join(outdir,'post_flux_overlay.png'),
                    os.path.join(outdir,'post_clean_overlay.png'),
                    csv_path]
    }

# ==============================
# CLI parsing (optional)
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(description="Interactive quantification of flux cleaning via red dye images.")
    ap.add_argument("--pre", default=None, help="Path to pre-flux image (optional; dialog if omitted).")
    ap.add_argument("--postflux", default=None, help="Path to post-flux image (optional; dialog if omitted).")
    ap.add_argument("--postclean", default=None, help="Path to post-clean image (optional; dialog if omitted).")
    ap.add_argument("--outdir", default=None, help="If provided, saves directly to this folder (no dialog).")
    ap.add_argument("--noalign", action="store_true", help="Disable alignment.")
    ap.add_argument("--nonorm", action="store_true", help="Disable color normalization.")
    return ap.parse_args()

# ==============================
# Main
# ==============================
def main():
    args = parse_args()

    # 0) Ask user where to save (unless --outdir provided)
    if args.outdir:
        outdir = Path(args.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            outdir = Path(pick_output_folder_dialog()).resolve()
        except FileNotFoundError:
            print("User canceled output folder selection. Exiting.")
            return

    # 1) Select images (dialogs if not provided)
    pre_path = args.pre or select_image_dialog("Select image", "Select BEFORE image (no flux).")
    pf_path  = args.postflux or select_image_dialog("Select image", "Select POST-FLUX image (red dye present).")
    pc_path  = args.postclean or select_image_dialog("Select image", "Select POST-CLEAN image.")

    # 2) ROI?
    use_roi = ask_yes_no("ROI", "Do you want to draw a ROI (analyze only a selected region)?")
    roi = None
    if use_roi:
        before_img = load_image_rgb(pre_path)
        roi = pick_roi_with_mouse(before_img)
        if roi is None:
            root = tk.Tk(); root.withdraw()
            messagebox.showinfo("ROI", "No ROI selected. Full image will be used.")
            root.destroy()

    # 3) Analyze
    results = analyze(
        pre_path, pf_path, pc_path,
        roi=roi, align=(not args.noalign), normalize_color=(not args.nonorm),
        outdir=str(outdir)
    )

    # 4) Open output folder and show summary
    try:
        os.startfile(str(outdir))
    except Exception:
        pass

    msg = (
        f"Done.\n\n"
        f"Threshold (Δ red): {results['threshold']:.6f}\n"
        f"Area eff: {results['area']['efficiency']:.4f}\n"
        f"Mean eff: {results['mean_delta']['efficiency']:.4f}\n"
        f"Integ eff: {results['integrated_delta']['efficiency']:.4f}\n\n"
        f"Saved to:\n{outdir}\n\n"
        f"- {Path(results['outputs'][0]).name}\n"
        f"- {Path(results['outputs'][1]).name}\n"
        f"- {Path(results['outputs'][2]).name}\n"
    )
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("Flux Cleaning Quantification", msg)
    root.destroy()

    print("Output folder:", outdir)
    print("Results:", results)

if __name__ == "__main__":
    main()