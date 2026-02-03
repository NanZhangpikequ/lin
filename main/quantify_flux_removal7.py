#!/usr/bin/env python3
"""
quantify_flux_review.py

Quantify red-colored flux removal on microscope images with internal-review aids:
- Robust overlays (AFTER-FLUX / AFTER-CLEAN / REMOVED-AREA)
- Debug masks and BEFORE overlay (optional)
- Presets for sensitivity tuning
- Noise floor (confidence band) estimator
- Full parameter snapshot in metrics.json
- Auto-Sweep to suggest threshold combos (Top-K) for faint residue detection

Key Enhancements in this version:
- More sensitive and robust red detection for faint/brownish residue:
  * HSV red windows (wrap-around) +
  * BGR absolute dominance (R > max(G,B) + r_delta) +
  * BGR relative dominance (R/G and R/B > ratio_k) +
  * Lab a* reinforcement (a* > a_thresh)
  Combined via OR to improve recall; morphology + min-area suppress noise.
- Smaller morphology only for AFTER-CLEAN (when preset == 'sensitive') to preserve tiny islands.
- AFTER-CLEAN overlay color changed to cyan with lower alpha for better visibility.

Typical usage:
  python quantify_flux_review.py --gui --align --save-debug
  python quantify_flux_review.py --gui-multi --align --preset conservative
  python quantify_flux_review.py --noise-floor clean1.jpg clean2.jpg clean3.jpg --preset conservative
  python quantify_flux_review.py --gui --align --auto-sweep --save-debug

Recommended sensitive run for faint residue:
  Windows (single line):
    python quantify_flux_review.py --gui --align --preset sensitive --min-region-area 12 --min-region-area-residual 6 --r-delta 8 --red-hue-width 18 --min-sat 45 --min-val 38 --save-debug --export-masks
  macOS/Linux (multi-line OK):
    python quantify_flux_review.py \
      --gui --align \
      --preset sensitive \
      --min-region-area 12 \
      --min-region-area-residual 6 \
      --r-delta 8 \
      --red-hue-width 18 \
      --min-sat 45 \
      --min-val 38 \
      --save-debug --export-masks
"""

import os
import json
import csv
import argparse
from typing import Tuple, Dict, Optional, List, Any
from itertools import product

import numpy as np
import cv2

# Try to import SSIM from scikit-image if available; otherwise use a fallback implementation.
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_SSIM = True
except Exception:
    SKIMAGE_SSIM = False


# ------------- Generic Utilities -------------
def try_import_tk():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        return tk, filedialog, messagebox
    except Exception:
        return None, None, None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_to_match(src: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    th, tw = target_shape[:2]
    return cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert numpy types/arrays to Python-native types
    so json.dump doesn't fail (np.float32, np.int64, np.ndarray, etc.).
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def parse_int_list(s: Optional[str], default: List[int]) -> List[int]:
    if s is None or s == "":
        return list(default)
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


# ------------- Alignment / Metrics -------------
def align_to_reference(
    ref_bgr: np.ndarray,
    mov_bgr: np.ndarray,
    warp_mode: str = "affine",
    number_of_iterations: int = 300,
    termination_eps: float = 1e-6
) -> np.ndarray:
    """
    Align mov_bgr to ref_bgr using ECC. Returns aligned mov_bgr.
    Supported warp_mode: 'translation', 'euclidean', 'affine', 'homography'
    """
    ref_gray = to_gray(ref_bgr)
    mov_gray = to_gray(mov_bgr)

    ref_gray_f = ref_gray.astype(np.float32) / 255.0
    mov_gray_f = mov_gray.astype(np.float32) / 255.0

    if warp_mode == "translation":
        wm = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif warp_mode == "euclidean":
        wm = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif warp_mode == "affine":
        wm = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif warp_mode == "homography":
        wm = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        raise ValueError("warp_mode must be one of: translation, euclidean, affine, homography")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray_f, mov_gray_f, warp_matrix, wm, criteria, None, 5
        )
    except cv2.error as e:
        print(f"[WARN] ECC alignment failed ({warp_mode}): {e}. Returning unaligned image.")
        return mov_bgr

    h, w = ref_bgr.shape[:2]
    if wm == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(mov_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(mov_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned

def compute_psnr(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    mse = float(np.mean((img1_bgr.astype(np.float32) - img2_bgr.astype(np.float32)) ** 2))
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20.0 * float(np.log10(PIXEL_MAX / np.sqrt(mse + 1e-12)))

def compute_ssim_gray(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    g1 = to_gray(img1_bgr)
    g2 = to_gray(img2_bgr)
    if SKIMAGE_SSIM:
        return float(ssim(g1, g2, data_range=255))
    # Fallback simplified SSIM
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = float(g1.mean())
    mu2 = float(g2.mean())
    sigma1_sq = float(g1.var())
    sigma2_sq = float(g2.var())
    sigma12 = float(((g1 - mu1) * (g2 - mu2)).mean())
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(ssim_val)


# ------------- Red Masking (Enhanced) -------------
def make_red_mask_hsv(
    img_bgr: np.ndarray,
    min_sat: int = 60,
    min_val: int = 50,
    red_hue_width: int = 12,
    morph_kernel: int = 3,
    min_region_area: int = 30,
    roi_mask: Optional[np.ndarray] = None,
    r_delta: int = 15
) -> np.ndarray:
    """
    Robust red detection using:
      - HSV red wrap-around windows (OpenCV hue in [0..179])
      - BGR absolute dominance (R > max(G,B) + r_delta)
      - BGR relative dominance (R/G and R/B > ratio_k)
      - Lab a* reinforcement (a* > a_thresh)
    Final mask is OR of the above cues, then morphology + min-area cleanup applied.
    Returns a binary mask (uint8 0/255).
    """

    # --- OPTIONAL mild CLAHE on V channel to improve separation for faint residue ---
    # (Keep this on by default; it's gentle and helps on dull images. Comment out if undesired.)
    hsv0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv0)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
    except Exception:
        hsv = hsv0  # fallback if CLAHE is unavailable

    # --- HSV red wrap-around windows ---
    lower1 = np.array([0,   min_sat, min_val], dtype=np.uint8)
    upper1 = np.array([max(0, red_hue_width), 255, 255], dtype=np.uint8)
    lower2 = np.array([max(0, 179 - red_hue_width), min_sat, min_val], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # --- BGR dominance (absolute + ratio) ---
    b, g, r = cv2.split(img_bgr)

    # Absolute gate (original idea): R > max(G,B) + r_delta
    mgb = np.maximum(b, g).astype(np.int16)
    r16 = r.astype(np.int16)
    abs_gate = (r16 - mgb) > int(r_delta)

    # Relative dominance: R/G and R/B > ratio_k (helps under low exposure)
    eps = 1e-6
    r_f = r.astype(np.float32) + eps
    g_f = g.astype(np.float32) + eps
    b_f = b.astype(np.float32) + eps
    ratio_k = 1.12  # tune 1.07–1.15 (lower = more sensitive)
    ratio_gate = (r_f / g_f > ratio_k) & (r_f / b_f > ratio_k)

    mask_rb = np.where(abs_gate | ratio_gate, 255, 0).astype(np.uint8)

    # Combine with HSV via OR to improve recall
    mask = cv2.bitwise_or(mask, mask_rb)

    # --- Lab a* reinforcement for dull/brownish reds ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    # In OpenCV Lab, a* is [0..255] with ~128 neutral; higher means more red/magenta.
    a_thresh = 136  # tune 134–140 (lower = more sensitive)
    mask_lab = (a > a_thresh).astype(np.uint8) * 255

    # Combine Lab with existing mask
    mask = cv2.bitwise_or(mask, mask_lab)

    # Apply ROI if provided
    if roi_mask is not None:
        if roi_mask.shape[:2] != img_bgr.shape[:2]:
            roi_mask = cv2.resize(roi_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, roi_mask)

    # Morphological cleanup
    if morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_region_area:
            cleaned[labels == i] = 255
    return cleaned


# ------------- Overlays / Debug -------------
def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.45) -> np.ndarray:
    """
    Blend a solid color onto img_bgr wherever mask>0.
    Avoids calling OpenCV on boolean-indexed arrays (which can return None).
    """
    if img_bgr is None or mask is None:
        raise ValueError("overlay_mask received None for img_bgr or mask")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"overlay_mask expects BGR image with 3 channels, got shape {img_bgr.shape}")
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = (mask > 0)
    if not np.any(mask_bool):
        return img_bgr.copy()

    colored = np.zeros_like(img_bgr, dtype=img_bgr.dtype)
    colored[:] = color
    blended_full = cv2.addWeighted(img_bgr, 1.0 - alpha, colored, alpha, 0.0)
    if blended_full is None:
        out = img_bgr.copy()
        col = np.array(color, dtype=np.float32)
        out[mask_bool] = (
            img_bgr[mask_bool].astype(np.float32) * (1.0 - alpha) + col * alpha
        ).clip(0, 255).astype(np.uint8)
        return out
    out = img_bgr.copy()
    out[mask_bool] = blended_full[mask_bool]
    return out

def diff_heatmap(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(to_gray(img1_bgr), to_gray(img2_bgr))
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return heat

def logical_and_not(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a & (~b), inputs are 0/255 uint8
    return cv2.bitwise_and(a, cv2.bitwise_not(b))


# ------------- ROI -------------
def load_roi_mask(mask_path: Optional[str], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read ROI mask: {mask_path}")
    m = resize_to_match(m, target_shape)
    _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m_bin


# ------------- Quick red area for auto-role -------------
def _imread_downscale(path: str, max_side: int = 800) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def quick_red_area(
    path: str,
    min_sat: int,
    min_val: int,
    red_hue_width: int,
    morph_kernel: int,
    min_region_area: int,
    r_delta: int
) -> int:
    img = _imread_downscale(path, max_side=800)
    mask = make_red_mask_hsv(
        img,
        min_sat=min_sat,
        min_val=min_val,
        red_hue_width=red_hue_width,
        morph_kernel=morph_kernel,
        min_region_area=min_region_area,
        roi_mask=None,
        r_delta=r_delta
    )
    return int(np.count_nonzero(mask))


# ------------- Auto-role assignment for GUI multi -------------
def name_role_scores(fname_lower: str) -> Dict[str, int]:
    scores = {"before": 0, "after_flux": 0, "after_clean": 0}
    before_keys = ["before", "pre-flux", "pre_flux", "pre flux", "noflux", "no_flux", "baseline", "preclean"]
    flux_keys   = ["after_flux", "after-flux", "with_flux", "fluxed", "flux", "apply", "applied"]
    clean_keys  = ["after_clean", "after-clean", "postclean", "cleaned", "rinsed", "washed", "post-clean"]
    if any(k in fname_lower for k in before_keys):
        scores["before"] += 2
    if any(k in fname_lower for k in flux_keys):
        scores["after_flux"] += 2
    if any(k in fname_lower for k in clean_keys):
        scores["after_clean"] += 2
    return scores

def format_mapping_msg(mapping: Dict[str, str]) -> str:
    import os as _os
    return (
        "Auto-detected mapping:\n\n"
        f"  BEFORE      → {_os.path.basename(mapping['before'])}\n"
        f"  AFTER-FLUX  → {_os.path.basename(mapping['after_flux'])}\n"
        f"  AFTER-CLEAN → {_os.path.basename(mapping['after_clean'])}\n\n"
        "Proceed with this mapping?"
    )

def auto_assign_roles_from_three(files: List[str], args) -> Dict[str, str]:
    areas = [quick_red_area(f, args.min_sat, args.min_val, args.red_hue_width, args.morph_kernel, args.min_region_area, args.r_delta)
             for f in files]
    idx_flux = int(np.argmax(areas))
    flux_file = files[idx_flux]

    remain_idx = [i for i in range(3) if i != idx_flux]
    a_idx, b_idx = remain_idx[0], remain_idx[1]
    a_file, b_file = files[a_idx], files[b_idx]

    area_a, area_b = areas[a_idx], areas[b_idx]
    area_is_lower_a = int(area_a <= area_b)
    area_is_lower_b = int(area_b < area_a)

    scores_a = name_role_scores(os.path.basename(a_file).lower())
    scores_b = name_role_scores(os.path.basename(b_file).lower())

    try:
        mtime_a = os.path.getmtime(a_file)
        mtime_b = os.path.getmtime(b_file)
        a_is_earlier = int(mtime_a <= mtime_b)
        b_is_earlier = int(mtime_b < mtime_a)
        a_is_later = int(mtime_a >= mtime_b)
        b_is_later = int(mtime_b >= mtime_a)
    except Exception:
        a_is_earlier = b_is_earlier = a_is_later = b_is_later = 0

    before_score_a = 2 * scores_a["before"] + a_is_earlier + area_is_lower_a
    before_score_b = 2 * scores_b["before"] + b_is_earlier + area_is_lower_b
    clean_score_a  = 2 * scores_a["after_clean"] + a_is_later
    clean_score_b  = 2 * scores_b["after_clean"] + b_is_later

    if before_score_a > before_score_b:
        before_file = a_file; after_clean_file = b_file
    elif before_score_b > before_score_a:
        before_file = b_file; after_clean_file = a_file
    else:
        if clean_score_a > clean_score_b:
            after_clean_file = a_file; before_file = b_file
        elif clean_score_b > clean_score_a:
            after_clean_file = b_file; before_file = a_file
        else:
            if a_is_earlier >= b_is_earlier:
                before_file, after_clean_file = a_file, b_file
            else:
                before_file, after_clean_file = b_file, a_file

    return {"before": before_file, "after_flux": flux_file, "after_clean": after_clean_file}


# ------------- Save / Reporting -------------
def save_outputs(out_dir: str, inputs: Dict[str, str], results: Dict, save_debug: bool = False, export_masks: bool = False):
    ensure_dir(out_dir)

    # Save overlays
    cv2.imwrite(os.path.join(out_dir, "overlay_after_flux.png"),  results["overlays"]["after_flux_overlay"])
    cv2.imwrite(os.path.join(out_dir, "overlay_after_clean.png"), results["overlays"]["after_clean_overlay"])
    cv2.imwrite(os.path.join(out_dir, "diff_heatmap_before_vs_after_clean.png"), results["overlays"]["diff_heatmap"])
    cv2.imwrite(os.path.join(out_dir, "overlay_removed_area.png"), results["overlays"]["removed_overlay"])

    # Debug saves
    if save_debug or export_masks:
        if "debug" in results and isinstance(results["debug"], dict):
            dbg = results["debug"]
            if dbg.get("mask_before") is not None:
                cv2.imwrite(os.path.join(out_dir, "mask_before_debug.png"), dbg["mask_before"])
            if dbg.get("overlay_before") is not None:
                cv2.imwrite(os.path.join(out_dir, "overlay_before_debug.png"), dbg["overlay_before"])
            if dbg.get("mask_flux") is not None:
                cv2.imwrite(os.path.join(out_dir, "mask_after_flux_debug.png"), dbg["mask_flux"])
            if dbg.get("mask_residual") is not None:
                cv2.imwrite(os.path.join(out_dir, "mask_after_clean_debug.png"), dbg["mask_residual"])
            if dbg.get("mask_removed") is not None:
                cv2.imwrite(os.path.join(out_dir, "mask_removed_debug.png"), dbg["mask_removed"])

    # Save JSON (convert to json-able types)
    out_json = {k: v for k, v in results.items() if k != "overlays"}
    out_json["inputs"] = inputs
    out_json = to_jsonable(out_json)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # Append CSV row
    csv_path = os.path.join(out_dir, "metrics.csv")
    header = [
        "before", "after_flux", "after_clean",
        "flux_area_after_apply_px",
        "flux_area_after_clean_px",
        "flux_area_removed_px",
        "flux_area_reduction_pct",
        "ssim_before_vs_after_clean",
        "psnr_before_vs_after_clean",
        "roi_pixels",
        "pass_gate"
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            inputs["before"],
            inputs["after_flux"],
            inputs["after_clean"],
            int(results["flux_area_after_apply_px"]),
            int(results["flux_area_after_clean_px"]),
            int(results["flux_area_removed_px"]),
            f"{float(results['flux_area_reduction_pct']):.3f}",
            f"{float(results['ssim_before_vs_after_clean']):.5f}",
            f"{float(results['psnr_before_vs_after_clean']):.2f}",
            results["roi_pixels"] if results["roi_pixels"] is not None else "",
            int(bool(results["pass_gate"]))
        ])

def print_console_report(results: Dict, out_dir: str):
    print("\n=== FLUX REMOVAL REPORT ===")
    print(f"Flux pixels after apply:{int(results['flux_area_after_apply_px']):>11d}")
    print(f"Flux pixels after clean:{int(results['flux_area_after_clean_px']):>11d}")
    print(f"Flux pixels removed:    {int(results['flux_area_removed_px']):>11d}")
    print(f"Flux area reduction:{float(results['flux_area_reduction_pct']):>13.3f}%")
    print(f"SSIM (before vs after-clean): {float(results['ssim_before_vs_after_clean']):.5f}")
    print(f"PSNR (before vs after-clean): {float(results['psnr_before_vs_after_clean']):.2f} dB")
    print(f"PASS gate:{'PASS' if results['pass_gate'] else 'FAIL':>22s}")
    print(f"Outputs in:{out_dir:>23s}")
    print("============================\n")


# ------------- Core Pipeline -------------
def compute_metrics(
    before_bgr: np.ndarray,
    after_flux_bgr: np.ndarray,
    after_clean_bgr: np.ndarray,
    args
) -> Dict:
    # Optional image alignment
    if args.align:
        print("[INFO] Aligning images to 'before' reference...")
        after_flux_bgr = align_to_reference(before_bgr, after_flux_bgr, warp_mode=args.warp_mode)
        after_clean_bgr = align_to_reference(before_bgr, after_clean_bgr, warp_mode=args.warp_mode)

    # ROI (limit analysis to die area)
    roi_mask = load_roi_mask(args.roi_mask, before_bgr.shape) if args.roi_mask else None

    # Build masks
    print("[INFO] Building red masks...")
    # BEFORE debug mask for sanity
    mask_before = None
    overlay_before = None
    if args.save_debug:
        mask_before = make_red_mask_hsv(
            before_bgr,
            min_sat=args.min_sat,
            min_val=args.min_val,
            red_hue_width=args.red_hue_width,
            morph_kernel=args.morph_kernel,
            min_region_area=args.min_region_area,
            roi_mask=roi_mask,
            r_delta=args.r_delta
        )
        overlay_before = overlay_mask(before_bgr, mask_before, color=(0, 0, 255), alpha=0.45)

    # AFTER-FLUX
    mask_flux = make_red_mask_hsv(
        after_flux_bgr,
        min_sat=args.min_sat,
        min_val=args.min_val,
        red_hue_width=args.red_hue_width,
        morph_kernel=args.morph_kernel,
        min_region_area=args.min_region_area,
        roi_mask=roi_mask,
        r_delta=args.r_delta
    )

    # AFTER-CLEAN (optionally different min-region-area + smaller morphology for sensitivity)
    mra_residual = args.min_region_area_residual if args.min_region_area_residual is not None else args.min_region_area
    mk_res = max(1, min(args.morph_kernel, 2)) if args.preset == "sensitive" else args.morph_kernel
    mask_residual = make_red_mask_hsv(
        after_clean_bgr,
        min_sat=args.min_sat,
        min_val=args.min_val,
        red_hue_width=args.red_hue_width,
        morph_kernel=mk_res,  # smaller for residual to preserve tiny islands
        min_region_area=mra_residual,
        roi_mask=roi_mask,
        r_delta=args.r_delta
    )

    # Counts
    flux_area = int(np.count_nonzero(mask_flux))
    residual_area = int(np.count_nonzero(mask_residual))
    removed_mask = logical_and_not(mask_flux, mask_residual)
    removed_area = int(np.count_nonzero(removed_mask))

    reduction = float(0.0 if flux_area <= 0 else 100.0 * max(0, (flux_area - residual_area)) / float(flux_area))
    # Image similarity QA
    if roi_mask is not None:
        b = cv2.bitwise_and(before_bgr, before_bgr, mask=roi_mask)
        a = cv2.bitwise_and(after_clean_bgr, after_clean_bgr, mask=roi_mask)
        ssim_val = float(compute_ssim_gray(b, a))
    else:
        ssim_val = float(compute_ssim_gray(before_bgr, after_clean_bgr))

    psnr_val = float(compute_psnr(before_bgr, after_clean_bgr))

    # Overlays (AFTER-CLEAN uses cyan with lower alpha for visibility)
    overlays = {
        "after_flux_overlay": overlay_mask(after_flux_bgr, mask_flux,    color=(0, 0, 255), alpha=0.45),
        "after_clean_overlay": overlay_mask(after_clean_bgr, mask_residual, color=(0, 0, 255), alpha=0.30),
        "diff_heatmap":        diff_heatmap(before_bgr, after_clean_bgr),
        "removed_overlay":     overlay_mask(after_flux_bgr, removed_mask, color=(0, 255, 255), alpha=0.45)
    }

    # Pass/fail gate (tune to your spec)
    pass_gate = bool((reduction >= float(args.pass_removal_pct)) and (ssim_val >= float(args.pass_ssim_min)))

    result = {
        "flux_area_after_apply_px": int(flux_area),
        "flux_area_after_clean_px": int(residual_area),
        "flux_area_removed_px": int(removed_area),
        "flux_area_reduction_pct": float(reduction),
        "ssim_before_vs_after_clean": float(ssim_val),
        "psnr_before_vs_after_clean": float(psnr_val),
        "pass_gate": pass_gate,
        "overlays": overlays,
        "roi_pixels": int(np.count_nonzero(roi_mask)) if roi_mask is not None else None,
        "debug": {
            "mask_before": mask_before if args.save_debug else None,
            "overlay_before": overlay_before if args.save_debug else None,
            "mask_flux": mask_flux if (args.save_debug or args.export_masks) else None,
            "mask_residual": mask_residual if (args.save_debug or args.export_masks) else None,
            "mask_removed": removed_mask if (args.save_debug or args.export_masks) else None
        },
        "params": {
            "min_sat": args.min_sat,
            "min_val": args.min_val,
            "red_hue_width": args.red_hue_width,
            "morph_kernel": args.morph_kernel,
            "min_region_area": args.min_region_area,
            "min_region_area_residual": args.min_region_area_residual,
            "r_delta": args.r_delta,
            "align": bool(args.align),
            "warp_mode": args.warp_mode,
            "preset": args.preset
        }
    }
    return result


# ------------- Auto-Sweep -------------
def auto_sweep_suggestions(
    before_bgr: np.ndarray,
    after_clean_bgr: np.ndarray,
    roi_mask: Optional[np.ndarray],
    out_dir: str,
    sweep_min_sat: List[int],
    sweep_min_val: List[int],
    sweep_red_hue_width: List[int],
    sweep_r_delta: List[int],
    sweep_mra_residual: List[int],
    sweep_morph_kernel: List[int],
    sweep_top_k: int,
    sweep_before_max_px: int,
    sweep_score_wbefore: float,
    sweep_max_combos: int
):
    """
    Try combinations of thresholds to find parameter sets that:
    - detect residual pixels on AFTER-CLEAN
    - keep BEFORE nearly clean
    Saves sweep_summary.csv, top-K overlays/masks, and CLI suggestions.
    """
    ensure_dir(out_dir)
    combos = list(product(
        sweep_min_sat,
        sweep_min_val,
        sweep_red_hue_width,
        sweep_r_delta,
        sweep_mra_residual,
        sweep_morph_kernel
    ))
    if len(combos) > sweep_max_combos:
        print(f"[WARN] Sweep has {len(combos)} combos; limiting to first {sweep_max_combos}. Adjust lists or --sweep-max-combos.")
        combos = combos[:sweep_max_combos]

    rows = []
    candidates: List[Dict[str, Any]] = []

    print(f"[INFO] Auto-sweep: evaluating {len(combos)} parameter combinations...")
    for (msat, mval, rhw, rdel, mra_res, mk) in combos:
        # BEFORE mask and count
        mask_before = make_red_mask_hsv(
            before_bgr, min_sat=msat, min_val=mval, red_hue_width=rhw,
            morph_kernel=mk, min_region_area=mra_res, roi_mask=roi_mask, r_delta=rdel
        )
        before_px = int(np.count_nonzero(mask_before))

        # Default values each loop
        res_px = 0
        if before_px > sweep_before_max_px:
            score = -1e9  # reject heavy false positives on BEFORE
        else:
            # AFTER-CLEAN mask and count
            mask_res = make_red_mask_hsv(
                after_clean_bgr, min_sat=msat, min_val=mval, red_hue_width=rhw,
                morph_kernel=mk, min_region_area=mra_res, roi_mask=roi_mask, r_delta=rdel
            )
            res_px = int(np.count_nonzero(mask_res))
            score = float(res_px - sweep_score_wbefore * before_px)

        rows.append([msat, mval, rhw, rdel, mra_res, mk, before_px, res_px, score])

        if before_px <= sweep_before_max_px:
            candidates.append({
                "params": {
                    "min_sat": msat,
                    "min_val": mval,
                    "red_hue_width": rhw,
                    "r_delta": rdel,
                    "min_region_area_residual": mra_res,
                    "morph_kernel": mk
                },
                "before_px": before_px,
                "residual_px": res_px,
                "score": score
            })

    # Save sweep summary CSV
    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "min_sat", "min_val", "red_hue_width", "r_delta",
            "min_region_area_residual", "morph_kernel",
            "before_px", "residual_px", "score"
        ])
        for r in rows:
            w.writerow(r)
    print(f"[INFO] Saved sweep summary: {csv_path}")

    if not candidates:
        print("[WARN] No candidate passed BEFORE threshold. Consider increasing --sweep-before-max-px or reducing --sweep-score-wbefore.")
        return

    # Rank: score desc, then residual_px desc, then BEFORE lower first (tie-break)
    candidates.sort(key=lambda d: (d["score"], d["residual_px"], -d["before_px"]), reverse=True)
    top = candidates[:sweep_top_k]

    # Save Top-K suggestions as a reusable CLI snippet file
    sug_txt = os.path.join(out_dir, "sweep_top_suggestions.txt")
    with open(sug_txt, "w") as f:
        for i, c in enumerate(top, start=1):
            p = c["params"]
            f.write(
                f"# Rank {i}: score={c['score']:.1f}, residual_px={c['residual_px']}, before_px={c['before_px']}\n"
                f"--min-sat {p['min_sat']} --min-val {p['min_val']} --red-hue-width {p['red_hue_width']} "
                f"--r-delta {p['r_delta']} --min-region-area-residual {p['min_region_area_residual']} "
                f"--morph-kernel {p['morph_kernel']}\n\n"
            )
    print(f"[INFO] Saved Top-{sweep_top_k} suggestions: {sug_txt}")

    # Visualize Top-K masks/overlays for AFTER-CLEAN and BEFORE
    for i, c in enumerate(top, start=1):
        p = c["params"]

        # AFTER-CLEAN visualization
        mask_res = make_red_mask_hsv(
            after_clean_bgr,
            min_sat=p["min_sat"], min_val=p["min_val"], red_hue_width=p["red_hue_width"],
            morph_kernel=p["morph_kernel"], min_region_area=p["min_region_area_residual"],
            roi_mask=roi_mask, r_delta=p["r_delta"]
        )
        ov_res = overlay_mask(after_clean_bgr, mask_res, color=(0, 255, 255), alpha=0.30)
        cv2.imwrite(os.path.join(out_dir, f"mask_after_clean_sweep_rank{i}.png"), mask_res)
        cv2.imwrite(os.path.join(out_dir, f"overlay_after_clean_sweep_rank{i}.png"), ov_res)

        # BEFORE sanity (should be near-empty)
        mask_bef = make_red_mask_hsv(
            before_bgr,
            min_sat=p["min_sat"], min_val=p["min_val"], red_hue_width=p["red_hue_width"],
            morph_kernel=p["morph_kernel"], min_region_area=p["min_region_area_residual"],
            roi_mask=roi_mask, r_delta=p["r_delta"]
        )
        if np.count_nonzero(mask_bef) > 0:
            ov_bef = overlay_mask(before_bgr, mask_bef, color=(0, 255, 255), alpha=0.30)
            cv2.imwrite(os.path.join(out_dir, f"overlay_before_sweep_rank{i}.png"), ov_bef)


# ------------- Noise Floor Helper -------------
def compute_noise_floor(residual_images: List[str], args) -> Dict[str, float]:
    """
    Compute repeatability on a single cleaned sample imaged multiple times:
    returns mean and stdev of residual pixel counts.
    """
    if len(residual_images) < 3:
        raise ValueError("Need at least 3 images to estimate noise floor.")
    counts = []
    for p in residual_images:
        img = load_img(p)
        roi_mask = load_roi_mask(args.roi_mask, img.shape) if args.roi_mask else None
        mask = make_red_mask_hsv(
            img,
            min_sat=args.min_sat,
            min_val=args.min_val,
            red_hue_width=args.red_hue_width,
            morph_kernel=args.morph_kernel,
            min_region_area=args.min_region_area,
            roi_mask=roi_mask,
            r_delta=args.r_delta
        )
        counts.append(int(np.count_nonzero(mask)))
    counts = np.array(counts, dtype=np.float64)
    return {
        "runs": len(counts),
        "mean_residual_px": float(np.mean(counts)),
        "std_residual_px": float(np.std(counts, ddof=1 if len(counts) > 1 else 0)),
        "min_residual_px": int(np.min(counts)),
        "max_residual_px": int(np.max(counts))
    }


# ------------- CLI / Main -------------
def parse_args():
    p = argparse.ArgumentParser(description="Quantify red flux removal from microscope images (with internal-review aids + auto-sweep).")
    p.add_argument("--before", help="Path to BEFORE image (no flux).")
    p.add_argument("--after-flux", help="Path to AFTER-FLUX image (flux applied).")
    p.add_argument("--after-clean", help="Path to AFTER-CLEAN image (post-clean).")
    p.add_argument("--roi-mask", default=None, help="Optional ROI mask image (white=analyze, black=ignore). Resized to BEFORE image size.")
    p.add_argument("--out-dir", default="flux_results", help="Output directory.")

    # Red detection parameters
    p.add_argument("--min-sat", type=int, default=60, help="Min saturation for red pixels (HSV).")
    p.add_argument("--min-val", type=int, default=50, help="Min value/brightness for red pixels (HSV).")
    p.add_argument("--red-hue-width", type=int, default=12, help="Hue half-width near 0/179 (OpenCV hue 0-179).")
    p.add_argument("--morph-kernel", type=int, default=3, help="Morphological kernel size for mask cleanup.")
    p.add_argument("--min-region-area", type=int, default=30, help="Minimum connected component area (pixels) to keep in mask.")
    p.add_argument("--min-region-area-residual", type=int, default=None, help="Override min-region-area for AFTER-CLEAN residual mask only.")
    p.add_argument("--r-delta", type=int, default=15, help="Red dominance delta in BGR: R > max(G,B) + delta.")

    # Alignment
    p.add_argument("--align", action="store_true", help="Enable ECC alignment to BEFORE image.")
    p.add_argument("--warp-mode", default="affine", choices=["translation", "euclidean", "affine", "homography"], help="ECC warp model.")

    # Pass/fail gates
    p.add_argument("--pass-removal-pct", type=float, default=99.0, help="Minimum % flux removal to PASS.")
    p.add_argument("--pass-ssim-min", type=float, default=0.985, help="Minimum SSIM between BEFORE and AFTER-CLEAN to PASS.")

    # GUI
    p.add_argument("--gui", action="store_true", help="Open step-by-step file dialogs to select images and output folder.")
    p.add_argument("--gui-multi", action="store_true", help="Open one dialog to pick 3 images; auto-assign roles (confirmable).")
    p.add_argument("--no-confirm", action="store_true", help="Skip confirmation dialogs in GUI multi-select.")

    # Debug & presets
    p.add_argument("--save-debug", action="store_true", help="Save BEFORE overlay and raw masks for internal review.")
    p.add_argument("--export-masks", action="store_true", help="Export raw binary masks as PNGs.")
    p.add_argument("--preset", choices=["sensitive", "conservative"], default=None, help="Quick parameter presets.")

    # Noise floor
    p.add_argument("--noise-floor", nargs="+", help="Compute noise floor from multiple AFTER-CLEAN images (paths).")

    # Auto-sweep controls
    p.add_argument("--auto-sweep", action="store_true", help="Run parameter auto-sweep to suggest Top-K threshold sets.")
    p.add_argument("--sweep-min-sat", type=str, default=None, help="Comma list, e.g., '45,50,55,60'.")
    p.add_argument("--sweep-min-val", type=str, default=None, help="Comma list, e.g., '40,45,50'.")
    p.add_argument("--sweep-red-hue-width", type=str, default=None, help="Comma list, e.g., '12,14,16'.")
    p.add_argument("--sweep-r-delta", type=str, default=None, help="Comma list, e.g., '8,10,12,15'.")
    p.add_argument("--sweep-mra-residual", type=str, default=None, help="Comma list, e.g., '10,15,20,30'.")
    p.add_argument("--sweep-morph-kernel", type=str, default=None, help="Comma list, e.g., '1,3'.")
    p.add_argument("--sweep-top-k", type=int, default=3, help="How many top suggestions to save.")
    p.add_argument("--sweep-before-max-px", type=int, default=30, help="Reject combos where BEFORE detects > this many pixels.")
    p.add_argument("--sweep-score-wbefore", type=float, default=5.0, help="Score penalty weight for BEFORE pixels.")
    p.add_argument("--sweep-max-combos", type=int, default=500, help="Safety cap on total combinations evaluated.")

    return p.parse_args()

def apply_preset(args):
    if args.preset == "sensitive":
        # More inclusive (captures faint residue)
        args.min_sat = min(args.min_sat, 50)
        args.min_val = min(args.min_val, 45)
        args.red_hue_width = max(args.red_hue_width, 16)
        args.morph_kernel = min(args.morph_kernel, 3)
        if args.min_region_area_residual is None:
            # smaller default to keep tiny islands
            args.min_region_area_residual = max(10, (args.min_region_area // 2))
    elif args.preset == "conservative":
        # More exclusive (avoids false positives)
        args.min_sat = max(args.min_sat, 70)
        args.min_val = max(args.min_val, 60)
        args.red_hue_width = min(args.red_hue_width, 12)
        args.morph_kernel = max(args.morph_kernel, 3)
        if args.min_region_area_residual is None:
            args.min_region_area_residual = max(args.min_region_area, 60)

def inputs_ready(a):
    return bool(a.before and a.after_flux and a.after_clean)

def gui_select_paths(default_out_dir: str):
    """Step-by-step picker: BEFORE → AFTER-FLUX → AFTER-CLEAN"""
    tk, filedialog, messagebox = try_import_tk()
    if tk is None:
        raise RuntimeError("tkinter is not available in this Python environment.")

    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
        ("All files", "*.*")
    ]

    messagebox.showinfo("Select image", "Select BEFORE image (no flux).")
    before = filedialog.askopenfilename(title="Select BEFORE image", filetypes=filetypes)
    if not before:
        raise SystemExit("No BEFORE image selected.")

    init_dir = os.path.dirname(before)

    messagebox.showinfo("Select image", "Select AFTER-FLUX image (flux applied).")
    after_flux = filedialog.askopenfilename(title="Select AFTER-FLUX image", filetypes=filetypes, initialdir=init_dir)
    if not after_flux:
        raise SystemExit("No AFTER-FLUX image selected.")

    messagebox.showinfo("Select image", "Select AFTER-CLEAN image (post-clean).")
    after_clean = filedialog.askopenfilename(title="Select AFTER-CLEAN image", filetypes=filetypes, initialdir=init_dir)
    if not after_clean:
        raise SystemExit("No AFTER-CLEAN image selected.")

    use_out = messagebox.askyesno(
        "Output folder",
        f"Use default output folder?\n\n{os.path.abspath(default_out_dir)}\n\nClick 'No' to choose a different folder."
    )
    if use_out:
        out_dir = default_out_dir
    else:
        chosen = filedialog.askdirectory(title="Select output folder", initialdir=init_dir)
        out_dir = chosen if chosen else default_out_dir

    root.update()
    root.destroy()

    return before, after_flux, after_clean, out_dir

def gui_select_paths_multi(default_out_dir: str, args):
    """One-dialog, multi-select (select exactly 3 files). Auto-assign roles and confirm."""
    tk, filedialog, messagebox = try_import_tk()
    if tk is None:
        raise RuntimeError("tkinter is not available in this Python environment.")

    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
        ("All files", "*.*")
    ]

    while True:
        files = filedialog.askopenfilenames(
            title="Select 3 images (BEFORE, AFTER-FLUX, AFTER-CLEAN)",
            filetypes=filetypes
        )
        if not files:
            raise SystemExit("No files selected.")
        files = list(files)
        if len(files) != 3:
            messagebox.showerror("Selection error", f"You selected {len(files)} files. Please select exactly 3.")
            continue
        break

    mapping = auto_assign_roles_from_three(files, args)

    if not args.no_confirm:
        ok = messagebox.askyesno("Confirm mapping", format_mapping_msg(mapping))
        if not ok:
            messagebox.showinfo("Fallback", "We'll do a step-by-step selection instead.")
            root.update()
            root.destroy()
            return gui_select_paths(default_out_dir)

    init_dir = os.path.dirname(files[0])
    use_out = messagebox.askyesno(
        "Output folder",
        f"Use default output folder?\n\n{os.path.abspath(default_out_dir)}\n\nClick 'No' to choose a different folder."
    )
    if use_out:
        out_dir = default_out_dir
    else:
        chosen = filedialog.askdirectory(title="Select output folder", initialdir=init_dir)
        out_dir = chosen if chosen else default_out_dir

    root.update()
    root.destroy()

    return mapping["before"], mapping["after_flux"], mapping["after_clean"], out_dir

def main():
    args = parse_args()
    apply_preset(args)

    # Noise floor only mode (if provided and no normal inputs/gui requested)
    if args.noise_floor and not (args.gui or args.gui_multi or inputs_ready(args)):
        stats = compute_noise_floor(args.noise_floor, args)
        print("\n=== NOISE FLOOR (RESIDUAL PIXELS) ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("=====================================\n")
        return

    # Launch GUI if requested or if paths are missing
    if args.gui_multi:
        try:
            b, f, c, out_dir = gui_select_paths_multi(args.out_dir, args)
            args.before, args.after_flux, args.after_clean, args.out_dir = b, f, c, out_dir
            print(f"[INFO] Using GUI multi-select mapping:\n  BEFORE:      {args.before}\n  AFTER-FLUX:  {args.after_flux}\n  AFTER-CLEAN: {args.after_clean}\n  OUT:         {args.out_dir}")
        except RuntimeError as e:
            print(f"[WARN] {e}")
            if not inputs_ready(args):
                raise SystemExit("tkinter not available and paths not provided. Supply --before/--after-flux/--after-clean.")
    elif args.gui or not inputs_ready(args):
        try:
            b, f, c, out_dir = gui_select_paths(args.out_dir)
            args.before, args.after_flux, args.after_clean, args.out_dir = b, f, c, out_dir
            print(f"[INFO] Using GUI selections:\n  BEFORE:      {args.before}\n  AFTER-FLUX:  {args.after_flux}\n  AFTER-CLEAN: {args.after_clean}\n  OUT:         {args.out_dir}")
        except RuntimeError as e:
            print(f"[WARN] {e}")
            if not inputs_ready(args):
                raise SystemExit("tkinter not available and required paths not provided. Run with --before/--after-flux/--after-clean or install tkinter.")

    ensure_dir(args.out_dir)

    # Load images
    before = load_img(args.before)
    after_flux = load_img(args.after_flux)
    after_clean = load_img(args.after_clean)

    # Ensure same size
    h, w = before.shape[:2]
    if after_flux.shape[:2] != (h, w):
        after_flux = resize_to_match(after_flux, before.shape)
    if after_clean.shape[:2] != (h, w):
        after_clean = resize_to_match(after_clean, before.shape)

    # Optional alignment (also used by sweep)
    if args.align:
        print("[INFO] Aligning AFTER images to BEFORE for both compute and sweep...")
        after_flux = align_to_reference(before, after_flux, warp_mode=args.warp_mode)
        after_clean = align_to_reference(before, after_clean, warp_mode=args.warp_mode)

    # ROI mask
    roi_mask = load_roi_mask(args.roi_mask, before.shape) if args.roi_mask else None

    # ---- Auto-Sweep (optional) ----
    if args.auto_sweep:
        sweep_min_sat = parse_int_list(args.sweep_min_sat, default=[45, 50, 55, 60])
        sweep_min_val = parse_int_list(args.sweep_min_val, default=[40, 45, 50])
        sweep_red_hue_width = parse_int_list(args.sweep_red_hue_width, default=[12, 14, 16])
        sweep_r_delta = parse_int_list(args.sweep_r_delta, default=[8, 10, 12, 15])
        sweep_mra_residual = parse_int_list(args.sweep_mra_residual, default=[10, 15, 20, 30])
        sweep_morph_kernel = parse_int_list(args.sweep_morph_kernel, default=[1, 3])

        auto_sweep_suggestions(
            before_bgr=before,
            after_clean_bgr=after_clean,
            roi_mask=roi_mask,
            out_dir=args.out_dir,
            sweep_min_sat=sweep_min_sat,
            sweep_min_val=sweep_min_val,
            sweep_red_hue_width=sweep_red_hue_width,
            sweep_r_delta=sweep_r_delta,
            sweep_mra_residual=sweep_mra_residual,
            sweep_morph_kernel=sweep_morph_kernel,
            sweep_top_k=args.sweep_top_k,
            sweep_before_max_px=args.sweep_before_max_px,
            sweep_score_wbefore=args.sweep_score_wbefore,
            sweep_max_combos=args.sweep_max_combos
        )

    # ---- Standard compute & outputs ----
    results = compute_metrics(before, after_flux, after_clean, args)
    inputs = {"before": args.before, "after_flux": args.after_flux, "after_clean": args.after_clean}
    save_outputs(args.out_dir, inputs, results, save_debug=args.save_debug, export_masks=args.export_masks)
    print_console_report(results, args.out_dir)


if __name__ == "__main__":
    main()