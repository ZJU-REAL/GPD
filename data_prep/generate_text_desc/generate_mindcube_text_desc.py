#!/usr/bin/env python3
"""
MindCube text-description generation script.

Unlike SPAR/VSI (ScanNet 3D back-projection), MindCube uses:
  - Depth source  : DA3-estimated float32 .npy files (relative depth, no absolute metric)
  - Semantic source: Grounded-SAM-2 HSV pseudo-colour PNGs + inst_id->label JSON

Generates one JSON per image: <OUTPUT_DIR>/<category>/<object>/<stem>.json

JSON fields per object:
  label        : label string from Grounding DINO detection (e.g. "chair", "bottle")
  inst_id      : instance index (1-based)
  depth_rel    : median relative depth, normalised to [0, 1] (0=nearest, 1=farthest)
  cam_right    : normalised pixel centroid x (0=left edge, 1=right edge)
  cam_up       : normalised pixel centroid y (0=top, 1=bottom)
  grid_pos     : 3x3 grid position, e.g. "left-top", "center", "right-bottom"
  coverage_pct : pixel coverage percentage of the instance in the frame

Prerequisites (run first):
  generate_mindcube_semantic.py  -> produces <SEMANTIC_DIR>/
  generate_mindcube_depth_bev.py -> produces <DEPTH_NPY_DIR>/

Run (any conda environment with numpy / Pillow):
  python3 -u generate_mindcube_text_desc.py 2>&1 | tee logs/mindcube_text_desc_gen.log

Environment variables (all optional; defaults shown below):
  MINDCUBE_JSONL       - JSONL annotation file for MindCube
  MINDCUBE_DEPTH_NPY   - directory containing DA3 depth .npy files
  MINDCUBE_SEMANTIC    - directory containing Grounded-SAM-2 semantic outputs
  MINDCUBE_TEXT_DESC   - output directory for text-desc JSONs
"""

import json
import os
import colorsys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==================== Configuration ====================
MINDCUBE_JSONL   = Path(os.environ.get("MINDCUBE_JSONL",     "/PATH/TO/WORKSPACE/data/mindcube_10k_grpo.jsonl"))
DEPTH_NPY_DIR    = Path(os.environ.get("MINDCUBE_DEPTH_NPY", "/PATH/TO/WORKSPACE/data/mindcube_depth_npy"))
SEMANTIC_DIR     = Path(os.environ.get("MINDCUBE_SEMANTIC",  "/PATH/TO/WORKSPACE/data/mindcube_semantic"))
OUTPUT_DIR       = Path(os.environ.get("MINDCUBE_TEXT_DESC", "/PATH/TO/WORKSPACE/data/mindcube_text_desc"))
MIN_COVERAGE_PCT = 0.5   # instances below this pixel-coverage threshold are ignored
NUM_WORKERS      = 8
# =======================================================


# ─────────────────────── HSV palette (must match generate_mindcube_semantic.py) ───

def _build_palette(n: int = 256) -> np.ndarray:
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette

PALETTE = _build_palette(256)   # shape (256, 3)


def rgb_to_inst_map(rgb: np.ndarray) -> np.ndarray:
    """
    Decode an HSV pseudo-colour PNG (H, W, 3) back to an instance-ID map (H, W).
    Background pixels map to 0.  Uses exact colour matching against PALETTE;
    only unique colours present in the image are looked up for speed.
    """
    h, w = rgb.shape[:2]
    # Pack RGB channels into a single integer key for fast lookup
    rgb_flat  = rgb.reshape(-1, 3).astype(np.int32)
    key_flat  = rgb_flat[:, 0] * 65536 + rgb_flat[:, 1] * 256 + rgb_flat[:, 2]

    palette_keys = (
        PALETTE[:, 0].astype(np.int32) * 65536
        + PALETTE[:, 1].astype(np.int32) * 256
        + PALETTE[:, 2].astype(np.int32)
    )  # shape (256,)

    inst_flat = np.zeros(len(key_flat), dtype=np.int32)
    for inst_id in range(1, 256):
        inst_flat[key_flat == palette_keys[inst_id]] = inst_id

    return inst_flat.reshape(h, w)


# ─────────────────────── Per-image processing ───────────────────────

def text_desc_needs_regenerate(
    out_path: Path,
    depth_path: Path,
    rgb_path: str,
) -> bool:
    """Return True if the output JSON needs to be (re-)generated."""
    if not out_path.is_file():
        return True
    if not depth_path.is_file() or not Path(rgb_path).is_file():
        return True

    rw, rh = Image.open(rgb_path).size
    depth = np.load(str(depth_path))
    if depth.shape != (rh, rw):
        return True

    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return True

    if isinstance(data, dict):
        return data.get("img_w") != rw or data.get("img_h") != rh
    return out_path.stat().st_mtime < depth_path.stat().st_mtime


def process_image(img_path: str, obj_key: str, stem: str) -> int:
    """
    Generate text-description JSON for a single MindCube image.

    Returns the number of instances written, -1 if skipped (already up to date),
    or -2 on failure.
    """
    out_path      = OUTPUT_DIR   / obj_key / f"{stem}.json"
    depth_path    = DEPTH_NPY_DIR / obj_key / f"{stem}.npy"
    sem_png_path  = SEMANTIC_DIR  / obj_key / f"{stem}.png"
    sem_json_path = SEMANTIC_DIR  / obj_key / f"{stem}.json"

    if not text_desc_needs_regenerate(out_path, depth_path, img_path):
        return -1   # already up to date; skip

    # Skip if any required input is missing
    if not depth_path.exists() or not sem_png_path.exists() or not sem_json_path.exists():
        return -2

    try:
        depth = np.load(str(depth_path))              # (H_d, W_d) float32, DA3 resolution
        with open(sem_json_path, encoding="utf-8") as f:
            id_to_label: dict[str, str] = json.load(f)   # {"1": "chair", ...}

        img_h, img_w = depth.shape   # use depth map size as the reference resolution

        # Semantic PNG is generated at original image resolution; resize to depth resolution
        # using NEAREST interpolation to preserve exact colour values
        sem_img = Image.open(str(sem_png_path)).convert("RGB")
        if sem_img.size != (img_w, img_h):   # PIL.size = (W, H)
            sem_img = sem_img.resize((img_w, img_h), Image.NEAREST)
        sem_rgb = np.array(sem_img)

        # Decode instance-ID map from pseudo-colour image
        inst_map = rgb_to_inst_map(sem_rgb)   # (H, W), int32

        # Global depth normalisation parameters (for depth_rel)
        valid_depth = depth[depth > 0]
        if valid_depth.size == 0:
            out_path.write_text("[]", encoding="utf-8")
            return 0
        d_min, d_max = float(valid_depth.min()), float(valid_depth.max())
        d_range = d_max - d_min if d_max > d_min else 1.0

        total_px = img_h * img_w
        stats: list[dict] = []

        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue   # background

            mask = (inst_map == inst_id)
            n_px = int(mask.sum())
            coverage_pct = n_px / total_px * 100.0
            if coverage_pct < MIN_COVERAGE_PCT:
                continue

            rows, cols = np.where(mask)

            # Sample DA3 depth values within the instance mask
            depth_vals = depth[rows, cols]
            valid = depth_vals > 0
            if valid.sum() == 0:
                median_depth_rel = 0.5   # use mid-range when no valid depth
            else:
                median_depth_rel = round(
                    float((np.median(depth_vals[valid]) - d_min) / d_range), 3
                )

            # Normalised pixel centroid
            cam_right = round(float(cols.mean()) / img_w, 3)
            cam_up    = round(float(rows.mean()) / img_h, 3)   # 0=top, 1=bottom

            # 3x3 grid position
            h_pos = "left"   if cam_right < 0.33 else ("right"  if cam_right > 0.67 else "center")
            v_pos = "top"    if cam_up    < 0.33 else ("bottom" if cam_up    > 0.67 else "middle")
            if h_pos == "center" and v_pos == "middle":
                grid_pos = "center"
            else:
                grid_pos = f"{h_pos}-{v_pos}"

            label = id_to_label.get(str(inst_id), f"object#{inst_id}")

            stats.append({
                "label":        label,
                "inst_id":      int(inst_id),
                "depth_rel":    median_depth_rel,
                "cam_right":    cam_right,
                "cam_up":       cam_up,
                "grid_pos":     grid_pos,
                "coverage_pct": round(coverage_pct, 2),
            })

        # Sort by relative depth, nearest first
        stats.sort(key=lambda x: x["depth_rel"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "img_w": img_w,
            "img_h": img_h,
            "objects": stats,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return len(stats)

    except Exception as e:
        print(f"\n  WARNING: {obj_key}/{stem}: {e}")
        return -2


# ─────────────────────── Main pipeline ───────────────────────

def main():
    global MINDCUBE_JSONL
    env_j = os.environ.get("MINDCUBE_JSONL")
    if env_j:
        MINDCUBE_JSONL = Path(env_j)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique images
    tasks: list[tuple[str, str, str]] = []   # (img_path, obj_key, stem)
    with open(MINDCUBE_JSONL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for img_path in s["image"]:
                parts   = img_path.split("/")
                obj_key = parts[-3] + "/" + parts[-2]   # e.g. among/bottle_118
                stem    = Path(img_path).stem            # e.g. front_000
                tasks.append((img_path, obj_key, stem))

    # Deduplicate (the same image may appear in multiple samples)
    seen = set()
    unique_tasks = []
    for t in tasks:
        key = (t[1], t[2])
        if key not in seen:
            seen.add(key)
            unique_tasks.append(t)

    print(f"Images to process: {len(unique_tasks)}")
    print(f"Output directory:  {OUTPUT_DIR}\n")

    done, skipped, failed, total_insts = 0, 0, 0, 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_image, img_path, obj_key, stem): (obj_key, stem)
            for img_path, obj_key, stem in unique_tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            n = fut.result()
            if n == -1:
                skipped += 1
            elif n == -2:
                failed += 1
            else:
                done += 1
                total_insts += n

    print(f"\nDone!")
    print(f"   Generated: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"   Total instance records: {total_insts}")


if __name__ == "__main__":
    main()
