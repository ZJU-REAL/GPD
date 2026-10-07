#!/usr/bin/env python3
"""
MindCube depth-map and bird's-eye-view (BEV) generation script.

Core idea:
  DA3 (DepthAnything3) can process multiple images simultaneously and produces
  multi-view-consistent depth maps + estimated camera intrinsics + extrinsics.
  This makes it well-suited for MindCube's 4-viewpoint setup
  (front / left / back / right).

For each group of 4 images belonging to one object:
  1. Run DA3 -> 4 depth maps -> inferno-coloured depth RGB JPGs
  2. Using intrinsics + extrinsics, back-project to 3D -> merge 4 point clouds
     -> top-down projection -> BEV PNG

Outputs:
  <DEPTH_DIR>/<category>/<object>/<stem>.jpg   (4 per object)
  <DEPTH_NPY_DIR>/<category>/<object>/<stem>.npy  (raw float32 depth, 4 per object)
  <BEV_DIR>/<category>/<object>.png            (1 per object)
  <CAM_DIR>/<category>/<object>.npz            (camera params, 1 per object)

Prerequisites:
  - DepthAnything3 source installed (set DA3_SRC to the src/ directory)
  - DA3 model weights downloaded (set DA3_MODEL_PATH)
  - Run in a conda environment with DA3, PyTorch, and matplotlib installed

Run:
  <your_conda_env>/bin/python3 generate_mindcube_depth_bev.py \\
      2>&1 | tee logs/mindcube_depth_bev_gen.log

Environment variables (all optional; defaults shown below):
  DA3_SRC          - path to DepthAnything3 src/ directory
  MINDCUBE_JSONL   - JSONL annotation file for MindCube
  DA3_MODEL_PATH   - path to DA3 model weights directory
  MINDCUBE_DEPTH   - output directory for coloured depth JPGs
  MINDCUBE_DEPTH_NPY - output directory for raw depth .npy files
  MINDCUBE_BEV     - output directory for BEV PNGs
  MINDCUBE_CAM     - output directory for camera parameter NPZs
"""

import sys
import os
import json
import colorsys
from pathlib import Path
from collections import defaultdict

# DA3 may be installed as an editable package; insert the src/ path explicitly
# in case the .pth file is not picked up automatically in all environments.
_DA3_SRC = os.environ.get("DA3_SRC", "/PATH/TO/Depth-Anything-3/src")
if _DA3_SRC not in sys.path:
    sys.path.insert(0, _DA3_SRC)

import numpy as np
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm

# ==================== Configuration ====================
MINDCUBE_JSONL  = Path(os.environ.get("MINDCUBE_JSONL",    "/PATH/TO/WORKSPACE/data/mindcube_10k_grpo.jsonl"))
DA3_MODEL_PATH  = os.environ.get("DA3_MODEL_PATH",         "/PATH/TO/MODEL/DA3NESTED-GIANT-LARGE-1.1")
DEPTH_DIR       = Path(os.environ.get("MINDCUBE_DEPTH",    "/PATH/TO/WORKSPACE/data/mindcube_depth_rgb"))
DEPTH_NPY_DIR   = Path(os.environ.get("MINDCUBE_DEPTH_NPY","/PATH/TO/WORKSPACE/data/mindcube_depth_npy"))
BEV_DIR         = Path(os.environ.get("MINDCUBE_BEV",      "/PATH/TO/WORKSPACE/data/mindcube_bev"))
CAM_DIR         = Path(os.environ.get("MINDCUBE_CAM",      "/PATH/TO/WORKSPACE/data/mindcube_cam"))
PROCESS_RES     = 504    # DA3 inference resolution
BEV_IMG_SIZE    = 512    # BEV output image side length
BEV_PERCENTILE  = 2      # percentile for clipping outlier points in BEV
MIN_DEPTH       = 0.05   # discard depth values below this threshold (normalised)
# =======================================================


def resize_depth_to_image(depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """NEAREST-resize a DA3 depth map to the original RGB image resolution."""
    h, w = depth.shape
    if w == target_w and h == target_h:
        return depth.astype(np.float32, copy=False)
    resized = np.array(
        Image.fromarray(depth.astype(np.float32)).resize(
            (target_w, target_h), Image.NEAREST
        )
    )
    return resized.astype(np.float32)


def depth_outputs_match_rgb(
    depth_jpg: Path,
    depth_npy: Path,
    target_w: int,
    target_h: int,
) -> bool:
    """Return True if both depth outputs exist and have the expected resolution."""
    if not depth_jpg.is_file() or not depth_npy.is_file():
        return False
    try:
        if Image.open(depth_jpg).size != (target_w, target_h):
            return False
        arr = np.load(str(depth_npy))
        return arr.shape == (target_h, target_w)
    except Exception:
        return False


# ─────────────────────── Utility functions ───────────────────────

def as_homogeneous_44(ext: np.ndarray) -> np.ndarray:
    """
    Pad or return an extrinsic matrix as a standard (4, 4) homogeneous matrix.
    DA3 may return (3, 4) when use_ray_pose=True; this function handles both cases.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=ext.dtype)
        return np.concatenate([ext, bottom], axis=0)
    raise ValueError(f"Expected shape (3,4) or (4,4), got {ext.shape}")


# ─────────────────────── Depth colouring ───────────────────────

def compute_global_depth_range(
    depths: np.ndarray | list[np.ndarray],
) -> tuple[float, float]:
    """
    Compute global valid depth min/max across multiple frames for consistent colouring.

    depths: (N, H, W) array or list of (H, W) arrays (frame dimensions may differ).
    """
    if isinstance(depths, list):
        arrays = depths
    elif isinstance(depths, np.ndarray) and depths.ndim == 3:
        arrays = [depths[i] for i in range(depths.shape[0])]
    else:
        arrays = [depths]

    d_min, d_max = None, None
    for d in arrays:
        valid = d[d > MIN_DEPTH]
        if valid.size == 0:
            continue
        lo, hi = float(valid.min()), float(valid.max())
        d_min = lo if d_min is None else min(d_min, lo)
        d_max = hi if d_max is None else max(d_max, hi)

    if d_min is None or d_max is None:
        return 0.0, 1.0
    return d_min, d_max


def depth_to_rgb(
    depth: np.ndarray,
    d_min: float,
    d_max: float,
    colormap: str = "inferno",
) -> np.ndarray:
    """
    Convert a single (H, W) depth map to a 3-channel uint8 RGB image.

    d_min / d_max should come from the global range across all frames in the
    same group to ensure consistent colour scaling across viewpoints.
    """
    d = depth.copy().astype(np.float32)
    if d_max > d_min:
        d = np.clip((d - d_min) / (d_max - d_min), 0, 1)
    else:
        d = np.zeros_like(d)

    cmap = cm.get_cmap(colormap)
    rgb  = (cmap(d)[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ─────────────────────── BEV generation ───────────────────────

def back_project_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    C2W: np.ndarray,
    stride: int = 4,
) -> np.ndarray:
    """
    Back-project a single depth map into world-space 3D points.

    stride: downsampling step for controlling point cloud density.
    Returns (N, 3) float32 world-space coordinates.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Downsampled coordinate grid
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[vv, uu].flatten()

    valid = z > MIN_DEPTH
    z   = z[valid]
    uu  = uu.flatten()[valid].astype(np.float32)
    vv  = vv.flatten()[valid].astype(np.float32)

    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pts_cam   = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N, 4)
    pts_world = (C2W @ pts_cam.T).T                            # (N, 4)
    return pts_world[:, :3].astype(np.float32)


def points_to_bev(pts: np.ndarray, img_size: int = BEV_IMG_SIZE) -> np.ndarray:
    """
    Project 3D point cloud to a top-down BEV image.

    In DA3's world coordinate system the camera roughly faces +Z;
    the horizontal plane is formed by X (right) and Y (down/forward).
    BEV uses the X-Z plane (ignoring Y-axis height variation) to produce
    an intuitive overhead view.
    """
    if pts.shape[0] == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    bev_x = pts[:, 0]
    bev_z = pts[:, 2]   # depth direction as BEV forward axis

    # Clip outliers
    x_lo, x_hi = np.percentile(bev_x, BEV_PERCENTILE), np.percentile(bev_x, 100 - BEV_PERCENTILE)
    z_lo, z_hi = np.percentile(bev_z, BEV_PERCENTILE), np.percentile(bev_z, 100 - BEV_PERCENTILE)

    mask = (bev_x >= x_lo) & (bev_x <= x_hi) & (bev_z >= z_lo) & (bev_z <= z_hi)
    bev_x = bev_x[mask]
    bev_z = bev_z[mask]
    y_h   = pts[mask, 1]   # height dimension (used for colouring)

    if bev_x.size == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Normalise to pixel coordinates
    px = ((bev_x - x_lo) / max(x_hi - x_lo, 1e-6) * (img_size - 1)).astype(np.int32)
    pz = ((bev_z - z_lo) / max(z_hi - z_lo, 1e-6) * (img_size - 1)).astype(np.int32)
    px = np.clip(px, 0, img_size - 1)
    pz = np.clip(pz, 0, img_size - 1)

    # Colour by height (low = dark, high = bright)
    y_norm = (y_h - y_h.min()) / max(y_h.max() - y_h.min(), 1e-6)
    colors = (cm.get_cmap("viridis")(y_norm)[:, :3] * 255).astype(np.uint8)

    bev = np.full((img_size, img_size, 3), 30, dtype=np.uint8)  # dark grey background
    # Draw far-to-near so nearer points overwrite farther ones
    order = np.argsort(bev_z)[::-1]
    bev[pz[order], px[order]] = colors[order]

    return bev


# ─────────────────────── Main pipeline ───────────────────────

def load_model():
    """Load the DA3 model (called once per run)."""
    import torch
    from depth_anything_3.api import DepthAnything3
    print(f"Loading DA3 model: {DA3_MODEL_PATH}")
    model = DepthAnything3.from_pretrained(DA3_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({device})")
    return model


def try_resize_existing_depth_outputs(
    img_paths: list[str],
    depth_paths: list[Path],
    depth_npy_paths: list[Path],
    rgb_sizes: list[tuple[int, int]],
) -> bool:
    """
    If DA3 .npy files already exist but their resolution does not match the RGB
    images, NEAREST-resize them in-place to avoid re-running DA3.
    Returns True if all outputs are now aligned (no DA3 inference needed).
    """
    if not all(npp.is_file() for npp in depth_npy_paths):
        return False

    resized: list[np.ndarray] = []
    for npp, (tw, th) in zip(depth_npy_paths, rgb_sizes):
        raw = np.load(str(npp))
        depth = resize_depth_to_image(raw, tw, th)
        if depth.shape != (th, tw):
            return False
        resized.append(depth)

    d_min, d_max = compute_global_depth_range(resized)
    for depth, depth_path, npy_path, (tw, th) in zip(
        resized, depth_paths, depth_npy_paths, rgb_sizes
    ):
        if depth_outputs_match_rgb(depth_path, npy_path, tw, th):
            continue
        np.save(str(npy_path), depth.astype(np.float32))
        rgb = depth_to_rgb(depth, d_min, d_max, colormap="inferno")
        Image.fromarray(rgb).save(str(depth_path), quality=95)

    return all(
        depth_outputs_match_rgb(dp, npp, tw, th)
        for dp, npp, (tw, th) in zip(depth_paths, depth_npy_paths, rgb_sizes)
    )


def process_object(model, obj_key: str, img_paths: list[str]):
    """
    Process one group of 4 images, generating depth maps and a BEV image.

    obj_key   : e.g. "among/bottle_118"
    img_paths : 4 absolute image paths (front/left/back/right)
    """
    # Output paths
    depth_out_dir   = DEPTH_DIR     / obj_key
    depth_npy_dir   = DEPTH_NPY_DIR / obj_key
    bev_out_path    = BEV_DIR       / f"{obj_key}.png"
    cam_npz_path    = CAM_DIR       / f"{obj_key}.npz"   # camera parameter NPZ

    depth_paths     = [depth_out_dir / (Path(p).stem + ".jpg") for p in img_paths]
    depth_npy_paths = [depth_npy_dir / (Path(p).stem + ".npy") for p in img_paths]

    rgb_sizes = [Image.open(p).size for p in img_paths]

    all_done = (
        all(
            depth_outputs_match_rgb(dp, npp, tw, th)
            for dp, npp, (tw, th) in zip(depth_paths, depth_npy_paths, rgb_sizes)
        )
        and bev_out_path.exists()
        and cam_npz_path.exists()
    )
    if all_done:
        return True

    depth_out_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)
    bev_out_path.parent.mkdir(parents=True, exist_ok=True)
    cam_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if bev_out_path.exists() and cam_npz_path.exists():
        try:
            if try_resize_existing_depth_outputs(
                img_paths, depth_paths, depth_npy_paths, rgb_sizes
            ):
                return True
        except Exception as e:
            print(f"  WARNING: depth resize failed for {obj_key}: {e}, re-running DA3")

    try:
        prediction = model.inference(
            image=img_paths,
            process_res=PROCESS_RES,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="mini_npz",   # retrieve depth + camera without writing files
        )
    except Exception as e:
        print(f"  WARNING: DA3 inference failed for {obj_key}: {e}")
        return False

    # Retrieve DA3 predictions
    depths   = prediction.depth      # (N, H, W)
    extrs    = prediction.extrinsics  # (N, 3|4, 4) World-to-Camera (W2C)
    intrs    = prediction.intrinsics  # (N, 3, 3)

    # Compute global depth range for consistent 4-frame colour scale
    d_min, d_max = compute_global_depth_range(depths)

    # Save camera parameter NPZ (for downstream novel-view synthesis use)
    if not cam_npz_path.exists() and extrs is not None and intrs is not None:
        c2ws_list = []
        for i in range(len(img_paths)):
            ext44 = as_homogeneous_44(extrs[i])
            c2w   = np.linalg.inv(ext44)          # W2C -> C2W
            c2ws_list.append(c2w.astype(np.float32))

        # Record original image sizes (W, H) so DA3's K can be rescaled to native coordinates
        img_whs = []
        for p in img_paths:
            try:
                w, h = Image.open(p).size
            except Exception:
                w, h = PROCESS_RES, PROCESS_RES  # fallback
            img_whs.append([w, h])

        np.savez(
            str(cam_npz_path),
            c2ws      = np.array(c2ws_list),                           # (N, 4, 4) OpenCV C2W
            Ks        = intrs.astype(np.float32),                      # (N, 3, 3) DA3 internal coords
            img_stems = np.array([Path(p).stem for p in img_paths]),   # (N,) filename stems
            img_whs   = np.array(img_whs, dtype=np.int32),             # (N, 2) original size [W, H]
        )

    all_pts_world = []

    for i, (img_path, depth_path) in enumerate(zip(img_paths, depth_paths)):
        depth_da3 = depths[i]
        orig_w, orig_h = rgb_sizes[i]
        depth = resize_depth_to_image(depth_da3, orig_w, orig_h)

        npy_path = depth_npy_paths[i]
        if not depth_outputs_match_rgb(depth_path, npy_path, orig_w, orig_h):
            np.save(str(npy_path), depth.astype(np.float32))
            rgb = depth_to_rgb(depth, d_min, d_max, colormap="inferno")
            Image.fromarray(rgb).save(str(depth_path), quality=95)

        # Back-project at DA3 resolution using DA3's camera parameters (consistent with SVC pipeline)
        if extrs is not None and intrs is not None:
            ext44 = as_homogeneous_44(extrs[i])
            c2w   = np.linalg.inv(ext44)
            pts   = back_project_to_world(depth_da3, intrs[i], c2w)
            if pts.shape[0] > 0:
                all_pts_world.append(pts)

    # Generate BEV image
    if not bev_out_path.exists():
        if all_pts_world:
            merged_pts = np.concatenate(all_pts_world, axis=0)
            bev_img    = points_to_bev(merged_pts)
        else:
            bev_img = np.zeros((BEV_IMG_SIZE, BEV_IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(bev_img).save(str(bev_out_path))

    return True


def main():
    global MINDCUBE_JSONL
    env_j = os.environ.get("MINDCUBE_JSONL")
    if env_j:
        MINDCUBE_JSONL = Path(env_j)

    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_NPY_DIR.mkdir(parents=True, exist_ok=True)
    BEV_DIR.mkdir(parents=True, exist_ok=True)
    CAM_DIR.mkdir(parents=True, exist_ok=True)

    # Collect unique object -> sorted 4-image-path mapping
    obj_images: dict[str, list[str]] = defaultdict(list)
    with open(MINDCUBE_JSONL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for img_path in s["image"]:
                parts   = img_path.split("/")
                obj_key = parts[-3] + "/" + parts[-2]   # e.g. among/bottle_118
                if img_path not in obj_images[obj_key]:
                    obj_images[obj_key].append(img_path)

    # Sort each group; support two naming conventions:
    #   front_000 / left_126  -> trailing number (angle format)
    #   1_frame / 2_frame     -> leading number (frame-index format)
    def _img_sort_key(p: str) -> int:
        parts = Path(p).stem.split("_")
        for candidate in [parts[-1], parts[0]]:
            try:
                return int(candidate)
            except ValueError:
                continue
        return 0

    for key in obj_images:
        obj_images[key].sort(key=_img_sort_key)

    obj_list = list(obj_images.items())
    print(f"Objects to process: {len(obj_list)} groups (4 images each)")
    print(f"Depth output:   {DEPTH_DIR}")
    print(f"BEV output:     {BEV_DIR}\n")

    model = load_model()

    done, failed = 0, 0
    for obj_key, img_paths in tqdm(obj_list, desc="Objects"):
        ok = process_object(model, obj_key, img_paths)
        if ok:
            done += 1
        else:
            failed += 1

    total_depth = done * 4
    print(f"\nDone! Objects: {done}/{len(obj_list)}  "
          f"Depth maps: {total_depth}  BEVs: {done}  Failed: {failed}")


if __name__ == "__main__":
    main()
