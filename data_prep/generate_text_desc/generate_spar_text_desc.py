#!/usr/bin/env python3
"""
SPAR text-description generation script.

For each frame in the SPAR-10k dataset, performs a 3D->2D back-projection
(Z-buffer occlusion model) against the corresponding ScanNet mesh and
extracts structured spatial information for visible objects.  Output is saved
as JSON files for injection into the model's text input at evaluation time.

Per-frame output  : <OUTPUT_DIR>/<scene_id>/<frame_id>.json
Scene-level output: <OUTPUT_DIR>/<scene_id>/_scene.json

Key design choices:
1. Labels are taken directly from aggregation.json (raw natural language, e.g. "office chair").
2. Depth and camera-relative positions are computed in camera space (X=right, Z=forward).
3. Z-buffer is used only for occlusion culling; no hole-filling is performed.
4. Scene-level BEV uses axisAlignment to produce a room-centroid compass layout.

Environment variables (all optional; defaults shown below):
  SCANNET_SCANS_DIR  - path to ScanNet scans directory
  SPAR_IMG_DIR       - path to SPAR image directory (scene/image_color or scene/video_color)
  SPAR_JSONL         - path to SPAR JSONL annotation file
  SPAR_TEXT_DESC_DIR - output directory for text-desc JSONs
"""

import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

# ==================== Configuration ====================
SCANNET_SCANS_DIR = Path(os.environ.get("SCANNET_SCANS_DIR", "/PATH/TO/ScanNet/scans"))
SPAR_IMG_DIR      = Path(os.environ.get("SPAR_IMG_DIR",      "/PATH/TO/SPAR-7M-RGBD/spar/scannet/images"))
SPAR_JSONL        = Path(os.environ.get("SPAR_JSONL",        "/PATH/TO/WORKSPACE/data/spar/spar_scannet_10k_grpo.jsonl"))
OUTPUT_DIR        = Path(os.environ.get("SPAR_TEXT_DESC_DIR","/PATH/TO/WORKSPACE/data/spar/spar_text_desc"))
MIN_COVERAGE_PCT  = 0.5   # instances below this pixel-coverage threshold are ignored
NUM_WORKERS       = 16
# =======================================================


# ─────────────────────── Scene-level data loading ───────────────────────

def load_scene_data(scene_id: str):
    """
    Load 3D mesh vertices, per-vertex instance IDs, instance label map,
    and camera intrinsics for a given ScanNet scene.

    Returns:
        vertices            : (N, 3) float32  world-space coordinates
        vertex_instance_ids : (N,)   int32    instance ID per vertex (0 = background)
        inst_to_label       : dict   int -> str   instance ID -> raw label string
        intrinsic           : (3, 3) float64  camera intrinsic matrix
    """
    scene_dir = SCANNET_SCANS_DIR / scene_id

    # Mesh vertices
    ply_path = scene_dir / f"{scene_id}_vh_clean.ply"
    plydata  = PlyData.read(str(ply_path))
    vx = np.asarray(plydata["vertex"]["x"], dtype=np.float32)
    vy = np.asarray(plydata["vertex"]["y"], dtype=np.float32)
    vz = np.asarray(plydata["vertex"]["z"], dtype=np.float32)
    vertices = np.stack([vx, vy, vz], axis=1)

    # Supervoxel IDs per vertex
    segs_path   = scene_dir / f"{scene_id}_vh_clean.segs.json"
    with open(segs_path, encoding="utf-8") as f:
        segs_data = json.load(f)
    seg_indices = np.asarray(segs_data["segIndices"], dtype=np.int32)

    # Supervoxel -> instance ID mapping and instance labels
    agg_path = scene_dir / f"{scene_id}_vh_clean.aggregation.json"
    with open(agg_path, encoding="utf-8") as f:
        agg_data = json.load(f)

    max_seg_id  = int(seg_indices.max())
    seg_to_inst = np.zeros(max_seg_id + 1, dtype=np.int32)
    inst_to_label: dict[int, str] = {}

    for group in agg_data["segGroups"]:
        inst_id = group["objectId"] + 1           # 0 reserved for background
        inst_to_label[inst_id] = group["label"]   # keep original label string
        for seg_id in group["segments"]:
            if seg_id <= max_seg_id:
                seg_to_inst[seg_id] = inst_id

    vertex_instance_ids = seg_to_inst[seg_indices]
    intrinsic = np.loadtxt(str(scene_dir / "intrinsic_color.txt"))[:3, :3]

    return vertices, vertex_instance_ids, inst_to_label, intrinsic


def get_native_color_resolution(scene_id: str) -> tuple[int, int]:
    """Read the native color resolution from the scene metadata text file."""
    native_w, native_h = 1296, 968
    scene_txt = SCANNET_SCANS_DIR / scene_id / f"{scene_id}.txt"
    with open(scene_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("colorWidth"):
                native_w = int(line.split("=")[1].strip())
            elif line.startswith("colorHeight"):
                native_h = int(line.split("=")[1].strip())
    return native_w, native_h


def get_spar_image_size(scene_id: str) -> tuple[int, int] | None:
    """Detect SPAR image resolution from the first available image in the scene."""
    spar_scene_dir = SPAR_IMG_DIR / scene_id
    for subdir in ("image_color", "video_color"):
        imgs = list((spar_scene_dir / subdir).glob("*.jpg"))
        if imgs:
            return Image.open(imgs[0]).size   # (w, h)
    return None


def text_desc_needs_regenerate(out_path: Path, img_w: int, img_h: int) -> bool:
    """Return True if the output JSON is missing or was generated with a different resolution."""
    if not out_path.is_file():
        return True
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if data.get("fallback"):
        return True
    return data.get("img_w") != img_w or data.get("img_h") != img_h


def load_axis_alignment(scene_id: str) -> np.ndarray:
    """Read the axisAlignment matrix (4x4) from scene metadata; returns identity if absent."""
    scene_txt = SCANNET_SCANS_DIR / scene_id / f"{scene_id}.txt"
    with open(scene_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = list(map(float, line.split("=")[1].strip().split()))
                return np.array(vals, dtype=np.float64).reshape(4, 4)
    return np.eye(4)


# ─────────────────────── Per-frame statistics extraction ───────────────────────

def project_frame_to_stats(
    vertices: np.ndarray,
    vertex_instance_ids: np.ndarray,
    inst_to_label: dict[int, str],
    intrinsic: np.ndarray,
    pose_path: Path,
    img_h: int,
    img_w: int,
) -> list[dict] | None:
    """
    Z-buffer back-projection -> structured statistics for each visible instance.

    Returns None if the camera pose is invalid; returns an empty list if no
    visible objects exceed the coverage threshold.  The returned list is sorted
    by depth (nearest first) and each entry contains:
        label        : raw label string from aggregation.json
        inst_id      : instance ID (int)
        depth_m      : median depth in the camera-forward direction (metres)
        cam_right_m  : signed camera-right offset (positive = right, metres)
        cam_fwd_m    : same as depth_m (forward component, kept for BEV use)
        grid_pos     : 3x3 image-grid position, e.g. "left-middle" or "center"
        coverage_pct : pixel coverage percentage of the instance in the frame
    """
    C2W = np.loadtxt(str(pose_path))
    if np.any(~np.isfinite(C2W)):
        return None
    W2C = np.linalg.inv(C2W)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # World -> camera coordinates
    R, t  = W2C[:3, :3], W2C[:3, 3]
    P_cam = (R @ vertices.T).T + t   # (N, 3)

    # Discard vertices behind the camera
    front = P_cam[:, 2] > 0.1
    P_cam    = P_cam[front]
    inst_ids = vertex_instance_ids[front]

    if inst_ids.size == 0:
        return []

    Z = P_cam[:, 2]
    u = (fx * P_cam[:, 0] / Z + cx).astype(np.int32)
    v = (fy * P_cam[:, 1] / Z + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, Z, inst_ids = u[in_bounds], v[in_bounds], Z[in_bounds], inst_ids[in_bounds]

    if inst_ids.size == 0:
        return []

    # Z-buffer: sort far-to-near so nearer points overwrite farther ones
    order = np.argsort(Z)[::-1]
    u, v, Z_sorted, inst_ids = u[order], v[order], Z[order], inst_ids[order]

    label_map = np.zeros((img_h, img_w), dtype=np.int32)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    label_map[v, u] = inst_ids
    depth_map[v, u] = Z_sorted   # nearer values overwrite farther ones

    total_px = img_h * img_w
    stats: list[dict] = []

    for inst_id in np.unique(inst_ids):
        if inst_id == 0:
            continue

        mask = (label_map == inst_id)
        n_px = int(mask.sum())
        coverage_pct = n_px / total_px * 100.0

        if coverage_pct < MIN_COVERAGE_PCT:
            continue

        rows_inst, cols_inst = np.where(mask)
        depths_inst = depth_map[rows_inst, cols_inst]

        median_depth    = float(np.median(depths_inst))
        # Back-project to camera X direction (positive = right, negative = left)
        cam_x_vals      = (cols_inst.astype(float) - cx) * depths_inst / fx
        median_cam_right = float(np.median(cam_x_vals))

        # 3x3 grid position based on pixel centroid
        u_norm = float(cols_inst.mean()) / img_w
        v_norm = float(rows_inst.mean()) / img_h
        h_pos  = "left" if u_norm < 0.33 else ("right" if u_norm > 0.67 else "center")
        v_pos  = "top"  if v_norm < 0.33 else ("bottom" if v_norm > 0.67 else "middle")
        if h_pos == "center" and v_pos == "middle":
            grid_pos = "center"
        else:
            grid_pos = f"{h_pos}-{v_pos}"

        label = inst_to_label.get(int(inst_id), f"object#{inst_id}")

        stats.append({
            "label":        label,
            "inst_id":      int(inst_id),
            "depth_m":      round(median_depth, 2),
            "cam_right_m":  round(median_cam_right, 2),
            "cam_fwd_m":    round(median_depth, 2),   # equals depth in the camera-forward axis
            "grid_pos":     grid_pos,
            "coverage_pct": round(coverage_pct, 1),
        })

    stats.sort(key=lambda x: x["depth_m"])
    return stats


# ─────────────────────── Scene-level BEV ───────────────────────

def compute_scene_bev(
    scene_id: str,
    vertices: np.ndarray,
    vertex_instance_ids: np.ndarray,
    inst_to_label: dict[int, str],
) -> list[dict]:
    """
    Compute per-instance floor-plan centroids in the axis-aligned world frame (X-Y plane).

    The axisAlignment matrix aligns the floor to Z=0 so that X-Y represents the
    horizontal plane.  Compass directions (north/south/east/west) are given relative
    to the room centroid (+Y=north, -Y=south, +X=east, -X=west).
    """
    axis_align = load_axis_alignment(scene_id)
    R_a, t_a   = axis_align[:3, :3], axis_align[:3, 3]
    verts_aligned = (R_a @ vertices.T).T + t_a   # (N, 3) axis-aligned, Z-up

    room_objs: list[dict] = []
    for inst_id, label in sorted(inst_to_label.items()):
        mask = (vertex_instance_ids == inst_id)
        if not mask.any():
            continue
        centroid = verts_aligned[mask].mean(axis=0)
        room_objs.append({
            "label":   label,
            "inst_id": inst_id,
            "x":       round(float(centroid[0]), 2),   # axis-aligned X
            "y":       round(float(centroid[1]), 2),   # axis-aligned Y
        })

    if not room_objs:
        return room_objs

    # Compass directions relative to room centroid
    cx_scene = sum(o["x"] for o in room_objs) / len(room_objs)
    cy_scene = sum(o["y"] for o in room_objs) / len(room_objs)

    for o in room_objs:
        dx, dy = o["x"] - cx_scene, o["y"] - cy_scene
        dist   = (dx ** 2 + dy ** 2) ** 0.5
        # 4-direction compass: +Y=north, -Y=south, +X=east, -X=west
        if dist < 0.3:
            compass = "center"
        elif abs(dx) >= abs(dy):
            compass = "east" if dx > 0 else "west"
        else:
            compass = "north" if dy > 0 else "south"
        o["compass"]  = compass
        o["dist_m"]   = round(dist, 2)

    return room_objs


# ─────────────────────── Per-scene processing ───────────────────────

def process_scene(scene_id: str, frame_ids: list[str]):
    """Process all target frames for one scene and write JSON output files."""
    out_dir = OUTPUT_DIR / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine actual image resolution
    size = get_spar_image_size(scene_id)
    if size is None:
        return scene_id, 0, "cannot determine image resolution"
    img_w, img_h = size

    try:
        vertices, vertex_instance_ids, inst_to_label, intrinsic = load_scene_data(scene_id)
    except Exception as e:
        return scene_id, 0, f"load_scene_data failed: {e}"

    # Scale intrinsics if SPAR images differ from native ScanNet resolution
    native_w, native_h = get_native_color_resolution(scene_id)
    if img_w != native_w or img_h != native_h:
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] *= img_w / native_w   # fx
        intrinsic[1, 1] *= img_h / native_h   # fy
        intrinsic[0, 2] *= img_w / native_w   # cx
        intrinsic[1, 2] *= img_h / native_h   # cy

    done = 0

    # Per-frame loop
    for frame_id in frame_ids:
        out_path = out_dir / f"{frame_id}.json"
        if not text_desc_needs_regenerate(out_path, img_w, img_h):
            done += 1
            continue

        pose_path = SCANNET_SCANS_DIR / scene_id / "pose" / f"{frame_id}.txt"
        if not pose_path.exists():
            continue

        try:
            obj_stats = project_frame_to_stats(
                vertices, vertex_instance_ids, inst_to_label,
                intrinsic, pose_path, img_h, img_w,
            )
        except Exception:
            continue

        if obj_stats is None:
            continue   # invalid pose; skip frame

        frame_json = {
            "scene_id": scene_id,
            "frame_id": frame_id,
            "img_h":    img_h,
            "img_w":    img_w,
            "objects":  obj_stats,
        }
        out_path.write_text(
            json.dumps(frame_json, ensure_ascii=False),
            encoding="utf-8",
        )
        done += 1

    # Write scene-level BEV once
    scene_bev_path = out_dir / "_scene.json"
    if not scene_bev_path.exists():
        try:
            room_objs = compute_scene_bev(
                scene_id, vertices, vertex_instance_ids, inst_to_label
            )
            scene_bev_path.write_text(
                json.dumps(
                    {"scene_id": scene_id, "room_objects": room_objs},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass   # BEV failure does not block per-frame processing

    return scene_id, done, None


# ─────────────────────── Entry point ───────────────────────

def main():
    global SPAR_JSONL
    if os.environ.get("SPAR_JSONL"):
        SPAR_JSONL = Path(os.environ["SPAR_JSONL"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect scene -> frame_ids mapping
    scene_frames: dict[str, set] = {}
    with open(SPAR_JSONL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            for img_path in s["image"]:
                parts    = img_path.split("/")
                scene_id = next(p for p in parts if p.startswith("scene"))
                stem     = parts[-1].rsplit(".", 1)[0]
                frame_id = stem.split("_", 1)[1] if stem.startswith("frame") else stem
                scene_frames.setdefault(scene_id, set()).add(frame_id)

    scene_list   = [(sid, sorted(fids, key=int)) for sid, fids in scene_frames.items()]
    total_frames = sum(len(fids) for _, fids in scene_list)
    print(f"Total: {len(scene_list)} scenes, {total_frames} frames to process")
    print(f"Output directory: {OUTPUT_DIR}\n")

    done_total = 0
    errors: list[tuple] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_scene, sid, fids): sid
            for sid, fids in scene_list
        }
        with tqdm(total=len(scene_list), desc="Scenes", unit="scene") as pbar:
            for future in as_completed(futures):
                sid, done, err = future.result()
                done_total += done
                if err:
                    errors.append((sid, err))
                pbar.update(1)
                pbar.set_postfix(frames=done_total, errors=len(errors))

    print(f"\nDone! Generated text-desc JSONs: {done_total} / {total_frames} frames")
    if errors:
        print(f"WARNING: {len(errors)} scene(s) failed (first 5):")
        for sid, err in errors[:5]:
            print(f"   {sid}: {err}")


if __name__ == "__main__":
    main()
