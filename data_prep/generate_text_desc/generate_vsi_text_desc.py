#!/usr/bin/env python3
"""
VSI text-description generation script.

Generates per-frame structured JSON descriptions for the VSI-30k dataset
by back-projecting ScanNet 3D mesh vertices into each video frame using a
Z-buffer occlusion model.  Logic is identical to generate_spar_text_desc.py;
the only difference is the frame-index mapping:

  SPAR : image path stem == ScanNet frame number (e.g. 1004.jpg -> frame 1004)
  VSI  : images are frame_00 ~ frame_15; the corresponding ScanNet frame is
         recovered via:
           scannet_frame = linspace(0, total_pose_frames-1, 16).astype(int)[frame_idx]

Per-frame output  : <OUTPUT_DIR>/<scene_id>/frame_{idx:02d}.json
Scene-level output: <OUTPUT_DIR>/<scene_id>/_scene.json

Environment variables (all optional; defaults shown below):
  SCANNET_SCANS_DIR  - path to ScanNet scans directory
  VSI_FRAMES_DIR     - path to extracted VSI video frames
  VSI_JSONL          - path to VSI JSONL annotation file
  VSI_TEXT_DESC_DIR  - output directory for text-desc JSONs
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
VSI_FRAMES_DIR    = Path(os.environ.get("VSI_FRAMES_DIR",    "/PATH/TO/WORKSPACE/data/frames"))
VSI_JSONL         = Path(os.environ.get("VSI_JSONL",         "/PATH/TO/WORKSPACE/data/vsi/vsi_rl_30k_16frames_ready.jsonl"))
OUTPUT_DIR        = Path(os.environ.get("VSI_TEXT_DESC_DIR", "/PATH/TO/WORKSPACE/data/vsi/vsi_text_desc"))
MIN_COVERAGE_PCT  = 0.5
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

    ply_path = scene_dir / f"{scene_id}_vh_clean.ply"
    plydata  = PlyData.read(str(ply_path))
    vx = np.asarray(plydata["vertex"]["x"], dtype=np.float32)
    vy = np.asarray(plydata["vertex"]["y"], dtype=np.float32)
    vz = np.asarray(plydata["vertex"]["z"], dtype=np.float32)
    vertices = np.stack([vx, vy, vz], axis=1)

    segs_path = scene_dir / f"{scene_id}_vh_clean.segs.json"
    with open(segs_path, encoding="utf-8") as f:
        segs_data = json.load(f)
    seg_indices = np.asarray(segs_data["segIndices"], dtype=np.int32)

    agg_path = scene_dir / f"{scene_id}_vh_clean.aggregation.json"
    with open(agg_path, encoding="utf-8") as f:
        agg_data = json.load(f)

    max_seg_id  = int(seg_indices.max())
    seg_to_inst = np.zeros(max_seg_id + 1, dtype=np.int32)
    inst_to_label: dict[int, str] = {}

    for group in agg_data["segGroups"]:
        inst_id = group["objectId"] + 1   # 0 reserved for background
        inst_to_label[inst_id] = group["label"]
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


def get_vsi_frame_size(scene_id: str) -> tuple[int, int] | None:
    """Read the actual resolution of extracted VSI video frames."""
    sample = VSI_FRAMES_DIR / scene_id / "frame_00.jpg"
    if not sample.is_file():
        return None
    return Image.open(sample).size


def scale_intrinsic(intrinsic: np.ndarray, img_w: int, img_h: int, native_w: int, native_h: int) -> np.ndarray:
    """Scale intrinsic matrix from native ScanNet resolution to the target frame resolution."""
    if img_w == native_w and img_h == native_h:
        return intrinsic
    k = intrinsic.copy()
    k[0, 0] *= img_w / native_w
    k[1, 1] *= img_h / native_h
    k[0, 2] *= img_w / native_w
    k[1, 2] *= img_h / native_h
    return k


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


# ─────────────────────── VSI core: frame-index mapping ───────────────────────

def get_scannet_frame_indices(scene_id: str) -> np.ndarray:
    """
    Return a length-16 array mapping VSI frame index (0-15) to ScanNet frame numbers.
    Uses linspace over the total number of pose files in the scene.
    """
    pose_dir = SCANNET_SCANS_DIR / scene_id / "pose"
    n        = len(list(pose_dir.glob("*.txt")))
    return np.linspace(0, n - 1, 16).astype(int)


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

    R, t  = W2C[:3, :3], W2C[:3, 3]
    P_cam = (R @ vertices.T).T + t

    # Discard vertices behind the camera
    front    = P_cam[:, 2] > 0.1
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
    depth_map[v, u] = Z_sorted

    total_px = img_h * img_w
    stats: list[dict] = []

    for inst_id in np.unique(inst_ids):
        if inst_id == 0:
            continue

        mask         = (label_map == inst_id)
        n_px         = int(mask.sum())
        coverage_pct = n_px / total_px * 100.0

        if coverage_pct < MIN_COVERAGE_PCT:
            continue

        rows_inst, cols_inst = np.where(mask)
        depths_inst = depth_map[rows_inst, cols_inst]

        median_depth     = float(np.median(depths_inst))
        # Back-project to camera X direction (positive = right, negative = left)
        cam_x_vals       = (cols_inst.astype(float) - cx) * depths_inst / fx
        median_cam_right = float(np.median(cam_x_vals))

        # 3x3 grid position based on pixel centroid
        u_norm = float(cols_inst.mean()) / img_w
        v_norm = float(rows_inst.mean()) / img_h
        h_pos  = "left" if u_norm < 0.33 else ("right" if u_norm > 0.67 else "center")
        v_pos  = "top"  if v_norm < 0.33 else ("bottom" if v_norm > 0.67 else "middle")
        grid_pos = "center" if (h_pos == "center" and v_pos == "middle") \
                             else f"{h_pos}-{v_pos}"

        label = inst_to_label.get(int(inst_id), f"object#{inst_id}")

        stats.append({
            "label":        label,
            "inst_id":      int(inst_id),
            "depth_m":      round(median_depth, 2),
            "cam_right_m":  round(median_cam_right, 2),
            "cam_fwd_m":    round(median_depth, 2),
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
    to the room centroid.
    """
    axis_align  = load_axis_alignment(scene_id)
    R_a, t_a    = axis_align[:3, :3], axis_align[:3, 3]
    verts_aligned = (R_a @ vertices.T).T + t_a

    room_objs: list[dict] = []
    for inst_id, label in sorted(inst_to_label.items()):
        mask = (vertex_instance_ids == inst_id)
        if not mask.any():
            continue
        centroid = verts_aligned[mask].mean(axis=0)
        room_objs.append({
            "label":   label,
            "inst_id": inst_id,
            "x":       round(float(centroid[0]), 2),
            "y":       round(float(centroid[1]), 2),
        })

    if not room_objs:
        return room_objs

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
        o["compass"] = compass
        o["dist_m"]  = round(dist, 2)

    return room_objs


# ─────────────────────── Per-scene processing ───────────────────────

def process_scene(scene_id: str, frame_index_map: dict[int, int]):
    """
    Process all 16 frames for one scene.

    frame_index_map: {frame_idx (0-15) -> scannet_frame_number}
    """
    out_dir = OUTPUT_DIR / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    size = get_vsi_frame_size(scene_id)
    if size is None:
        return scene_id, 0, "frame_00.jpg not found"

    img_w, img_h = size

    try:
        vertices, vertex_instance_ids, inst_to_label, intrinsic = load_scene_data(scene_id)
    except Exception as e:
        return scene_id, 0, f"load_scene_data failed: {e}"

    native_w, native_h = get_native_color_resolution(scene_id)
    intrinsic = scale_intrinsic(intrinsic, img_w, img_h, native_w, native_h)

    done = 0

    for frame_idx, scannet_frame in frame_index_map.items():
        out_name  = f"frame_{frame_idx:02d}.json"
        out_path  = out_dir / out_name
        if not text_desc_needs_regenerate(out_path, img_w, img_h):
            done += 1
            continue

        pose_path = SCANNET_SCANS_DIR / scene_id / "pose" / f"{scannet_frame}.txt"
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
            continue

        frame_json = {
            "scene_id":      scene_id,
            "frame_idx":     frame_idx,
            "scannet_frame": int(scannet_frame),
            "img_h":         img_h,
            "img_w":         img_w,
            "objects":       obj_stats,
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
            pass

    return scene_id, done, None


# ─────────────────────── Entry point ───────────────────────

def main():
    global VSI_JSONL
    if os.environ.get("VSI_JSONL"):
        VSI_JSONL = Path(os.environ["VSI_JSONL"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect scene -> frame_index_map
    scene_frame_map: dict[str, dict[int, int]] = {}

    with open(VSI_JSONL, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            img_path = s["image"][0]
            parts    = img_path.split("/")
            scene_id = next(p for p in parts if p.startswith("scene"))

            if scene_id not in scene_frame_map:
                scannet_indices = get_scannet_frame_indices(scene_id)
                scene_frame_map[scene_id] = {
                    i: int(scannet_indices[i]) for i in range(16)
                }

    scene_list   = list(scene_frame_map.items())
    total_frames = len(scene_list) * 16
    print(f"Total: {len(scene_list)} scenes, {total_frames} frames to process")
    print(f"Output directory: {OUTPUT_DIR}\n")

    done_total = 0
    errors: list[tuple] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_scene, sid, fmap): sid
            for sid, fmap in scene_list
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
