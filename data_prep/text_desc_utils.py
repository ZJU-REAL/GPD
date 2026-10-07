"""
Structured text description rendering utilities.

Reads JSON files produced by generate_spar/vsi_text_desc.py and renders them
into plain-text strings that can be injected directly into model inputs.

Core functions:
    frame_json_to_text(frame_data, mode)   : single-frame JSON → text
    scene_json_to_bev_text(scene_data)     : scene-level BEV JSON → text
    build_scene_context(...)               : multi-frame + BEV → full <scene_context> block

The ``mode`` parameter controls which feature dimensions are injected
(consistent with feature_router output):
    "depth"    → emit depth-ordered list only
    "semantic" → emit object list + image-grid positions only
    "bev"      → emit camera-relative BEV + scene-level compass layout only
    list form e.g. ["depth", "semantic"] → emit multiple dimensions
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


# ─────────────────────── Direction description helpers ───────────────────────

def _cam_direction(cam_right_m: float, cam_fwd_m: float, thresh: float = 0.4) -> str:
    """
    Convert camera-relative coordinates to a natural-language direction string.
    cam_right_m > 0 = right, cam_fwd_m > 0 = forward
    """
    right = abs(cam_right_m) >= thresh
    side  = ("right" if cam_right_m > 0 else "left") if right else ""
    fwd   = "front" if cam_fwd_m >= 0 else "back"
    return f"{side}-{fwd}" if side else fwd


def _horiz_dist(cam_right_m: float, cam_fwd_m: float) -> float:
    return math.sqrt(cam_right_m ** 2 + cam_fwd_m ** 2)


# ─────────────────────── Single-frame rendering ───────────────────────

def frame_json_to_text(
    frame_data: dict,
    mode: str | Sequence[str] = ("depth", "semantic", "bev"),
) -> str:
    """
    Render a single-frame JSON dict to text.

    Parameters
    ----------
    frame_data : dict
        Frame-level JSON written by generate_*_text_desc.py.
    mode : str or list[str]
        "depth"    — depth-sorted row
        "semantic" — object list + image-grid positions
        "bev"      — camera-relative horizontal positions

    Returns
    -------
    str  Rendered text string; empty string if no visible objects.
    """
    if isinstance(mode, str):
        mode = [mode]
    mode_set = set(mode)

    objs = frame_data.get("objects", [])
    if not objs:
        return ""

    lines: list[str] = []

    if "depth" in mode_set:
        parts = [
            f"{o['label']}({o['depth_m']}m)"
            for o in objs
        ]
        lines.append("Depth near→far: " + " > ".join(parts))

    if "semantic" in mode_set:
        parts = [
            f"{o['label']} [{o['grid_pos']}, {o['coverage_pct']:.0f}%]"
            for o in objs
        ]
        lines.append("Objects: " + "; ".join(parts))

    if "bev" in mode_set:
        parts = []
        for o in objs:
            d    = _cam_direction(o["cam_right_m"], o["cam_fwd_m"])
            dist = _horiz_dist(o["cam_right_m"], o["cam_fwd_m"])
            parts.append(f"{o['label']}({d},{dist:.1f}m)")
        lines.append("Cam-BEV: " + ", ".join(parts))

    return "\n".join(lines)


# ─────────────────────── Scene-level BEV rendering ───────────────────────

def scene_json_to_bev_text(scene_data: dict, max_objects: int = 20) -> str:
    """
    Render _scene.json into a room top-down layout text string.

    Keeps only the top max_objects entries (sorted by dist_m) to avoid
    excessively long text.
    """
    room_objs = scene_data.get("room_objects", [])
    if not room_objs:
        return ""

    # Sort by distance from centroid; objects closer to center tend to be more salient
    sorted_objs = sorted(room_objs, key=lambda o: o.get("dist_m", 9999))[:max_objects]

    parts = [f"{o['label']}({o.get('compass','?')})" for o in sorted_objs]
    return "Room layout: " + ", ".join(parts)


# ─────────────────────── Multi-frame aggregation ───────────────────────

def build_scene_context(
    frame_jsons: list[dict],
    scene_json:  dict | None = None,
    mode:        str | Sequence[str] = ("depth", "semantic", "bev"),
    frame_labels: list[str] | None = None,
) -> str:
    """
    Aggregate multi-frame text descriptions into a single <scene_context> block.

    Parameters
    ----------
    frame_jsons   : List of frame-level JSON dicts in frame order.
    scene_json    : Scene-level _scene.json (optional; used to inject room-level BEV).
    mode          : Feature dimensions to inject (same as frame_json_to_text).
    frame_labels  : Per-frame labels, e.g. ["Frame 0", "Frame 1", ...];
                    auto-generated as "Frame 0"..."Frame N-1" when None.

    Returns
    -------
    str  Complete <scene_context>...</scene_context> string.
    """
    if isinstance(mode, str):
        mode = [mode]

    if frame_labels is None:
        frame_labels = [f"Frame {i}" for i in range(len(frame_jsons))]

    lines: list[str] = ["<scene_context>"]

    # Scene-level BEV (emitted only when "bev" is in mode)
    if scene_json is not None and "bev" in mode:
        bev_text = scene_json_to_bev_text(scene_json)
        if bev_text:
            lines.append(bev_text)
            lines.append("")   # blank line separator

    # Per-frame text
    for label, frame_data in zip(frame_labels, frame_jsons):
        frame_text = frame_json_to_text(frame_data, mode=mode)
        if frame_text:
            lines.append(f"[{label}] {frame_text}")

    lines.append("</scene_context>")
    # Do not inject an empty block if there is no valid content
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


# ─────────────────────── File loading helpers ───────────────────────

def load_frame_json(text_desc_dir: str | Path, scene_id: str, frame_name: str) -> dict | None:
    """
    Load a single-frame JSON. frame_name is the filename without extension,
    e.g. "000500" or "frame_03".
    Fallback placeholder entries (invalid pose) are treated as no privilege; returns None.
    """
    path = Path(text_desc_dir) / scene_id / f"{frame_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fallback"):
        return None
    return data


def load_scene_json(text_desc_dir: str | Path, scene_id: str) -> dict | None:
    """Load the scene-level _scene.json."""
    path = Path(text_desc_dir) / scene_id / "_scene.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_spar_context(
    sample: dict,
    spar_text_dir: str | Path,
    mode: str | Sequence[str] = ("depth", "semantic", "bev"),
) -> str:
    """
    Given a SPAR sample dict, automatically load JSON for all frames and
    produce a <scene_context> block.

    sample["image"] format: ["/path/to/.../scene0001_00/image_color/000500.jpg", ...]
    """
    frame_jsons: list[dict] = []
    frame_labels: list[str] = []
    scene_id: str = ""

    for img_path in sample.get("image", []):
        parts    = img_path.split("/")
        sid      = next((p for p in parts if p.startswith("scene")), None)
        if sid is None:
            continue
        if not scene_id:
            scene_id = sid
        stem     = parts[-1].rsplit(".", 1)[0]
        frame_id = stem.split("_", 1)[1] if stem.startswith("frame") else stem

        fj = load_frame_json(spar_text_dir, sid, frame_id)
        if fj is None:
            continue
        frame_jsons.append(fj)
        frame_labels.append(frame_id)

    scene_json = load_scene_json(spar_text_dir, scene_id) if scene_id else None

    return build_scene_context(
        frame_jsons, scene_json=scene_json, mode=mode, frame_labels=frame_labels
    )


def build_vsi_context(
    sample: dict,
    vsi_text_dir: str | Path,
    mode: str | Sequence[str] = ("depth", "semantic", "bev"),
) -> str:
    """
    Given a VSI sample dict (containing a list of 16 frame images), automatically
    load corresponding JSONs and produce a <scene_context> block.

    sample["image"] format: ["/path/to/frame_00.jpg", ..., "/path/to/frame_15.jpg"]
    Corresponding JSON filenames: frame_00.json ~ frame_15.json
    """
    frame_jsons: list[dict] = []
    frame_labels: list[str] = []
    scene_id: str = ""

    for img_path in sample.get("image", []):
        parts    = img_path.split("/")
        sid      = next((p for p in parts if p.startswith("scene")), None)
        if sid is None:
            continue
        if not scene_id:
            scene_id = sid
        stem = parts[-1].rsplit(".", 1)[0]  # e.g. "frame_03"

        fj = load_frame_json(vsi_text_dir, sid, stem)
        if fj is None:
            continue
        frame_jsons.append(fj)
        frame_labels.append(stem)

    scene_json = load_scene_json(vsi_text_dir, scene_id) if scene_id else None

    return build_scene_context(
        frame_jsons, scene_json=scene_json, mode=mode, frame_labels=frame_labels
    )


def build_vsi_interleaved_user_content(
    sample: dict,
    vsi_text_dir: str | Path,
    features: Sequence[str],
    question_text: str,
    cot_post_prompt: str,
) -> list[dict]:
    """
    VSI: LLM-routed + image-interleaved 3D text (Qwen-VL user content list).

    Ordering (differs from the single-block <scene_context> version):
      After each RGB image, append that frame's depth/semantic text
      (the subset determined by ``features``; may be depth-only or semantic-only);
      after all views, if ``features`` contains ``bev``, inject the scene bird's-eye
      view from ``_scene.json`` (Room layout);
      finally append ``Question`` + CoT suffix.

    Frame-level ``bev`` (Cam-BEV) is NOT placed after individual frames to avoid
    confusion with the scene-wide BEV appended at the end; if per-frame Cam-BEV is
    needed it can be added later by allowing ``bev`` inside the per-frame block.
    """
    vsi_text_dir = Path(vsi_text_dir)
    if not features:
        feat_set = set()
    elif isinstance(features, str):
        feat_set = {features.strip()}
    else:
        feat_set = set(features)

    per_frame_modes = [m for m in ("depth", "semantic") if m in feat_set]
    want_scene_bev = "bev" in feat_set

    imgs = sample.get("image", [])
    content: list[dict] = []
    scene_id = ""

    for img_path in imgs:
        content.append({"type": "image", "image": img_path})

        parts = img_path.split("/")
        sid = next((p for p in parts if p.startswith("scene")), None)
        stem = parts[-1].rsplit(".", 1)[0] if parts else ""

        if sid is None:
            continue
        if not scene_id:
            scene_id = sid

        if per_frame_modes:
            fj = load_frame_json(vsi_text_dir, sid, stem)
            frame_data = fj if fj is not None else {}
            frame_txt = frame_json_to_text(frame_data, mode=per_frame_modes)
            if frame_txt.strip():
                content.append({"type": "text", "text": f"\n[{stem}] {frame_txt}\n"})

    if want_scene_bev and scene_id:
        scene_json = load_scene_json(vsi_text_dir, scene_id)
        if scene_json is not None:
            bev_txt = scene_json_to_bev_text(scene_json)
            if bev_txt.strip():
                content.append(
                    {"type": "text", "text": f"\nScene bird's-eye view (room layout):\n{bev_txt}\n"}
                )

    pre = "Question: " + question_text.strip() + "\nThink step by step.\n"
    content.append({"type": "text", "text": "\n" + pre + cot_post_prompt})

    return content


def _routing_feat_set(features: Sequence[str] | str) -> set[str]:
    if not features:
        return set()
    if isinstance(features, str):
        return {features.strip()}
    return set(features)


def mindcube_frame_list_to_text(objs: list, feature_modes: Sequence[str]) -> str:
    """
    MindCube single-frame JSON (object list) → plain text with the same semantics
    as mindcube_text_cot_eval.load_frame_text.
    feature_modes is a subset of ``depth`` / ``semantic`` (MindCube baseline does not use bev).
    """
    if not objs:
        return ""
    modes = set(feature_modes)
    lines: list[str] = []
    if "semantic" in modes:
        for o in objs:
            label = o.get("label", "object")
            pos = o.get("position", {})
            h = pos.get("horizontal", "")
            v = pos.get("vertical", "")
            lines.append(f"  {label}: {h} {v}".strip())
    if "depth" in modes:
        sorted_by_depth = sorted(objs, key=lambda x: x.get("depth_rank", 99))
        depth_labels = [o.get("label", "object") for o in sorted_by_depth]
        if depth_labels:
            lines.append("  depth order (near→far): " + ", ".join(depth_labels))
    return "\n".join(lines) if lines else ""


def _mindcube_obj_key_stem(img_path: str) -> tuple[str, str]:
    """Return (text_desc subdirectory key, frame filename without extension), consistent with mindcube_text_cot_eval."""
    parts = img_path.split("/")
    key = parts[-3] + "/" + parts[-2]
    stem = parts[-1].rsplit(".", 1)[0] if parts else ""
    return key, stem


def build_mindcube_interleaved_user_content(
    sample: dict,
    text_desc_dir: str | Path,
    features: Sequence[str],
    question_text: str,
    cot_post_prompt: str,
) -> list[dict]:
    """
    MindCube: same data source and per-frame text rules as mindcube_text_cot_eval,
    but each image is immediately followed by that frame's 3D text
    (depth / semantic only; bev is silently ignored if routed but unavailable,
    consistent with the baseline).
    """
    text_desc_dir = Path(text_desc_dir)
    feat_set = _routing_feat_set(features)
    per_frame_modes = [m for m in ("depth", "semantic") if m in feat_set]

    imgs = sample.get("image", [])
    content: list[dict] = []

    for img_path in imgs:
        content.append({"type": "image", "image": img_path})
        if not per_frame_modes:
            continue
        subdir, stem = _mindcube_obj_key_stem(img_path)
        path = text_desc_dir / subdir / f"{stem}.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                objs = json.load(f)
        except Exception:
            continue
        if not objs:
            continue
        frame_txt = mindcube_frame_list_to_text(objs, per_frame_modes)
        if frame_txt.strip():
            content.append({"type": "text", "text": f"\n[{stem}] {frame_txt}\n"})

    pre = "Question: " + question_text.strip() + "\nThink step by step.\n"
    content.append({"type": "text", "text": "\n" + pre + cot_post_prompt})
    return content


def build_spar_interleaved_user_content(
    sample: dict,
    spar_text_dir: str | Path,
    features: Sequence[str],
    question_text: str,
    cot_post_prompt: str,
) -> list[dict]:
    """
    SPAR: interleaved layout identical to VSI; frame JSON key parsing follows
    ``build_spar_context`` (frame_xxx → frame_id).
    """
    spar_text_dir = Path(spar_text_dir)
    feat_set = _routing_feat_set(features)
    per_frame_modes = [m for m in ("depth", "semantic") if m in feat_set]
    want_scene_bev = "bev" in feat_set

    imgs = sample.get("image", [])
    content: list[dict] = []
    scene_id = ""

    for img_path in imgs:
        content.append({"type": "image", "image": img_path})

        parts = img_path.split("/")
        sid = next((p for p in parts if p.startswith("scene")), None)
        stem = parts[-1].rsplit(".", 1)[0] if parts else ""
        frame_id = stem.split("_", 1)[1] if stem.startswith("frame") else stem

        if sid is None:
            continue
        if not scene_id:
            scene_id = sid

        if per_frame_modes:
            fj = load_frame_json(spar_text_dir, sid, frame_id)
            frame_data = fj if fj is not None else {}
            frame_txt = frame_json_to_text(frame_data, mode=per_frame_modes)
            if frame_txt.strip():
                content.append({"type": "text", "text": f"\n[{frame_id}] {frame_txt}\n"})

    if want_scene_bev and scene_id:
        scene_json = load_scene_json(spar_text_dir, scene_id)
        if scene_json is not None:
            bev_txt = scene_json_to_bev_text(scene_json)
            if bev_txt.strip():
                content.append(
                    {"type": "text", "text": f"\nScene bird's-eye view (room layout):\n{bev_txt}\n"}
                )

    pre = "Question: " + question_text.strip() + "\nThink step by step.\n"
    content.append({"type": "text", "text": "\n" + pre + cot_post_prompt})
    return content


# ─────────────────────── Quick preview (command-line debug) ───────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_desc_utils.py <frame_json_path> [mode1,mode2,...]")
        sys.exit(0)

    json_path = Path(sys.argv[1])
    modes     = sys.argv[2].split(",") if len(sys.argv) > 2 else ["depth", "semantic", "bev"]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Determine whether this is a frame-level or scene-level JSON
    if "objects" in data:
        print(frame_json_to_text(data, mode=modes))
    elif "room_objects" in data:
        print(scene_json_to_bev_text(data))
    else:
        print("Unknown JSON format")
