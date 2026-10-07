"""
RL mixed data: Teacher privilege text / image path construction.

Seven parquet variants (extra_info.priv_variant):
  pure_grpo              — no privilege
  text_routed            — routed 3D text + reference answer
  text_routed_no_answer  — routed 3D text, no reference answer (Teacher sees <scene_context> only)
  text_full              — full 3D text (depth+semantic+bev) + reference answer
  image_full             — full 3D images + reference answer (text is reference answer block only)
  image_routed           — routed 3D images + reference answer
  answer_only            — reference answer only
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
# Same-package imports (text_desc_utils / feature_router live beside this file).
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Offline 3D feature roots. Override with GPD_DATA_ROOT pointing at a tree that
# contains vsi_*/spar_*/mindcube_* privilege assets used to build scene context.
_DATA_ROOT = Path(os.environ.get("GPD_DATA_ROOT", "/PATH/TO/WORKSPACE/data"))

PATHS = {
    "vsi": {
        "text_desc": _DATA_ROOT / "vsi/vsi_text_desc",
        "depth_rgb": _DATA_ROOT / "vsi/vsi_depth_rgb",
        "semantic": _DATA_ROOT / "vsi/vsi_semantic",
        "bev": _DATA_ROOT / "vsi/vsi_bev",
    },
    "spar": {
        "text_desc": _DATA_ROOT / "spar/spar_text_desc",
        "depth_rgb": _DATA_ROOT / "spar/spar_depth_rgb",
        "semantic": _DATA_ROOT / "spar/spar_semantic",
        "bev": _DATA_ROOT / "spar/spar_bev",
    },
    "mindcube": {
        "text_desc": _DATA_ROOT / "mindcube_text_desc",
        "depth_rgb": _DATA_ROOT / "mindcube_depth_rgb",
        "semantic": _DATA_ROOT / "mindcube_semantic",
        "bev": None,
    },
}

FULL_FEATURES_VSI_SPAR = ["depth", "semantic", "bev"]
FULL_FEATURES_MINDCUBE = ["depth", "semantic"]

PRIV_VARIANTS = (
    "pure_grpo",
    "text_routed",
    "text_routed_no_answer",
    "text_full",
    "image_full",
    "image_routed",
    "answer_only",
)

REFERENCE_ANSWER_TPL = "<reference_answer>\n{answer}\n</reference_answer>"


def build_reference_answer_block(answer: str) -> str:
    ans = (answer or "").strip()
    if not ans:
        return ""
    return REFERENCE_ANSWER_TPL.format(answer=ans)


def _merge_priv_text(scene_ctx: str, answer: str) -> str:
    parts = []
    sc = (scene_ctx or "").strip()
    if sc and sc != "<scene_context>\n</scene_context>":
        parts.append(sc)
    ref = build_reference_answer_block(answer)
    if ref:
        parts.append(ref)
    return "\n\n".join(parts)


def _scene_context_only(scene_ctx: str) -> str:
    """text_routed_no_answer: keep only a valid <scene_context>; exclude <reference_answer>."""
    sc = (scene_ctx or "").strip()
    if sc and sc != "<scene_context>\n</scene_context>":
        return sc
    return ""


# ── Routing ─────────────────────────────────────────────────────────────

def rule_route(sample: dict) -> list[str]:
    from feature_router import rule_route as _rr

    return list(_rr(sample))


def load_routing_map(routing_file: Optional[str]) -> dict[int, list[str]]:
    if not routing_file or not Path(routing_file).is_file():
        return {}
    m: dict[int, list[str]] = {}
    with open(routing_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("line_idx", r.get("_line_idx"))
            if idx is not None:
                m[int(idx)] = r.get("features", [])
    return m


def features_for_sample(
    sample: dict,
    line_idx: int,
    routing_map: dict[int, list[str]],
    *,
    full: bool = False,
) -> list[str]:
    ds = sample.get("dataset", "")
    if full:
        return FULL_FEATURES_MINDCUBE if ds == "mindcube" else FULL_FEATURES_VSI_SPAR
    if line_idx in routing_map:
        return routing_map[line_idx]
    return rule_route(sample)


# ── 3D text ─────────────────────────────────────────────────────────

def build_mindcube_scene_context(
    image_paths: list[str],
    text_desc_dir: Path,
    features: list[str],
) -> str:
    if not features:
        return ""
    frame_texts = []
    for i, img_path in enumerate(image_paths, start=1):
        parts = img_path.split("/")
        key = parts[-3] + "/" + parts[-2]
        stem = os.path.splitext(parts[-1])[0]
        jp = text_desc_dir / key / f"{stem}.json"
        if not jp.is_file():
            continue
        try:
            objs = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not objs or (isinstance(objs, dict) and objs.get("fallback")):
            continue
        if isinstance(objs, dict):
            objs = objs.get("objects", [])
        if not objs:
            continue
        lines = []
        if "semantic" in features:
            lines.append("Visible objects (label, grid position, coverage):")
            for o in objs:
                lines.append(
                    f"  - {o['label']} at {o['grid_pos']} ({o['coverage_pct']:.1f}% of frame)"
                )
        if "depth" in features:
            sorted_objs = sorted(objs, key=lambda x: x.get("depth_rel", 0))
            dp = [f"{o['label']} (rel_depth={o['depth_rel']:.2f})" for o in sorted_objs]
            lines.append("Depth order (nearest to farthest): " + " < ".join(dp))
        if lines:
            frame_texts.append(f"[Image {i}]\n" + "\n".join(lines))
    if not frame_texts:
        return ""
    body = "\n\n".join(frame_texts)
    return f"<scene_context>\n{body}\n</scene_context>"


def build_scene_context_text(
    sample: dict,
    features: list[str],
    paths_cfg: dict,
) -> str:
    if not features:
        return ""
    ds = sample.get("dataset", "")
    imgs = sample.get("image", [])
    if isinstance(imgs, str):
        imgs = [imgs]
    text_dir = paths_cfg.get("text_desc")
    if not text_dir:
        return ""

    try:
        from text_desc_utils import build_vsi_context, build_spar_context
    except ImportError:
        build_vsi_context = build_spar_context = None  # type: ignore

    sample_dict = {"image": imgs}
    if ds == "vsi" and build_vsi_context:
        return build_vsi_context(sample_dict, text_dir, mode=features)
    if ds == "spar" and build_spar_context:
        return build_spar_context(sample_dict, text_dir, mode=features)
    if ds == "mindcube":
        return build_mindcube_scene_context(imgs, Path(text_dir), features)
    return ""


# ── 3D image paths (for Teacher) ─────────────────────────────────────────

def _vsi_scene_frame(img_path: str) -> tuple[str, str]:
    parts = img_path.split("/")
    sid = next((p for p in parts if p.startswith("scene")), "")
    stem = os.path.splitext(parts[-1])[0]
    return sid, stem


def _spar_scene_frame(img_path: str) -> tuple[str, str]:
    return _vsi_scene_frame(img_path)


def _mindcube_key_stem(img_path: str) -> tuple[str, str]:
    parts = img_path.split("/")
    return parts[-3] + "/" + parts[-2], os.path.splitext(parts[-1])[0]


def _is_usable_semantic_png(path: Path) -> bool:
    """Skip missing or all-black fallback semantic images (invalid-pose frames must not inject semantic privilege)."""
    if not path.is_file():
        return False
    try:
        import numpy as np
        from PIL import Image

        arr = np.array(Image.open(path))
        if arr.size == 0:
            return False
        return bool(arr.max() > 0)
    except Exception:
        return False


def build_teacher_image_paths(
    sample: dict,
    features: list[str],
    paths_cfg: dict,
) -> list[str]:
    """Student RGB paths plus depth/semantic/BEV image paths appended according to features (ordered)."""
    ds = sample.get("dataset", "")
    imgs = sample.get("image", [])
    if isinstance(imgs, str):
        imgs = [imgs]
    if not imgs:
        return []

    depth_dir = paths_cfg.get("depth_rgb")
    sem_dir = paths_cfg.get("semantic")
    bev_dir = paths_cfg.get("bev")

    use_depth = "depth" in features
    use_sem = "semantic" in features
    use_bev = "bev" in features and bev_dir

    out: list[str] = []
    n = len(imgs)

    # SPAR 32-frame video type: BEV + RGB only (consistent with zero-shot setup)
    spar_video_only_bev = ds == "spar" and n > 8

    if use_bev and bev_dir and not spar_video_only_bev:
        if ds in ("vsi", "spar"):
            sid, _ = _vsi_scene_frame(imgs[0])
            bev_p = Path(bev_dir) / f"{sid}.png"
            if bev_p.is_file():
                out.append(str(bev_p))
        # mindcube has no scene-level BEV

    for img_path in imgs:
        out.append(img_path)
        if spar_video_only_bev:
            continue
        if ds in ("vsi", "spar"):
            sid, stem = _vsi_scene_frame(img_path)
            if use_depth and depth_dir:
                dp = Path(depth_dir) / sid / f"{stem}.jpg"
                if dp.is_file():
                    out.append(str(dp))
            if use_sem and sem_dir:
                sp = Path(sem_dir) / sid / f"{stem}.png"
                if _is_usable_semantic_png(sp):
                    out.append(str(sp))
        elif ds == "mindcube":
            key, stem = _mindcube_key_stem(img_path)
            if use_depth and depth_dir:
                dp = Path(depth_dir) / key / f"{stem}.jpg"
                if dp.is_file():
                    out.append(str(dp))
            if use_sem and sem_dir:
                sp = Path(sem_dir) / key / f"{stem}.png"
                if sp.is_file():
                    out.append(str(sp))

    if spar_video_only_bev and use_bev and bev_dir:
        sid, _ = _spar_scene_frame(imgs[0])
        bev_p = Path(bev_dir) / f"{sid}.png"
        if bev_p.is_file() and str(bev_p) not in out:
            out.insert(0, str(bev_p))

    return out


# ── Variant population ─────────────────────────────────────────────────────────

def apply_priv_variant(
    record: dict,
    raw_sample: dict,
    variant: str,
    routing_map: dict[int, list[str]],
    paths_root: Optional[dict[str, dict]] = None,
) -> dict:
    """Populate priv_context_text / teacher_images / extra_info in-place."""
    paths_root = paths_root or PATHS
    ds = raw_sample.get("dataset", record.get("extra_info", {}).get("dataset", "vsi"))
    cfg = paths_root.get(ds, paths_root["vsi"])
    answer = record.get("answer", "")
    line_idx = int(raw_sample.get("_line_idx", record.get("extra_info", {}).get("index", -1)))

    ei = dict(record.get("extra_info") or {})
    ei["priv_variant"] = variant
    ei["line_idx"] = line_idx
    af = raw_sample.get("answer_format", ei.get("answer_format", "mc"))
    ei["answer_format"] = af
    ei["answer_kind"] = "numeric" if str(af).lower() in ("open", "fill") else "choice"

    student_imgs = [x["path"] for x in record.get("images", [])]

    if variant == "pure_grpo":
        record["priv_context_text"] = ""
        record["teacher_images"] = [{"path": p} for p in student_imgs]
        record["extra_info"] = ei
        return record

    if variant == "answer_only":
        record["priv_context_text"] = build_reference_answer_block(answer)
        record["teacher_images"] = [{"path": p} for p in student_imgs]
        ei["priv_context_mode"] = "answer_only"
        record["extra_info"] = ei
        return record

    if variant == "text_routed":
        feats = features_for_sample(raw_sample, line_idx, routing_map, full=False)
        scene = build_scene_context_text(raw_sample, feats, cfg)
        record["priv_context_text"] = _merge_priv_text(scene, answer)
        record["teacher_images"] = [{"path": p} for p in student_imgs]
        ei["features"] = feats
        ei["priv_context_mode"] = "text_routed"
        record["extra_info"] = ei
        return record

    if variant == "text_routed_no_answer":
        feats = features_for_sample(raw_sample, line_idx, routing_map, full=False)
        scene = build_scene_context_text(raw_sample, feats, cfg)
        record["priv_context_text"] = _scene_context_only(scene)
        record["teacher_images"] = [{"path": p} for p in student_imgs]
        ei["features"] = feats
        ei["priv_context_mode"] = "text_routed_no_answer"
        record["extra_info"] = ei
        return record

    if variant == "text_full":
        feats = features_for_sample(raw_sample, line_idx, routing_map, full=True)
        scene = build_scene_context_text(raw_sample, feats, cfg)
        record["priv_context_text"] = _merge_priv_text(scene, answer)
        record["teacher_images"] = [{"path": p} for p in student_imgs]
        ei["features"] = feats
        ei["priv_context_mode"] = "text_full"
        record["extra_info"] = ei
        return record

    if variant == "image_routed":
        feats = features_for_sample(raw_sample, line_idx, routing_map, full=False)
        tpaths = build_teacher_image_paths(raw_sample, feats, cfg)
        record["priv_context_text"] = build_reference_answer_block(answer)
        record["teacher_images"] = [{"path": p} for p in tpaths]
        ei["features"] = feats
        ei["priv_context_mode"] = "image_routed"
        record["extra_info"] = ei
        return record

    if variant == "image_full":
        feats = features_for_sample(raw_sample, line_idx, routing_map, full=True)
        tpaths = build_teacher_image_paths(raw_sample, feats, cfg)
        record["priv_context_text"] = build_reference_answer_block(answer)
        record["teacher_images"] = [{"path": p} for p in tpaths]
        ei["features"] = feats
        ei["priv_context_mode"] = "image_full"
        record["extra_info"] = ei
        return record

    raise ValueError(f"unknown priv variant: {variant}")
