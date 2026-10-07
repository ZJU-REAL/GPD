#!/usr/bin/env python3
"""
Stage 1 – Feature Router

Calls a model on the question text (without images) for each sample to decide
which 3D features are required to answer it:
  depth    - depth maps: for depth prediction, distance estimation, near/far judgement
  semantic - semantic segmentation: for object recognition, counting, spatial relations
  bev      - bird's-eye view: for direction reasoning, viewpoint transformation, floor layout

Output: per-sample routing decision JSONL, consumed by *_adaptive_eval.py.

Two routing methods are supported:
  --method llm   (default) calls Qwen3-VL text-only reasoning for classification
  --method rule  rule-based mapping from question_type / type field (no inference, very fast)

Usage:
  # LLM routing (~5-10 min for 1000 samples)
  CUDA_VISIBLE_DEVICES=0 python data_prep/feature_router.py \\
      --jsonl data/vsi_rl_30k_16frames_ready.jsonl \\
      --output data_prep/routing/vsi_routing_1k.jsonl \\
      --num_samples 1000 --method llm --gpu 0

  # Rule routing (completes in seconds, no GPU required)
  python data_prep/feature_router.py \\
      --jsonl data/vsi_rl_30k_16frames_ready.jsonl \\
      --output data_prep/routing/vsi_routing_1k_rule.jsonl \\
      --num_samples 1000 --method rule
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

MODEL_PATH = os.environ.get(
    "ROUTING_MODEL_PATH",
    "Qwen/Qwen3-VL-4B-Instruct",
)

# ─────────────────────── Routing Prompt ───────────────────────────────

ROUTING_SYSTEM = (
    "You are a feature selector for a 3D spatial reasoning system. "
    "Your task is to decide which additional visual features would help answer a question."
)

ROUTING_TEMPLATE = """Given the following spatial reasoning question, decide which additional 3D visual features are needed to answer it correctly.

Available features:
  depth    – Colorized depth maps showing per-pixel distance from the camera.
             Useful for: depth prediction, distance estimation, "how far", "how close", metric values.
  semantic – Instance segmentation maps where each object instance has a unique color.
             Useful for: identifying specific named objects, counting objects, spatial relations between named objects.
  bev      – Bird's-eye view (top-down floor plan) of the scene.
             Useful for: direction reasoning ("to the left/right when facing X"), viewpoint transformation, floor layout.

Question:
{question}

Instructions:
- Output ONLY the feature names that are directly needed, space-separated.
- Use exactly these names: depth  semantic  bev
- If the question can be answered from the images alone with no additional features, output: none
- Do NOT explain, do NOT output anything else.

Examples:
  Q: "What is the estimated depth of the chair?"               → depth
  Q: "Which is farther, the sofa or the lamp?"                 → depth
  Q: "If I face the window, is the door on my left or right?"  → bev
  Q: "How many chairs are visible?"                            → semantic
  Q: "Which object is larger, the table or the cabinet?"       → semantic
  Q: "From the bed, looking toward the closet, where is the window?" → bev semantic

Your answer:"""

# ─────────────────────── Rule mapping tables ───────────────────────────────
# question_type (VSI) / type (SPAR) → features

RULE_MAP_VSI = {
    "relative_direction_object":  ["bev", "semantic"],
    "relative_distance_object":   ["depth"],
    "relative_size_object":       ["semantic"],
    "appearance_order":           ["semantic"],
    "relative_count":             ["semantic"],
}

RULE_MAP_SPAR = {
    # depth-related
    "depth_prediction_oc":              ["depth"],
    "depth_prediction_oc_mv":           ["depth"],
    "depth_prediction_oo":              ["depth"],
    "depth_prediction_oo_mv":           ["depth"],
    "depth_prediction_oo_video":        ["depth"],
    # distance-related
    "distance_prediction_oc":           ["depth"],
    "distance_prediction_oc_mv":        ["depth"],
    "distance_prediction_oo":           ["depth"],
    "distance_prediction_oo_mv":        ["depth"],
    "distance_prediction_oo_video":     ["depth"],
    "distance_infer_center_oc_mv":      ["depth"],
    "distance_infer_center_oo":         ["depth"],
    "distance_infer_center_oo_mv":      ["depth"],
    "distance_infer_center_oo_video":   ["depth"],
    # spatial relations / object
    "obj_spatial_relation_oc_mv":       ["bev", "semantic"],
    "obj_spatial_relation_oo":          ["bev", "semantic"],
    "obj_spatial_relation_oo_mv":       ["bev", "semantic"],
    # viewpoint / imagination
    "spatial_imagination_oc":           ["bev", "semantic"],
    "spatial_imagination_oc_mv":        ["bev", "semantic"],
    "spatial_imagination_oo":           ["bev", "semantic"],
    "spatial_imagination_oo_mv":        ["bev", "semantic"],
    # camera motion
    "camera_motion_infer":              ["bev"],
    # position / size
    "position_matching":                ["bev", "semantic"],
    "room_size":                        ["bev"],
}

# MindCube type → features (bev excluded; MindCube BEV quality is insufficient)
RULE_MAP_MINDCUBE = {
    # Single object, 4 viewpoints; asks what is visible from a given viewpoint → semantic object recognition
    "0_frame": ["semantic"],
    "1_frame": ["semantic"],
    "2_frame": ["semantic"],
    "3_frame": ["semantic"],
    # Camera motion direction inference → depth assists motion estimation
    "two_view_clockwise":        ["depth"],
    "two_view_counterclockwise": ["depth"],
    "two_view_opposite":         ["depth"],
    # Multi-view scene; asks object positions or nearest object → semantic + depth
    "four_view":  ["semantic", "depth"],
    "three_view": ["semantic", "depth"],
    "1":          ["semantic", "depth"],
    "2":          ["semantic", "depth"],
    "3":          ["semantic", "depth"],
    "general":    ["semantic", "depth"],
}

VALID_FEATURES = {"depth", "semantic", "bev"}


def parse_llm_output(text: str) -> list[str]:
    """Extract valid feature label list from model output."""
    text = text.strip().lower()
    if "none" in text:
        return []
    found = [f for f in VALID_FEATURES if f in text]
    return found


def rule_route(sample: dict) -> list[str]:
    """Rule-based routing using question_type / type field (supports VSI / SPAR / MindCube)."""
    qt = sample.get("question_type", sample.get("type", ""))
    features = (
        RULE_MAP_VSI.get(qt)
        or RULE_MAP_SPAR.get(qt)
        or RULE_MAP_MINDCUBE.get(qt)
        or []
    )
    return list(features)


def load_samples(
    jsonl_path: str,
    num_samples: Optional[int],
    id_list: Optional[list[int]] = None,
) -> list[dict]:
    """Load samples and inject _line_idx (0-based row index in the original JSONL) for downstream alignment.

    When id_list is not None, only samples whose row index is in id_list are kept
    (num_samples is ignored in this case).
    """
    with open(jsonl_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    for idx, s in enumerate(samples):
        s["_line_idx"] = idx

    # Filter by id_list first (from prepare_data.py --export_indices)
    if id_list is not None:
        id_set = set(id_list)
        samples = [s for s in samples if s["_line_idx"] in id_set]
        print(f"id_list filter: kept {len(samples)} samples (requested {len(id_list)})")
        return samples

    if num_samples and num_samples < len(samples):
        by_type: dict[str, list] = defaultdict(list)
        for s in samples:
            key = s.get("question_type", s.get("type", ""))
            by_type[key].append(s)
        result = []
        for t, lst in by_type.items():
            k = max(1, round(len(lst) / len(samples) * num_samples))
            result.extend(lst[:k])
        return result[:num_samples]
    return samples


def run_llm_routing(samples: list[dict], gpu: int, output_path: str, model_path: str):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    mp = str(Path(model_path).resolve())
    if not Path(mp).is_dir():
        raise FileNotFoundError(f"Routing model directory not found: {mp}")

    device = f"cuda:{gpu}"
    print(f"Loading model: {mp}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(mp)
    print("Model loaded\n")

    stats: dict[str, int] = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="[Feature Routing]"):
            # Strip <image> tags; keep only the question text
            raw_text = sample["conversations"][0]["value"]
            q_text   = re.sub(r"(<image>\s*)+", "", raw_text).strip()

            prompt = ROUTING_TEMPLATE.format(question=q_text)
            messages = [
                {"role": "system", "content": ROUTING_SYSTEM},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ]

            try:
                text_prompt = processor.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                text_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=16,
                    do_sample=False, temperature=None, top_p=None,
                )
            out = out[:, inputs["input_ids"].shape[1]:]
            response = processor.decode(out[0], skip_special_tokens=True).strip()

            features = parse_llm_output(response)

            rec = {
                "id":            sample.get("id", ""),
                "line_idx":      sample.get("_line_idx", -1),
                "question_type": sample.get("question_type", sample.get("type", "")),
                "features":      features,
                "raw_response":  response,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            key = "+".join(sorted(features)) if features else "none"
            stats[key] += 1

    print(f"\nRouting complete → {output_path}")
    print("Feature distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k or 'none':30s}: {v}")


def run_rule_routing(samples: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    stats: dict[str, int] = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in samples:
            features = rule_route(sample)
            rec = {
                "id":            sample.get("id", ""),
                "line_idx":      sample.get("_line_idx", -1),
                "question_type": sample.get("question_type", sample.get("type", "")),
                "features":      features,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            key = "+".join(sorted(features)) if features else "none"
            stats[key] += 1

    print(f"Rule routing complete → {output_path}")
    print("Feature distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k or 'none':30s}: {v}")


def _routing_key(rec_or_sample: dict) -> str | None:
    """Use line_idx as primary key when merging routing results (dataset ids such as SPAR may not be unique)."""
    idx = rec_or_sample.get("line_idx", rec_or_sample.get("_line_idx"))
    if idx is not None and int(idx) >= 0:
        return f"idx:{int(idx)}"
    rid = rec_or_sample.get("id")
    if rid:
        return f"id:{rid}"
    return None


def load_existing_routing(path: str) -> dict:
    """Load an existing routing file; returns {routing_key: record}."""
    existing: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = _routing_key(rec)
            if key:
                existing[key] = rec
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",        required=True,  help="Input JSONL file")
    parser.add_argument("--output",       required=True,  help="Output routing result JSONL")
    parser.add_argument("--method",       default="llm",  choices=["llm", "rule"])
    parser.add_argument("--num_samples",  type=int, default=None)
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH,
        help="Local Qwen3-VL directory (defaults to ROUTING_MODEL_PATH env var or ~/models/...)",
    )
    parser.add_argument(
        "--from_routing",
        default=None,
        help="Path to an existing routing file; the script only routes samples missing from it, "
             "then merges new results with the existing ones and writes to --output",
    )
    parser.add_argument(
        "--id_list",
        default=None,
        help="Path to a JSON file whose content is a list of original row indices "
             "(generated by prepare_data.py --export_indices). "
             "When specified, only those rows are routed; --num_samples is ignored.",
    )
    parser.add_argument(
        "--redo_indices",
        default=None,
        help="JSON array file listing line_idx values that must be re-run through LLM/rule "
             "(used with --from_routing to overwrite old results)",
    )
    args = parser.parse_args()

    # Load the specified row-index list (if provided)
    id_list = None
    if args.id_list:
        with open(args.id_list, encoding="utf-8") as f:
            id_list = json.load(f)
        print(f"Loaded {len(id_list)} target row indices from {args.id_list}")

    redo_set: set[int] = set()
    if args.redo_indices:
        with open(args.redo_indices, encoding="utf-8") as f:
            redo_set = {int(x) for x in json.load(f)}
        print(f"Loaded {len(redo_set)} forced-rerun line_idx values from {args.redo_indices}")

    samples = load_samples(args.jsonl, args.num_samples, id_list=id_list)
    print(f"Raw samples: {len(samples)}  method={args.method}")

    if args.from_routing:
        existing = load_existing_routing(args.from_routing)
        print(f"Existing routing: {len(existing)} entries (from {args.from_routing})")
        # Keep samples that are missing from the existing routing or marked for re-run
        new_samples = [
            s for s in samples
            if not existing.get(_routing_key(s))
            or int(s.get("_line_idx", -1)) in redo_set
        ]
        print(f"Samples to route / re-run: {len(new_samples)}")
    else:
        existing = {}
        new_samples = samples

    # Merged output must follow the full JSONL order, not limited to the id_list subset
    all_samples = load_samples(args.jsonl, None, id_list=None)

    if new_samples:
        tmp_output = args.output + ".tmp"
        if args.method == "llm":
            run_llm_routing(new_samples, args.gpu, tmp_output, args.model_path)
        else:
            run_rule_routing(new_samples, tmp_output)

        # Load new routing results
        with open(tmp_output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = _routing_key(rec)
                if key:
                    existing[key] = rec
        import os as _os
        _os.remove(tmp_output)
    else:
        print("All samples already routed; no re-inference needed")

    # Write merged results in original sample order
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for s in all_samples:
            key = _routing_key(s)
            rec = existing.get(key) if key else None
            if rec:
                rec = dict(rec)
                rec["line_idx"] = s.get("_line_idx", rec.get("line_idx", -1))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"\nMerged output: wrote {written}/{len(all_samples)} entries → {args.output}")
    if written < len(all_samples):
        print(f"⚠️  {len(all_samples) - written} entries not written; check line_idx / id")


if __name__ == "__main__":
    main()
