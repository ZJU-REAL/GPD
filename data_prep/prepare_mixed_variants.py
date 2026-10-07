#!/usr/bin/env python3
"""
Mixed-15k JSONL → 7 sets of GRPO parquet files (each with train/val splits).

Variants (see privilege_utils.py):
  1. pure_grpo              — no privilege
  2. text_routed            — routed 3D text + reference answer
  3. text_routed_no_answer  — routed 3D text, no reference answer
  4. text_full              — full 3D text + reference answer
  5. image_full             — full 3D images + reference answer (teacher_images)
  6. image_routed           — routed 3D images + reference answer
  7. answer_only            — reference answer only

Split: 95/5 within each dataset, then merged into train / val.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

try:
    import datasets as hf_datasets

    _HAS_DS = True
except ImportError:
    _HAS_DS = False

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privilege_utils import PRIV_VARIANTS, apply_priv_variant, load_routing_map

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant for 3D spatial reasoning. "
    "Think step by step using <think> </think> tags, "
    "then give your final answer inside <answer> </answer> tags."
)

COT_USER_SUFFIX = (
    "\nThink step by step and provide your reasoning inside <think> </think> tags. "
    "Then give your final answer inside <answer> </answer> tags. "
    "For example: <answer>A</answer>"
)

NUMERIC_SUFFIX = (
    "\nThink step by step and provide your reasoning inside <think> </think> tags. "
    "Then give your final answer inside <answer> </answer> tags with a single number or value."
)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_parquet(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _HAS_DS:
        hf_datasets.Dataset.from_list(rows).to_parquet(path)
    else:
        pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"  Saved {len(rows)} → {path}")


def extract_answer(raw: dict) -> str:
    if raw.get("gt_answer"):
        return str(raw["gt_answer"]).strip()
    for turn in reversed(raw.get("conversations", [])):
        if turn.get("from") == "gpt":
            return turn["value"].strip()
    return ""


def build_image_dicts(paths: list[str]) -> list[dict]:
    return [{"path": p} for p in paths]


def build_problem_with_placeholders(text: str, n_images: int) -> str:
    existing = text.count("<image>")
    if existing == n_images:
        return text
    if existing == 0:
        return "\n".join(["<image>"] * n_images) + "\n" + text
    text_clean = text.replace("<image>", "").strip()
    return "\n".join(["<image>"] * n_images) + "\n" + text_clean


def _is_mc_format(answer_format: str) -> bool:
    return answer_format.lower() in ("mc", "select")


def build_prompt(problem: str, n_images: int, answer_format: str) -> list[dict]:
    prob = build_problem_with_placeholders(problem, n_images)
    suffix = COT_USER_SUFFIX if _is_mc_format(answer_format) else NUMERIC_SUFFIX
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": prob + suffix},
    ]


def convert_row(raw: dict, idx: int) -> Optional[dict]:
    imgs = raw.get("image", [])
    if isinstance(imgs, str):
        imgs = [imgs]
    if not imgs:
        return None
    conv = raw.get("conversations", [])
    question = next((t["value"] for t in conv if t.get("from") == "human"), "")
    if not question:
        return None
    answer = extract_answer(raw)
    if not answer:
        return None

    ds = raw.get("dataset", "vsi")
    task_type = raw.get("question_type", raw.get("type", ""))
    af = raw.get("answer_format", "mc")
    sample_id = raw.get("id", "")
    if not sample_id and imgs:
        sample_id = Path(imgs[0]).parts[-2]

    n = len(imgs)
    return {
        "data_source": f"GPD/{ds}",
        "prompt": build_prompt(question, n, af),
        "images": build_image_dicts(imgs),
        "teacher_images": build_image_dicts(imgs),
        "answer": answer,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": raw.get("_line_idx", idx),
            "answer": answer,
            "id": sample_id,
            "type": task_type,
            "dataset": ds,
            "answer_format": af,
        },
        "priv_context_text": "",
    }


def split_by_dataset(
    records: list[dict],
    raw_rows: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Split train/val independently within each dataset."""
    rng = random.Random(seed)
    by_ds: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for rec, raw in zip(records, raw_rows):
        ds = rec["extra_info"]["dataset"]
        by_ds[ds].append((rec, raw))

    train_recs, val_recs = [], []
    train_raw, val_raw = [], []
    for ds, pairs in by_ds.items():
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_part = shuffled[:n_val]
        train_part = shuffled[n_val:]
        for rec, raw in train_part:
            train_recs.append(rec)
            train_raw.append(raw)
        for rec, raw in val_part:
            val_recs.append(rec)
            val_raw.append(raw)
        print(f"  split {ds}: train={len(train_part)} val={len(val_part)}")
    return train_recs, val_recs, train_raw, val_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/PATH/TO/SOURCE/mixed_15k.jsonl",
    )
    ap.add_argument(
        "--output_dir",
        default="data/mixed_15k_privfix",
    )
    ap.add_argument(
        "--routing_file",
        default="data/mixed_15k_routing_llm.jsonl",
        help="Routing JSONL produced by zero-shot feature_router (used for text_routed / text_routed_no_answer / image_routed)",
    )
    ap.add_argument(
        "--allow_rule_fallback",
        action="store_true",
        help="Fall back to rule_route when routing file is missing entries; by default missing entries raise an error",
    )
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(PRIV_VARIANTS),
        choices=list(PRIV_VARIANTS),
    )
    args = ap.parse_args()

    raw_all = load_jsonl(args.input)
    print(f"Loaded {len(raw_all)} from {args.input}")

    records: list[dict] = []
    raw_kept: list[dict] = []
    skipped = 0
    for i, row in enumerate(raw_all):
        rec = convert_row(row, i)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
            raw_kept.append(row)
    print(f"Converted {len(records)}, skipped {skipped}")

    routing_map = load_routing_map(args.routing_file)
    routed_variants = {"text_routed", "text_routed_no_answer", "image_routed"} & set(args.variants)
    if routed_variants:
        if not routing_map:
            raise SystemExit(
                f"text_routed / text_routed_no_answer / image_routed require a routing file, "
                f"but none was found or it is empty: {args.routing_file}\n"
                "Please run first: data_prep/feature_router.py --jsonl mixed_15k.jsonl "
                "--output mixed_15k_routing_llm.jsonl --method llm"
            )
        missing = [i for i in range(len(raw_kept)) if i not in routing_map]
        if missing:
            msg = (
                f"Routing file is missing {len(missing)}/{len(raw_kept)} entries "
                f"(e.g. line_idx={missing[:5]})"
            )
            if args.allow_rule_fallback:
                print(f"WARNING: {msg}; falling back to rule_route for missing entries")
            else:
                raise SystemExit(msg + "; add --allow_rule_fallback to fall back to rule routing")
    if routing_map:
        print(f"Routing map: {len(routing_map)} entries from {args.routing_file}")
    elif routed_variants:
        print("Routing: rule-based fallback (feature_router.rule_route)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print(f"\n=== Variant: {variant} ===")
        variant_records = []
        for rec, raw in zip(records, raw_kept):
            r = json.loads(json.dumps(rec))  # deep copy via json
            apply_priv_variant(r, raw, variant, routing_map)
            variant_records.append(r)

        tr, va, tr_raw, va_raw = split_by_dataset(
            variant_records, raw_kept, args.val_ratio, args.seed
        )
        prefix = out_dir / variant
        save_parquet(tr, str(prefix) + "_train.parquet")
        save_parquet(va, str(prefix) + "_val.parquet")

        n_priv = sum(1 for r in tr if (r.get("priv_context_text") or "").strip())
        n_timg = sum(1 for r in tr if len(r.get("teacher_images", [])) > len(r.get("images", [])))
        print(f"  train priv_context non-empty: {n_priv}/{len(tr)}")
        print(f"  train teacher_images > student images: {n_timg}/{len(tr)}")

    # Write manifest
    manifest = {
        "input": args.input,
        "routing_file": args.routing_file if routing_map else None,
        "variants": args.variants,
        "val_ratio": args.val_ratio,
        "counts": {
            "total": len(records),
            "by_dataset": dict(Counter(r["extra_info"]["dataset"] for r in records)),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest → {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
