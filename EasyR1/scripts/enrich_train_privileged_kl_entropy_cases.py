#!/usr/bin/env python3
"""Back-fill question text and frame images for existing ``privileged_kl_entropy`` dump samples.

Usage::

  python scripts/enrich_train_privileged_kl_entropy_cases.py \\
      --checkpoint-dir /path/to/checkpoint

  python scripts/enrich_train_privileged_kl_entropy_cases.py \\
      --privileged-kl-dir /path/to/checkpoint/privileged_kl_entropy \\
      --parquet /path/to/vsi_10k_train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EASYR1 = Path(__file__).resolve().parents[1]
if str(_EASYR1) not in sys.path:
    sys.path.insert(0, str(_EASYR1))

from verl.trainer.privileged_kl_case_assets import (  # noqa: E402
    _normalize_image_paths,
    _priv_prefix_from_summary,
    match_parquet_row,
    write_case_assets,
)
from verl.utils.vsi_zeroshot_prompts import extract_vsi_q_text  # noqa: E402


def _qtext_from_row(row: pd.Series) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = list(prompt)
    return extract_vsi_q_text({"prompt": prompt}, "prompt")


def _iter_sample_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for step_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")):
        out.extend(sorted(p for p in step_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")))
    return out


def enrich_sample(sample_dir: Path, df: pd.DataFrame, *, force: bool = False) -> dict:
    summary_path = sample_dir / "summary.json"
    if not summary_path.is_file():
        return {"status": "skip", "reason": "no summary.json"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (sample_dir / "case.json").is_file() and not force:
        return {"status": "skip", "reason": "case.json exists"}

    rollout_path = sample_dir / "rollouts_decoded.txt"
    rollout_text = rollout_path.read_text(encoding="utf-8") if rollout_path.is_file() else ""

    q_text = str(summary.get("q_text") or "").strip()
    image_paths = _normalize_image_paths(summary.get("image_paths"))
    priv_full = ""
    dataset_meta = None
    confidence = "dump"
    note = "from summary.json"

    if not q_text or not image_paths:
        row_idx, confidence, note = match_parquet_row(
            df,
            ground_truth=str(summary.get("ground_truth") or ""),
            priv_context_preview=_priv_prefix_from_summary(summary),
            rollout_text=rollout_text,
            qtext_fn=lambda i: _qtext_from_row(df.loc[i]),
        )
        if row_idx is None:
            return {"status": "fail", "reason": note}
        row = df.loc[row_idx]
        q_text = _qtext_from_row(row)
        image_paths = _normalize_image_paths(row["images"])
        priv_full = str(row.get("priv_context_text") or "")
        extra = row.get("extra_info")
        if isinstance(extra, dict):
            dataset_meta = extra

    if not q_text or not image_paths:
        return {"status": "fail", "reason": "empty q_text or images after match"}

    if not priv_full:
        priv_full = str(summary.get("priv_context_text_preview") or "")

    write_case_assets(
        sample_dir,
        q_text=q_text,
        ground_truth=str(summary.get("ground_truth") or ""),
        image_paths=image_paths,
        priv_context_text=priv_full,
        dataset_meta=dataset_meta,
        match_confidence=confidence,
        match_note=note,
    )
    return {
        "status": "ok",
        "confidence": confidence,
        "note": note,
        "n_images": len(image_paths),
        "dataset": dataset_meta,
    }


def _write_index(root: Path, rows: list[dict]) -> None:
    index_path = root / "cases_index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich privileged_kl_entropy samples with question + images.")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--privileged-kl-dir", type=Path, default=None)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("/PATH/TO/WORKSPACE/data/vsi_10k_train.parquet"),
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing case.json/images/")
    args = ap.parse_args()

    if args.privileged_kl_dir is not None:
        root = args.privileged_kl_dir.resolve()
    elif args.checkpoint_dir is not None:
        root = args.checkpoint_dir.resolve() / "privileged_kl_entropy"
    else:
        ap.error("need --checkpoint-dir or --privileged-kl-dir")

    if not root.is_dir():
        raise FileNotFoundError(root)
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)

    df = pd.read_parquet(args.parquet)
    index_rows: list[dict] = []
    stats = {"ok": 0, "skip": 0, "fail": 0}

    for sample_dir in _iter_sample_dirs(root):
        rel = sample_dir.relative_to(root)
        result = enrich_sample(sample_dir, df, force=args.force)
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        if result["status"] == "ok":
            case = json.loads((sample_dir / "case.json").read_text(encoding="utf-8"))
            index_rows.append(
                {
                    "step": sample_dir.parent.name,
                    "sample": sample_dir.name,
                    "ground_truth": case.get("ground_truth"),
                    "match_confidence": case.get("match_confidence"),
                    "match_note": case.get("match_note"),
                    "dataset": case.get("dataset"),
                    "n_images": case.get("n_images"),
                    "question_file": str(rel / "question.txt"),
                    "images_dir": str(rel / "images"),
                }
            )
            print(f"OK {rel} [{case.get('match_confidence')}] images={case.get('n_images')}")
        elif result["status"] == "skip":
            print(f"SKIP {rel}: {result.get('reason')}")
        else:
            print(f"FAIL {rel}: {result.get('reason')}")

    _write_index(root, index_rows)
    print(
        f"Done: ok={stats.get('ok', 0)} skip={stats.get('skip', 0)} "
        f"fail={stats.get('fail', 0)} index={root / 'cases_index.jsonl'}"
    )


if __name__ == "__main__":
    main()
