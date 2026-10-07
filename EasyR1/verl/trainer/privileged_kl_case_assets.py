"""Organize question text and frame images (symlinks) for privileged_kl_entropy samples."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _normalize_image_paths(images: Any) -> list[str]:
    if images is None:
        return []
    if isinstance(images, np.ndarray):
        images = list(images)
    if not isinstance(images, (list, tuple)):
        images = [images]
    out: list[str] = []
    for item in images:
        if isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))
        elif isinstance(item, str):
            out.append(item)
    return out


def write_case_assets(
    sample_dir: Path,
    *,
    q_text: str,
    ground_truth: str,
    image_paths: list[str],
    priv_context_text: str = "",
    dataset_meta: Optional[dict[str, Any]] = None,
    match_confidence: str = "dump",
    match_note: str = "",
) -> Path:
    """Write ``question.txt`` / ``case.json`` and create symlinks for frames under ``images/``."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    images_dir = sample_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    linked: list[dict[str, str]] = []
    used_names: dict[str, int] = {}
    for src_s in image_paths:
        src = Path(src_s)
        if not src.is_file():
            continue
        name = src.name
        if name in used_names:
            used_names[name] += 1
            name = f"{src.stem}_{used_names[name]}{src.suffix}"
        else:
            used_names[name] = 0
        dst = images_dir / name
        os.symlink(src.resolve(), dst)
        linked.append({"src": str(src.resolve()), "link": str(dst.name)})

    question_path = sample_dir / "question.txt"
    question_path.write_text(q_text.strip() + "\n", encoding="utf-8")

    priv_path = sample_dir / "priv_context_text.txt"
    if priv_context_text.strip():
        priv_path.write_text(priv_context_text.strip() + "\n", encoding="utf-8")

    case = {
        "ground_truth": ground_truth,
        "q_text": q_text,
        "n_images": len(linked),
        "image_paths_src": [x["src"] for x in linked],
        "image_links": linked,
        "priv_context_text_file": priv_path.name if priv_context_text.strip() else None,
        "match_confidence": match_confidence,
        "match_note": match_note,
    }
    if dataset_meta:
        case["dataset"] = dataset_meta

    case_path = sample_dir / "case.json"
    case_path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return case_path


def _priv_prefix_from_summary(summary: dict[str, Any]) -> str:
    prev = str(summary.get("priv_context_text_preview") or "")
    if prev.endswith("..."):
        prev = prev[:-3]
    return prev


def _normalize_q_for_match(q_text: str) -> str:
    q = q_text.lower()
    q = re.sub(r"these are frames of a video\.\s*", "", q)
    q = re.sub(r"think step by step.*", "", q, flags=re.S)
    q = re.sub(r"answer with the option.*", "", q, flags=re.S)
    return q.strip()


def _score_rollout_against_question(rollout: str, q_text: str) -> float:
    rollout_l = rollout.lower()
    core = _normalize_q_for_match(q_text)
    opts_split = re.split(r"\boptions:\s*", core, maxsplit=1, flags=re.I)
    stem = opts_split[0]
    words = re.findall(r"[a-z']{4,}", stem)
    if not words:
        return 0.0
    hit = sum(1 for w in set(words) if w in rollout_l)
    grams = [stem[i : i + 12] for i in range(0, max(len(stem) - 11, 0), 6)]
    gram_hit = sum(1 for g in grams if g.strip() and g in rollout_l)
    return hit + 2.5 * gram_hit


def match_parquet_row(
    df: Any,
    *,
    ground_truth: str,
    priv_context_preview: str,
    rollout_text: str,
    qtext_fn: Any,
) -> tuple[Optional[int], str, str]:
    """Returns (row_index, confidence, note)."""
    prev = priv_context_preview
    if prev.endswith("..."):
        prev = prev[:-3]
    cands = df[(df["answer"] == ground_truth) & (df["priv_context_text"].str.startswith(prev))]
    if len(cands) == 0:
        return None, "missing", "no parquet row with matching answer + priv_context prefix"
    if len(cands) == 1:
        return int(cands.index[0]), "high", "unique priv_context prefix match"

    scored: list[tuple[float, int]] = []
    for idx in cands.index:
        q = qtext_fn(int(idx))
        scored.append((_score_rollout_against_question(rollout_text, q), int(idx)))
    scored.sort(reverse=True)
    best_s, best_i = scored[0]
    second_s = scored[1][0] if len(scored) > 1 else -1.0
    if best_s > second_s:
        return best_i, "medium", f"disambiguated by rollout overlap ({best_s:.1f}>{second_s:.1f})"
    return best_i, "low", f"ambiguous priv prefix; picked best rollout overlap ({best_s:.1f}≈{second_s:.1f})"
