#!/usr/bin/env python3
"""Summarize KL / entropy / logp relationships from training privileged_kl_entropy dumps.

Reads ``priv_kl_plots/*_per_token.csv`` for each sample and writes:
  - ``analysis/token_level.parquet``: all token rows (with phase column)
  - ``analysis/per_sample_summary.csv``: sequence-level statistics per incorrect trajectory
  - ``analysis/correlations_pearson.csv`` / ``correlations_spearman.csv``
  - ``analysis/report.txt``: human-readable summary

Usage::

  python scripts/summarize_train_privileged_kl_entropy.py \\
      --privileged-kl-dir /path/to/checkpoint/privileged_kl_entropy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLS = [
    "logp_S",
    "logp_T",
    "kl_B",
    "abs_kl_B",
    "low_var_kld",
    "H_stu",
    "H_tea",
    "H_tea_minus_H_stu",
]


def _find_csv(sample_dir: Path) -> Path | None:
    plots = sample_dir / "priv_kl_plots"
    if plots.is_dir():
        cands = sorted(plots.glob("*_per_token.csv"))
        if cands:
            return cands[0]
    cands = sorted(sample_dir.glob("*_per_token.csv"))
    return cands[0] if cands else None


def _load_sample_csv(sample_dir: Path) -> pd.DataFrame | None:
    csv_path = _find_csv(sample_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    rename = {
        "logp_student_ctx_A": "logp_S",
        "logp_teacher_ctx_B": "logp_T",
        "kl_ref_minus_student_B_minus_A": "kl_B",
        "low_var_kld_proxy": "low_var_kld",
        "entropy_student_ctx_A": "H_stu",
        "entropy_teacher_ctx_B": "H_tea",
        "entropy_diff_B_minus_A": "H_tea_minus_H_stu",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "H_tea_minus_H_stu" not in df.columns and {"H_stu", "H_tea"} <= set(df.columns):
        df["H_tea_minus_H_stu"] = df["H_tea"] - df["H_stu"]
    if "abs_kl_B" not in df.columns and "kl_B" in df.columns:
        df["abs_kl_B"] = df["kl_B"].abs()
    parts = sample_dir.parts
    step = next((p for p in parts if p.startswith("step_")), "")
    sample = sample_dir.name
    df["step"] = step
    df["sample"] = sample
    df["sample_dir"] = str(sample_dir)

    # phase: split by </think> first, fallback to <answer>.
    tok = df.get("token_decoded", pd.Series([""] * len(df))).astype(str).str.replace("?", "", regex=False)
    norm = [t.strip() for t in tok]

    answer_start = None

    for i in range(len(norm) - 2):
        tri = norm[i : i + 3]
        if tri == ["</", "think", ">"]:
            answer_start = i + 3
            break

    if answer_start is None:
        for i in range(len(norm) - 2):
            tri = norm[i : i + 3]
            if tri == ["<", "answer", ">"]:
                answer_start = i + 3
                break

    phase = np.array(["reasoning"] * len(df), dtype=object)
    if answer_start is not None:
        phase[answer_start:] = "answer"
    df["phase"] = phase
    return df


def _iter_samples(root: Path) -> list[Path]:
    out: list[Path] = []
    for step_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")):
        out.extend(sorted(p for p in step_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")))
    return out


def _corr_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in METRIC_COLS if c in df.columns]
    sub = df[cols].astype(float)
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    return pearson, spearman


def _per_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (step, sample), g in df.groupby(["step", "sample"], sort=False):
        row: dict = {"step": step, "sample": sample, "n_tokens": len(g)}
        if "case.json" in g.columns:
            pass
        case_path = Path(g["sample_dir"].iloc[0]) / "case.json"
        if case_path.is_file():
            case = json.loads(case_path.read_text(encoding="utf-8"))
            row["ground_truth"] = case.get("ground_truth")
            row["match_confidence"] = case.get("match_confidence")
            ds = case.get("dataset") or {}
            row["dataset_id"] = ds.get("id")
        for phase in ("all", "reasoning", "answer"):
            if phase == "all":
                sub = g
            else:
                sub = g[g["phase"] == phase]
            if sub.empty:
                continue
            pref = "" if phase == "all" else f"{phase}_"
            row[f"{pref}n_tokens"] = len(sub)
            row[f"{pref}mean_abs_kl_B"] = float(sub["abs_kl_B"].mean())
            row[f"{pref}mean_kl_B"] = float(sub["kl_B"].mean())
            row[f"{pref}mean_low_var_kld"] = float(sub["low_var_kld"].mean())
            row[f"{pref}mean_H_stu"] = float(sub["H_stu"].mean())
            row[f"{pref}mean_H_tea"] = float(sub["H_tea"].mean())
            row[f"{pref}mean_logp_S"] = float(sub["logp_S"].mean())
            row[f"{pref}mean_logp_T"] = float(sub["logp_T"].mean())
            row[f"{pref}p95_abs_kl_B"] = float(sub["abs_kl_B"].quantile(0.95))
            # Simple composite metrics related to KL
            row[f"{pref}corr_abs_kl_H_tea"] = float(sub["abs_kl_B"].corr(sub["H_tea"]))
            row[f"{pref}corr_abs_kl_H_stu"] = float(sub["abs_kl_B"].corr(sub["H_stu"]))
            row[f"{pref}corr_abs_kl_logp_gap"] = float(
                sub["abs_kl_B"].corr((sub["logp_T"] - sub["logp_S"]).abs())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    *,
    n_samples: int,
    n_tokens: int,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    per_sample: pd.DataFrame,
    token_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("privileged_kl_entropy sampling metric relationship summary\n")
    lines.append(f"samples (incorrect trajectories): {n_samples}")
    lines.append(f"total tokens: {n_tokens}\n")

    lines.append("=== Metric descriptions ===")
    lines.append("logp_S / logp_T: log pi of the selected token under student / teacher context")
    lines.append("kl_B = logp_T - logp_S  (aligned with training ref_logp - student_logp)")
    lines.append("low_var_kld: TRL low-variance KL proxy (per-token scalar before kl_mask/kl_coef)")
    lines.append("H_stu / H_tea: full-vocabulary Shannon entropy (normalized before writing to CSV)")
    lines.append("phase=answer: tokens after </think>; fallback to tokens after <answer>\n")

    def _fmt_corr(name: str, mat: pd.DataFrame) -> None:
        lines.append(f"--- {name} (correlated with |kl_B|/kl_B/low_var_kld) ---")
        for target in ("abs_kl_B", "kl_B", "low_var_kld"):
            if target not in mat.columns:
                continue
            lines.append(f"\n  target: {target}")
            s = mat[target].drop(labels=[target], errors="ignore").sort_values(key=abs, ascending=False)
            for k, v in s.head(8).items():
                lines.append(f"    {k:18s} {v:+.4f}")

    lines.append("\n=== Global token-level Pearson correlation ===")
    _fmt_corr("Pearson", pearson)
    lines.append("\n=== Global token-level Spearman correlation ===")
    _fmt_corr("Spearman", spearman)

    if "phase" in token_df.columns:
        lines.append("\n=== Phase-level means (pooled over all tokens) ===")
        for phase in ("reasoning", "answer"):
            sub = token_df[token_df["phase"] == phase]
            if sub.empty:
                continue
            lines.append(
                f"  {phase}: n={len(sub)}  mean|kl_B|={sub['abs_kl_B'].mean():.4f}  "
                f"mean_H_stu={sub['H_stu'].mean():.4f}  mean_H_tea={sub['H_tea'].mean():.4f}  "
                f"mean_logp_S={sub['logp_S'].mean():.4f}  mean_logp_T={sub['logp_T'].mean():.4f}"
            )

    if not per_sample.empty and "answer_mean_abs_kl_B" in per_sample.columns:
        lines.append("\n=== Sequence-level (per incorrect trajectory, answer phase) ===")
        lines.append(f"  answer mean|kl_B|  median={per_sample['answer_mean_abs_kl_B'].median():.4f}  "
                     f"mean={per_sample['answer_mean_abs_kl_B'].mean():.4f}")
        if "reasoning_mean_abs_kl_B" in per_sample.columns:
            lines.append(f"  reasoning mean|kl_B| median={per_sample['reasoning_mean_abs_kl_B'].median():.4f}  "
                         f"mean={per_sample['reasoning_mean_abs_kl_B'].mean():.4f}")

    lines.append("\nOutput files:")
    lines.append(f"  {out_dir / 'token_level.parquet'}")
    lines.append(f"  {out_dir / 'per_sample_summary.csv'}")
    lines.append(f"  {out_dir / 'correlations_pearson.csv'}")
    lines.append(f"  {out_dir / 'correlations_spearman.csv'}")

    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize KL/entropy/logp relations for privileged_kl_entropy dumps.")
    ap.add_argument("--privileged-kl-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory; defaults to <privileged-kl-dir>/analysis")
    args = ap.parse_args()

    root = args.privileged_kl_dir.resolve()
    out_dir = (args.out_dir or (root / "analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for sample_dir in _iter_samples(root):
        df = _load_sample_csv(sample_dir)
        if df is None:
            missing.append(str(sample_dir))
            continue
        frames.append(df)

    if not frames:
        raise SystemExit(f"No *_per_token.csv found under {root}")

    token_df = pd.concat(frames, ignore_index=True)
    pearson, spearman = _corr_tables(token_df)
    per_sample = _per_sample_summary(token_df)

    token_df.to_parquet(out_dir / "token_level.parquet", index=False)
    per_sample.to_csv(out_dir / "per_sample_summary.csv", index=False)
    pearson.to_csv(out_dir / "correlations_pearson.csv")
    spearman.to_csv(out_dir / "correlations_spearman.csv")

    meta = {
        "n_samples": int(per_sample.shape[0]),
        "n_tokens": int(token_df.shape[0]),
        "n_missing_csv": len(missing),
        "steps": sorted(token_df["step"].unique().tolist()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir,
        n_samples=int(per_sample.shape[0]),
        n_tokens=int(token_df.shape[0]),
        pearson=pearson,
        spearman=spearman,
        per_sample=per_sample,
        token_df=token_df,
    )

    print(f"OK samples={meta['n_samples']} tokens={meta['n_tokens']} -> {out_dir}")
    print((out_dir / "report.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
