#!/usr/bin/env python3
"""Convert training ``privileged_kl_entropy/step_*/sample_*`` dumps to per-token CSV / merged / paste_notes.

The training side (``trainer.log_privileged_kl_entropy=true``) only writes ``logits_summary.npz`` + ``summary.json``;
this script reuses ``export_vsi_logits_per_token_metrics`` and the post-processing annotators.

Usage::

  python scripts/export_train_privileged_kl_entropy.py \\
      --step-dir /path/to/checkpoint/privileged_kl_entropy/step_000030

  python scripts/export_train_privileged_kl_entropy.py \\
      --checkpoint-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_EASYR1 = Path(__file__).resolve().parents[1]
if str(_EASYR1) not in sys.path:
    sys.path.insert(0, str(_EASYR1))

_scripts = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _iter_sample_dirs(step_dir: Path) -> list[Path]:
    if not step_dir.is_dir():
        raise FileNotFoundError(step_dir)
    samples = sorted(p for p in step_dir.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    if samples:
        return samples
    if (step_dir / "logits_summary.npz").is_file():
        return [step_dir]
        raise FileNotFoundError(f"No sample_* subdirectories or logits_summary.npz found: {step_dir}")


def _resolve_step_dirs(checkpoint_dir: Path, steps: str | None) -> list[Path]:
    root = checkpoint_dir / "privileged_kl_entropy"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if steps:
        out = []
        for s in steps.split(","):
            s = s.strip()
            if not s:
                continue
            name = s if s.startswith("step_") else f"step_{int(s):06d}"
            p = root / name
            if p.is_dir():
                out.append(p)
            else:
                print(f"SKIP missing {p}")
        return out
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_"))


def postprocess_one_sample(sample_dir: Path, *, export_mod, post_mod, cases_dir_fp: Path | None, annotate: bool) -> None:
    if not (sample_dir / "logits_summary.npz").is_file():
        print(f"SKIP {sample_dir}: no logits_summary.npz")
        return

    fp_aligned = None
    if cases_dir_fp is not None:
        cand = cases_dir_fp / sample_dir.name
        if (cand / "logits_summary.npz").is_file():
            fp_aligned = cand

    if fp_aligned is not None:
        csv_p = post_mod.export_full_csv(sample_dir, fp_aligned)
    else:
        csv_p = export_mod.export_one_row(
            sample_dir,
            cases_dir_fp=None,
            write_csv=True,
            write_paste_notes=False,
            write_row_csv=True,
        )
        if csv_p is None:
            csv_p = sample_dir / "priv_kl_plots" / f"{sample_dir.name}_per_token.csv"

    post_mod.write_merged_txt(sample_dir, csv_p)
    summary = sample_dir / "summary.json"
    post_mod.write_paste_notes(sample_dir, csv_p, summary)

    if annotate:
        merged = sample_dir / f"{sample_dir.name}_kl_entropy_merged.txt"
        paste = sample_dir / f"{sample_dir.name}_kl_entropy_paste_notes.txt"
        _kl_path = _scripts / "annotate_kl_entropy_paste_notes.py"
        _sent_path = _scripts / "annotate_sentence_averages.py"
        _kl = _load_module("annotate_kl", _kl_path)
        _sent = _load_module("annotate_sent", _sent_path)
        _kl.annotate_metrics_file(merged, csv_p)
        _kl.annotate_metrics_file(paste, csv_p)
        notes = _sent.build_end_step_notes(csv_p)
        _sent.apply_to_file(merged, notes, csv_p)
        _sent.apply_to_file(paste, notes, csv_p)

    print(f"OK {sample_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export train privileged_kl_entropy dumps to per-token views.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--step-dir", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--steps", type=str, default=None)
    ap.add_argument("--cases-dir-fp", type=Path, default=None)
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    export_mod = _load_module("export_vsi", _scripts / "export_vsi_logits_per_token_metrics.py")
    post_mod = _load_module("post_vsi", _scripts / "vsi_logits_postprocess_batch.py")

    step_dirs = [args.step_dir.resolve()] if args.step_dir else _resolve_step_dirs(args.checkpoint_dir.resolve(), args.steps)

    for step_dir in step_dirs:
        print(f"=== {step_dir.name} ===")
        for sample_dir in _iter_sample_dirs(step_dir):
            postprocess_one_sample(
                sample_dir,
                export_mod=export_mod,
                post_mod=post_mod,
                cases_dir_fp=args.cases_dir_fp,
                annotate=not args.no_annotate,
            )


if __name__ == "__main__":
    main()
