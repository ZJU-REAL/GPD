#!/usr/bin/env python3
"""
Export per-token CSV and notes (kl + student/teacher entropy) from ``logits_summary.npz``
without re-running the model or re-plotting.

Entropy definitions (consistent with analyze_rollout_student_teacher_logits.py):
  - entropy_student_ctx_A: scenario A, student prompt + student rollout, per-step full-vocabulary Shannon entropy
  - entropy_teacher_ctx_B: scenario B, teacher prompt + same student rollout, per-step full-vocabulary Shannon entropy

Usage::

  # Single row, write to row directory and priv_kl_plots
  python scripts/export_vsi_logits_per_token_metrics.py \\
      --row-dir /path/to/vsi_logits_.../row_00004

  # Optional: merge kl_fp from a second cases-dir (full_priv) into the same row
  python scripts/export_vsi_logits_per_token_metrics.py \\
      --row-dir .../answer_only/row_00004 \\
      --cases-dir-fp .../full_priv

  # Batch over a whole cases-dir
  python scripts/export_vsi_logits_per_token_metrics.py \\
      --cases-dir /path/to/vsi_logits_10cases_answer_only \\
      --cases-dir-fp /path/to/vsi_logits_10cases_full_priv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

_EASYR1 = Path(__file__).resolve().parents[1]
if str(_EASYR1) not in sys.path:
    sys.path.insert(0, str(_EASYR1))

_plot_path = Path(__file__).resolve().parent / "plot_vsi_logits_priv_kl_per_token.py"
_spec = importlib.util.spec_from_file_location("plot_vsi_logits_priv_kl", _plot_path)
_plot_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_plot_mod)
_token_axis_labels = _plot_mod._token_axis_labels
low_var_kld_from_logp_diff = _plot_mod.low_var_kld_from_logp_diff


def _fmt_tok(t: str) -> str:
    if t in (">\u21b5", ">\u010a"):
        return ">"
    if t == ".\u21b5":
        return "."
    if t == "<|im_end|>":
        return "<eos>"
    return t.replace("\u21b5", "")


def _load_npz_metrics(zpath: Path) -> dict[str, np.ndarray]:
    z = np.load(zpath)
    pre_a = "A_student_prompt_student_resp__"
    pre_b = "B_teacher_prompt_student_resp__"
    ids = z[pre_a + "response_token_ids"]
    la = z[pre_a + "logp_selected"].astype(np.float64)
    lb = z[pre_b + "logp_selected"].astype(np.float64)
    if "student_resp_logp_diff_B_minus_A" in z.files:
        diff = z["student_resp_logp_diff_B_minus_A"].astype(np.float64)
    else:
        diff = lb - la
    kl, kld = low_var_kld_from_logp_diff(diff)
    out: dict[str, np.ndarray] = {
        "ids": ids,
        "la": la,
        "lb": lb,
        "kl": kl,
        "kld": kld,
    }
    ea_key = pre_a + "entropy"
    eb_key = pre_b + "entropy"
    if ea_key in z.files:
        out["ent_a"] = z[ea_key].astype(np.float64)
    if eb_key in z.files:
        out["ent_b"] = z[eb_key].astype(np.float64)
    return out


def _load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def export_one_row(
    row_dir: Path,
    *,
    cases_dir_fp: Path | None = None,
    model_path: str | None = None,
    write_csv: bool = True,
    write_paste_notes: bool = True,
    write_row_csv: bool = True,
) -> Path | None:
    zpath = row_dir / "logits_summary.npz"
    if not zpath.is_file():
        raise FileNotFoundError(zpath)

    m = _load_npz_metrics(zpath)
    n = int(m["ids"].shape[0])

    kl_fp = None
    if cases_dir_fp is not None:
        fp_row = cases_dir_fp / row_dir.name
        fp_z = fp_row / "logits_summary.npz"
        if fp_z.is_file():
            m_fp = _load_npz_metrics(fp_z)
            if m_fp["kl"].shape[0] == n:
                kl_fp = m_fp["kl"]

    summary_path = row_dir / "summary.json"
    model_path_eff = model_path
    if summary_path.is_file() and not model_path_eff:
        try:
            model_path_eff = json.loads(summary_path.read_text(encoding="utf-8")).get("model_path")
        except json.JSONDecodeError:
            pass

    tok = _load_tokenizer(model_path_eff)
    if tok is not None:
        tick, human, piece = _token_axis_labels(tok, m["ids"], "decoded", max_chars=32)
    else:
        human = [str(int(i)) for i in m["ids"]]
        piece = human[:]

    ent_a = m.get("ent_a")
    ent_b = m.get("ent_b")

    csv_name = f"{row_dir.name}_per_token.csv"
    csv_paths: list[Path] = []
    if write_csv:
        for parent in (row_dir / "priv_kl_plots", row_dir):
            if parent == row_dir and not write_row_csv:
                continue
            parent.mkdir(parents=True, exist_ok=True)
            csv_paths.append(parent / csv_name)

    header = [
        "step",
        "token_id",
        "token_decoded",
        "token_piece",
        "logp_student_ctx_A",
        "logp_teacher_ctx_B",
        "kl_ref_minus_student_B_minus_A",
        "low_var_kld_proxy",
    ]
    if ent_a is not None:
        header.append("entropy_student_ctx_A")
    if ent_b is not None:
        header.append("entropy_teacher_ctx_B")
    if ent_a is not None and ent_b is not None:
        header.append("entropy_diff_B_minus_A")
    if kl_fp is not None:
        header.append("kl_ref_minus_student_B_minus_A_fp")

    rows_csv: list[list] = []
    for i in range(n):
        row_out: list = [
            i,
            int(m["ids"][i]),
            human[i],
            piece[i],
            f"{m['la'][i]:.6f}",
            f"{m['lb'][i]:.6f}",
            f"{m['kl'][i]:.6f}",
            f"{m['kld'][i]:.6f}",
        ]
        if ent_a is not None:
            row_out.append(f"{ent_a[i]:.6f}")
        if ent_b is not None:
            row_out.append(f"{ent_b[i]:.6f}")
        if ent_a is not None and ent_b is not None:
            row_out.append(f"{(ent_b[i] - ent_a[i]):.6f}")
        if kl_fp is not None:
            row_out.append(f"{kl_fp[i]:.6f}")
        rows_csv.append(row_out)

    for cp in csv_paths:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_csv)

    paste_path = None
    if write_paste_notes:
        meta = ""
        if summary_path.is_file():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            prev = (s.get("q_text_preview") or "")[:80]
            meta = f"variant={s.get('variant', '?')} | {prev}...\n"
        lines = [
            f"{row_dir.name} per-token metrics (student rollout)\n",
            meta,
            "kl_ao = B-A on this cases-dir (teacher priv per run)\n",
        ]
        if kl_fp is not None:
            lines.append("kl_fp = B-A from --cases-dir-fp (3D+answer teacher)\n")
        lines.append(
            "H_stu = entropy_student_ctx_A | H_tea = entropy_teacher_ctx_B (full vocab Shannon)\n\n"
        )

        chunks: list[str] = []
        buf: list[str] = []
        for i in range(n):
            tok = _fmt_tok(human[i] or piece[i])
            parts = [
                f"  {tok:16s}",
                f"kl_ao={m['kl'][i]:+7.4f}",
            ]
            if kl_fp is not None:
                parts.append(f"kl_fp={kl_fp[i]:+7.4f}")
            if ent_a is not None:
                parts.append(f"H_stu={ent_a[i]:7.4f}")
            if ent_b is not None:
                parts.append(f"H_tea={ent_b[i]:7.4f}")
            buf.append("  ".join(parts))
            if tok in (".", "?", "!") or tok in (">", "<eos>"):
                chunks.append("\n".join(buf))
                buf = []
        if buf:
            chunks.append("\n".join(buf))

        paste_path = row_dir / f"{row_dir.name}_kl_entropy_paste_notes.txt"
        paste_path.write_text(
            "".join(lines) + "\n\n".join(chunks) + ("\n" if chunks else ""),
            encoding="utf-8",
        )
        csv_for_ann = (row_dir / "priv_kl_plots" / csv_name) if write_csv else None
        if csv_for_ann and csv_for_ann.is_file() and kl_fp is not None:
            try:
                _ann_path = Path(__file__).resolve().parent / "annotate_kl_entropy_paste_notes.py"
                _spec = importlib.util.spec_from_file_location("annotate_paste", _ann_path)
                _ann = importlib.util.module_from_spec(_spec)
                assert _spec.loader is not None
                _spec.loader.exec_module(_ann)
                _ann.annotate_paste_file(paste_path, csv_for_ann)
                _ann.append_tail_analysis(paste_path, row_dir)
            except Exception as e:
                print(f"WARN {row_dir.name}: paste annotate skipped: {e}", file=sys.stderr)

    return csv_paths[0] if csv_paths else paste_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-token CSV + kl/entropy paste notes from npz.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--row-dir", type=Path, help="Single row_XXXXX directory")
    g.add_argument("--cases-dir", type=Path, help="Batch mode: directory containing multiple row_*")
    ap.add_argument(
        "--cases-dir-fp",
        type=Path,
        default=None,
        help="Optional: full_priv root directory to merge kl_fp column",
    )
    ap.add_argument("--model-path", type=str, default=None, help="Tokenizer for decoding; falls back to summary.json")
    ap.add_argument("--no-paste-notes", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    if args.row_dir:
        row_dirs = [args.row_dir.resolve()]
    else:
        row_dirs = sorted(
            p for p in args.cases_dir.resolve().iterdir() if p.is_dir() and re.match(r"row_\d{5}", p.name)
        )

    for rd in row_dirs:
        p = export_one_row(
            rd,
            cases_dir_fp=args.cases_dir_fp.resolve() if args.cases_dir_fp else None,
            model_path=args.model_path,
            write_csv=not args.no_csv,
            write_paste_notes=not args.no_paste_notes,
        )
        print(f"OK {rd.name} -> {p}")


if __name__ == "__main__":
    main()
