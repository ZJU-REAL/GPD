#!/usr/bin/env python3
"""Batch post-process vsi_logits: merge ao+fp metric CSV, merged/paste text, sentence averages, and auto inline notes."""

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

_export_path = Path(__file__).resolve().parent / "export_vsi_logits_per_token_metrics.py"
_spec = importlib.util.spec_from_file_location("export_vsi", _export_path)
_export = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_export)
_fmt_tok = _export._fmt_tok
_load_npz_metrics = _export._load_npz_metrics

_plot_path = Path(__file__).resolve().parent / "plot_vsi_logits_priv_kl_per_token.py"
_pspec = importlib.util.spec_from_file_location("plot_vsi", _plot_path)
_plot = importlib.util.module_from_spec(_pspec)
assert _pspec.loader is not None
_pspec.loader.exec_module(_plot)
_token_axis_labels = _plot._token_axis_labels


def _load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def export_full_csv(row_ao: Path, row_fp: Path) -> Path:
    """Write priv_kl_plots/row_*_per_token.csv (including 3D kl and three entropy columns)."""
    z_ao = row_ao / "logits_summary.npz"
    z_fp = row_fp / "logits_summary.npz"
    if not z_ao.is_file() or not z_fp.is_file():
        raise FileNotFoundError(f"missing npz: {z_ao} or {z_fp}")

    m = _load_npz_metrics(z_ao)
    m_fp = _load_npz_metrics(z_fp)
    n = int(m["ids"].shape[0])
    if m_fp["kl"].shape[0] != n:
        raise ValueError(f"length mismatch {row_ao.name}: ao={n} fp={m_fp['kl'].shape[0]}")

    summary = row_ao / "summary.json"
    model_path = None
    if summary.is_file():
        model_path = json.loads(summary.read_text(encoding="utf-8")).get("model_path")
    tok = _load_tokenizer(model_path)
    if tok is not None:
        _, human, piece = _token_axis_labels(tok, m["ids"], "decoded", max_chars=32)
    else:
        human = [str(int(i)) for i in m["ids"]]
        piece = human[:]

    la = m["la"]
    lb_ao = m["lb"]
    lb_fp = m_fp["lb"]
    kl_ao = m["kl"]
    kl_fp = m_fp["kl"]
    kld = m["kld"]
    ent_a = m.get("ent_a")
    ent_ao = m.get("ent_b")
    ent_fp = m_fp.get("ent_b")

    out_dir = row_ao / "priv_kl_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{row_ao.name}_per_token.csv"
    header = [
        "step",
        "token_id",
        "token_decoded",
        "token_piece",
        "logp_student_ctx_A",
        "logp_teacher_answer_only_B",
        "logp_teacher_3d_full_priv_B",
        "kl_answer_only_B_minus_A",
        "kl_3d_full_priv_B_minus_A",
        "low_var_kld_proxy_answer_only",
        "entropy_student_ctx_A",
        "entropy_teacher_answer_only_B",
        "entropy_teacher_3d_full_priv_B",
        "entropy_student_minus_answer_only_teacher",
        "entropy_student_minus_3d_teacher",
    ]
    rows_out: list[list] = []
    for i in range(n):
        ea = float(ent_a[i]) if ent_a is not None else 0.0
        eb = float(ent_ao[i]) if ent_ao is not None else 0.0
        ef = float(ent_fp[i]) if ent_fp is not None else 0.0
        rows_out.append(
            [
                i,
                int(m["ids"][i]),
                human[i],
                piece[i],
                f"{la[i]:.6f}",
                f"{lb_ao[i]:.6f}",
                f"{lb_fp[i]:.6f}",
                f"{kl_ao[i]:.6f}",
                f"{kl_fp[i]:.6f}",
                f"{kld[i]:.6f}",
                f"{ea:.6f}",
                f"{eb:.6f}",
                f"{ef:.6f}",
                f"{(ea - eb):.6f}",
                f"{(ea - ef):.6f}",
            ]
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    # Row-directory copy
    copy_path = row_ao / f"{row_ao.name}_per_token_with_entropy.csv"
    copy_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return csv_path


def _read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_col(fieldnames: list[str] | None, *candidates: str) -> str | None:
    if not fieldnames:
        return None
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def _row_float(row: dict, col: str | None, default: float = 0.0) -> float:
    if not col:
        return default
    val = row.get(col)
    if val is None or val == "":
        return default
    return float(val)


def _csv_format_kind(fieldnames: list[str] | None) -> str:
    if _pick_col(fieldnames, "kl_answer_only_B_minus_A"):
        return "dual_teacher"
    if _pick_col(fieldnames, "kl_ref_minus_student_B_minus_A"):
        return "train_single"
    return "unknown"


def _train_token_metric_parts(r: dict, fieldnames: list[str]) -> list[str]:
    kl_col = _pick_col(fieldnames, "kl_ref_minus_student_B_minus_A")
    kl_fp_col = _pick_col(fieldnames, "kl_ref_minus_student_B_minus_A_fp")
    logp_s_col = _pick_col(fieldnames, "logp_student_ctx_A")
    logp_t_col = _pick_col(fieldnames, "logp_teacher_ctx_B")
    h_stu_col = _pick_col(fieldnames, "entropy_student_ctx_A")
    h_tea_col = _pick_col(fieldnames, "entropy_teacher_ctx_B")
    parts: list[str] = []
    if logp_s_col:
        parts.append(f"logp_S={_row_float(r, logp_s_col):+.4f}")
    if logp_t_col:
        parts.append(f"logp_T={_row_float(r, logp_t_col):+.4f}")
    parts.append(f"kl_B={_row_float(r, kl_col):+.4f}")
    if kl_fp_col:
        parts.append(f"kl_fp={_row_float(r, kl_fp_col):+.4f}")
    parts.append(f"H_stu={_row_float(r, h_stu_col):.4f}")
    parts.append(f"H_tea={_row_float(r, h_tea_col):.4f}")
    return parts


def write_merged_txt(row_dir: Path, csv_path: Path, *, preview: str = "") -> Path:
    rows = _read_csv_rows(csv_path)
    fieldnames = list(rows[0].keys()) if rows else []
    kind = _csv_format_kind(fieldnames)

    if kind == "dual_teacher":
        lines = [
            f"{row_dir.name} | merged entropy + kl (student rollout fixed)\n",
            "H_stu = entropy_student_ctx_A\n",
            "H_tea_ao = entropy_teacher_answer_only_B (<reference_answer> only)\n",
            "H_tea_fp = entropy_teacher_3d_full_priv_B (scene_context + reference_answer)\n",
            "kl_ao / kl_fp = teacher-student logp (B-A) per setting\n\n",
        ]
        if preview:
            lines.insert(1, preview + "\n")
        for r in rows:
            tok = _fmt_tok((r.get("token_decoded") or "").strip())
            parts = [
                f"  {tok:16s}",
                f"kl_ao={_row_float(r, 'kl_answer_only_B_minus_A'):+7.4f}",
                f"kl_fp={_row_float(r, 'kl_3d_full_priv_B_minus_A'):+7.4f}",
                f"H_stu={_row_float(r, 'entropy_student_ctx_A'):7.4f}",
                f"H_tea_ao={_row_float(r, 'entropy_teacher_answer_only_B'):7.4f}",
                f"H_tea_fp={_row_float(r, 'entropy_teacher_3d_full_priv_B'):7.4f}",
            ]
            lines.append("  ".join(parts) + "\n")
    elif kind == "train_single":
        lines = [
            f"{row_dir.name} | merged entropy + kl (student rollout fixed)\n",
            "logp_S = log pi_S (student context A, selected token)\n",
            "logp_T = log pi_T (teacher context B, same rollout token)\n",
            "H_stu = entropy_student_ctx_A | H_tea = entropy_teacher_ctx_B\n",
            "kl_B = logp_T - logp_S = ref_logp - student_logp\n\n",
        ]
        if preview:
            lines.insert(1, preview + "\n")
        for r in rows:
            tok = _fmt_tok((r.get("token_decoded") or "").strip())
            parts = [f"  {tok:16s}"] + _train_token_metric_parts(r, fieldnames)
            lines.append("  ".join(parts) + "\n")
    else:
        raise KeyError(f"unsupported CSV columns in {csv_path}: {fieldnames}")

    out = row_dir / f"{row_dir.name}_kl_entropy_merged.txt"
    out.write_text("".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def write_paste_notes(row_dir: Path, csv_path: Path, summary_path: Path) -> Path:
    rows = _read_csv_rows(csv_path)
    fieldnames = list(rows[0].keys()) if rows else []
    kind = _csv_format_kind(fieldnames)
    meta = ""
    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        prev = (s.get("q_text_preview") or "")[:80]
        meta = f"variant={s.get('variant', '?')} | {prev}...\n"

    if kind == "dual_teacher":
        lines = [
            f"{row_dir.name} per-token metrics (student rollout)\n",
            meta,
            "kl_ao = B-A on answer_only cases-dir\n",
            "kl_fp = B-A from full_priv (3D+answer teacher)\n",
            "H_stu = entropy_student_ctx_A | H_tea = entropy_teacher_answer_only_B\n\n",
        ]
        for r in rows:
            tok = _fmt_tok((r.get("token_decoded") or "").strip())
            parts = [
                f"  {tok:16s}",
                f"kl_ao={_row_float(r, 'kl_answer_only_B_minus_A'):+7.4f}",
                f"kl_fp={_row_float(r, 'kl_3d_full_priv_B_minus_A'):+7.4f}",
                f"H_stu={_row_float(r, 'entropy_student_ctx_A'):7.4f}",
                f"H_tea={_row_float(r, 'entropy_teacher_answer_only_B'):7.4f}",
            ]
            lines.append("  ".join(parts) + "\n")
    elif kind == "train_single":
        lines = [
            f"{row_dir.name} per-token metrics (student rollout)\n",
            meta,
            "logp_S = log pi_S | logp_T = log pi_T (log prob of selected token)\n",
            "kl_B = logp_T - logp_S | H_stu / H_tea = full-vocabulary Shannon entropy\n\n",
        ]
        for r in rows:
            tok = _fmt_tok((r.get("token_decoded") or "").strip())
            parts = [f"  {tok:16s}"] + _train_token_metric_parts(r, fieldnames)
            lines.append("  ".join(parts) + "\n")
    else:
        raise KeyError(f"unsupported CSV columns in {csv_path}: {fieldnames}")

    out = row_dir / f"{row_dir.name}_kl_entropy_paste_notes.txt"
    out.write_text("".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def postprocess_row(row_ao: Path, row_fp: Path, *, annotate: bool = True) -> None:
    csv_p = export_full_csv(row_ao, row_fp)
    write_merged_txt(row_ao, csv_p)
    write_paste_notes(row_ao, csv_p, row_ao / "summary.json")

    if not annotate:
        return

    merged = row_ao / f"{row_ao.name}_kl_entropy_merged.txt"
    paste = row_ao / f"{row_ao.name}_kl_entropy_paste_notes.txt"

    _kl_path = Path(__file__).resolve().parent / "annotate_kl_entropy_paste_notes.py"
    spec = importlib.util.spec_from_file_location("annotate_kl", _kl_path)
    _kl = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_kl)
    _kl.annotate_metrics_file(merged, csv_p)
    _kl.annotate_metrics_file(paste, csv_p)

    _sent_path = Path(__file__).resolve().parent / "annotate_sentence_averages.py"
    spec2 = importlib.util.spec_from_file_location("annotate_sent", _sent_path)
    _sent = importlib.util.module_from_spec(spec2)
    assert spec2.loader is not None
    spec2.loader.exec_module(_sent)
    sent_notes = _sent.build_end_step_notes(csv_p)
    _sent.apply_to_file(merged, sent_notes, csv_p)
    _sent.apply_to_file(paste, sent_notes, csv_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir-ao", type=Path, required=True)
    ap.add_argument("--cases-dir-fp", type=Path, required=True)
    ap.add_argument("--row", type=str, default=None, help="Process a single row_00020 only; defaults to all row_*")
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    ao_root = args.cases_dir_ao.resolve()
    fp_root = args.cases_dir_fp.resolve()
    if args.row:
        names = [args.row]
    else:
        names = sorted(p.name for p in ao_root.iterdir() if p.is_dir() and re.match(r"row_\d{5}", p.name))

    for name in names:
        ra, rf = ao_root / name, fp_root / name
        if not (ra / "logits_summary.npz").is_file():
            print(f"SKIP {name}: no ao npz")
            continue
        postprocess_row(ra, rf, annotate=not args.no_annotate)
        print(f"OK postprocess {name}")


if __name__ == "__main__":
    main()
