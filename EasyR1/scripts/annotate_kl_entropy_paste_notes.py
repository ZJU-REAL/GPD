#!/usr/bin/env python3
"""Append inline analysis comments (# per token) at the end of token lines in paste_notes / merged files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Per-row inline notes keyed by row name -> step -> annotation (takes priority over auto rules)
INLINE_NOTES: dict[str, dict[int, str]] = {
    "row_00004": {
        8: "3D suppresses opening 'we' (|kl_fp|~1.9)",
        14: "suppresses 'is closest framing' (kl_fp~-1.1)",
        23: "3D strongly suppresses 'mounted', deviating from 'nearest object' task (~-6.2)",
        49: "kl_fp>0: 3D briefly approves 'In frame' opening",
        52: "* |kl_fp| max ~-8.9: rejects frame number 6",
        60: "* suppresses 'directly below' (key spatial error, ~-4.3)",
        66: "* suppresses 'table=nearest object'; answer-only |kl_ao|~0.47, 3D~-4.2",
        70: "chair: kl_fp>0, visible but not rejected",
        97: "plant: kl_fp>0, GT mentioned but not adopted",
        114: "comparison sentence 'table' still negative",
        165: "conclusion 'table', kl_fp still negative",
        179: "answer >C: KL~0, privileged info barely constrains the answer token",
    },
    "row_00014": {
        14: "comma after 'bed' sentence raises warning (kl_fp~-2.4)",
        22: "first 'the' in 'bed' sentence, moderate kl (kl_fp~-2.1)",
        28: "'towel' should not be ranked 2nd (kl_fp~-2.0)",
        35: "* |kl_fp| max ~-8.9: suppresses 'placed on bed' (core misordering)",
        39: "end of 'towel' sentence, misordering closed",
        46: "suppresses 'on the floor' (~-3.2)",
        51: "strongly suppresses 'chair' (~-5.4)",
        86: "conclusion enumerates 'towel' still in second position",
        98: "answer >B: KL~0",
    },
}

_TOKEN_LINE = re.compile(
    r"^(\s+)(\S+)\s+kl_ao=([+-]?\d+\.\d+)\s+kl_fp=([+-]?\d+\.\d+)(.*)$"
)
_TOKEN_LINE_TRAIN = re.compile(
    r"^(\s+)(\S+)\s+(?:logp_S=([+-]?\d+\.\d+)\s+logp_T=([+-]?\d+\.\d+)\s+)?"
    r"kl_B=([+-]?\d+\.\d+)(?:\s+kl_fp=([+-]?\d+\.\d+))?\s+H_stu=\s*([+-]?\d+\.\d+)\s+H_tea=\s*([+-]?\d+\.\d+)(.*)$"
)
_SEP = "=" * 72


def _kl_ao_col(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    for c in ("kl_answer_only_B_minus_A", "kl_ref_minus_student_B_minus_A"):
        if c in fieldnames:
            return c
    return None


def _kl_fp_col(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    for c in ("kl_3d_full_priv_B_minus_A", "kl_ref_minus_student_B_minus_A_fp"):
        if c in fieldnames:
            return c
    return None


def _load_csv_steps(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kl_ao_col = _kl_ao_col(reader.fieldnames)
        kl_fp_col = _kl_fp_col(reader.fieldnames)
        rows = []
        for r in reader:
            step = int(r["step"])
            if kl_fp_col:
                kl_fp = float(r[kl_fp_col])
            elif kl_ao_col:
                kl_fp = float(r[kl_ao_col])
            else:
                kl_fp = 0.0
            tok = (r.get("token_decoded") or r.get("token_piece") or "").strip()
            rows.append({"step": step, "tok": tok, "kl_fp": kl_fp})
        return rows


def _auto_note(step: int, tok: str, kl_fp: float, row: str) -> str | None:
    if tok in (">C", ">B", ">D", ">A") or (tok.startswith(">") and len(tok) == 2):
        if abs(kl_fp) < 0.05:
            return "answer token: KL~0"
    if abs(kl_fp) < 1.2:
        return None
    sign = "suppresses" if kl_fp < 0 else "3D approves"
    return f"{sign} {tok!r} (kl_fp={kl_fp:+.2f})"


def _note_for_step(row: str, step: int, tok: str, kl_fp: float) -> str | None:
    custom = INLINE_NOTES.get(row, {}).get(step)
    if custom:
        return custom
    return _auto_note(step, tok, kl_fp, row)


def _split_body_tail(text: str) -> tuple[str, str]:
    if _SEP in text:
        body, tail = text.split(_SEP, 1)
        return body.rstrip(), _SEP + tail
    if "## Key token analysis" in text:
        idx = text.index("## Key token analysis")
        return text[:idx].rstrip(), text[idx:]
    return text.rstrip(), ""


def annotate_metrics_file(
    target_path: Path,
    csv_path: Path,
    *,
    row_name: str | None = None,
    min_auto: float = 1.2,
) -> int:
    """Annotate only the token lines in the body; leave the ===== analysis section unchanged."""
    rows = _load_csv_steps(csv_path)
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    row = row_name or target_path.parent.name
    text = target_path.read_text(encoding="utf-8")
    body, tail = _split_body_tail(text)

    out_lines: list[str] = []
    step = 0
    n_annot = 0

    for line in body.splitlines():
        if line.strip().startswith("## "):
            out_lines.append(line)
            continue

        line = re.sub(r"\s+# .*$", "", line)
        m = _TOKEN_LINE.match(line)
        if not m:
            m = _TOKEN_LINE_TRAIN.match(line)
        if not m and step < len(rows):
            m2 = re.match(
                r"^(\s+)(\S+)\s+(?:logp_S=[+-]?\d+\.\d+\s+logp_T=[+-]?\d+\.\d+\s+)?kl_(ao|B)=([+-]?\d+\.\d+)(.*)$",
                line,
            )
            if m2:
                prefix, tok, kl_kind, kl_val, rest = (
                    m2.group(1),
                    m2.group(2),
                    m2.group(3),
                    m2.group(4),
                    m2.group(5),
                )
                note = _note_for_step(row, step, tok, rows[step]["kl_fp"])
                if note:
                    line = f"{prefix}{tok:16s} kl_{kl_kind}={kl_val}{rest}  # {note}"
                    n_annot += 1
                step += 1
                out_lines.append(line)
                continue

        if m and m.re is _TOKEN_LINE_TRAIN:
            tok = m.group(2).strip()
            kl_fp_opt = m.group(5)
            kl_b = m.group(4)
            kl_fp = rows[step]["kl_fp"] if step < len(rows) else float(kl_fp_opt or kl_b)
            note = _note_for_step(row, step, tok, kl_fp)
            if note is None and abs(kl_fp) >= min_auto:
                note = _auto_note(step, tok, kl_fp, row)
            if note:
                line = line.rstrip() + f"  # {note}"
                n_annot += 1
            step += 1
        elif m:
            prefix, tok, kl_ao, kl_fp_s, tail_rest = m.groups()
            kl_fp = rows[step]["kl_fp"] if step < len(rows) else float(kl_fp_s)
            note = _note_for_step(row, step, tok.strip(), kl_fp)
            if note is None and abs(kl_fp) >= min_auto:
                note = _auto_note(step, tok.strip(), kl_fp, row)
            if note:
                line = f"{prefix}{tok:16s} kl_ao={kl_ao}  kl_fp={kl_fp_s}{tail_rest}  # {note}"
                n_annot += 1
            step += 1
        out_lines.append(line)

    new_text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if tail:
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += "\n" + tail.lstrip("\n")
    target_path.write_text(new_text, encoding="utf-8")
    return n_annot


def annotate_paste_file(paste_path: Path, csv_path: Path, *, min_auto: float = 1.2) -> int:
    return annotate_metrics_file(paste_path, csv_path, min_auto=min_auto)


def append_tail_analysis(paste_path: Path, row_dir: Path) -> bool:
    """Append the key-token analysis section from merged to the end of paste_notes."""
    merged = row_dir / f"{row_dir.name}_kl_entropy_merged.txt"
    if not merged.is_file():
        return False
    parts = merged.read_text(encoding="utf-8").split(_SEP, 1)
    if len(parts) < 2 or "## Key token analysis" not in parts[1]:
        return False

    text = paste_path.read_text(encoding="utf-8")
    body, _ = _split_body_tail(text)
    paste_path.write_text(body + "\n\n" + _SEP + parts[1].lstrip("\n"), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate kl/entropy notes inline (# per token).")
    ap.add_argument("--row-dir", type=Path, required=True)
    ap.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Target file; defaults to paste_notes. Pass a merged path to annotate merged only.",
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--no-tail", action="store_true", help="paste mode: do not append the tail analysis section")
    args = ap.parse_args()

    row_dir = args.row_dir.resolve()
    target = args.file or (row_dir / f"{row_dir.name}_kl_entropy_paste_notes.txt")
    csv_p = args.csv or (row_dir / "priv_kl_plots" / f"{row_dir.name}_per_token.csv")
    if not target.is_file():
        raise FileNotFoundError(target)
    if not csv_p.is_file():
        raise FileNotFoundError(csv_p)

    n = annotate_metrics_file(target, csv_p)
    tail = False
    if not args.no_tail and target.name.endswith("_kl_entropy_paste_notes.txt"):
        tail = append_tail_analysis(target, row_dir)

    kind = "merged" if "merged" in target.name else "paste"
    msg = f"OK {row_dir.name} ({kind}): {n} inline notes"
    if tail:
        msg += ", tail analysis appended"
    print(f"{msg} -> {target}")


if __name__ == "__main__":
    main()
