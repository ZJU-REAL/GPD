#!/usr/bin/env python3
"""Segment by sentence-ending punctuation (. ? !) and write per-sentence average kl/entropy at the last token of each sentence."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_TOKEN_LINE = re.compile(
    r"^(\s+)(\S+)\s+kl_ao=([+-]?\d+\.\d+)\s+kl_fp=([+-]?\d+\.\d+)(.*)$"
)
_TOKEN_LINE_TRAIN = re.compile(
    r"^(\s+)(\S+)\s+(?:logp_S=([+-]?\d+\.\d+)\s+logp_T=([+-]?\d+\.\d+)\s+)?"
    r"kl_B=([+-]?\d+\.\d+)(?:\s+kl_fp=([+-]?\d+\.\d+))?\s+H_stu=\s*([+-]?\d+\.\d+)\s+H_tea=\s*([+-]?\d+\.\d+)(.*)$"
)
_SEP = "=" * 72
_SENT_TAG = re.compile(r"(?:\s+# \[sent-avg\][^\n]*)+")


def _is_sentence_end(tok: str) -> bool:
    t = tok.strip().replace("\u21b5", "")
    return t in (".", "?", "!")


def _pick_col(fieldnames: list[str] | None, *candidates: str) -> str | None:
    if not fieldnames:
        return None
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        kl_ao_col = _pick_col(fn, "kl_answer_only_B_minus_A", "kl_ref_minus_student_B_minus_A")
        kl_fp_col = _pick_col(fn, "kl_3d_full_priv_B_minus_A", "kl_ref_minus_student_B_minus_A_fp")
        h_stu_col = _pick_col(fn, "entropy_student_ctx_A")
        h_ao_col = _pick_col(fn, "entropy_teacher_answer_only_B", "entropy_teacher_ctx_B")
        h_fp_col = _pick_col(fn, "entropy_teacher_3d_full_priv_B")
        if not kl_ao_col or not h_stu_col or not h_ao_col:
            raise KeyError(f"unsupported CSV columns in {csv_path}: {fn}")
        rows = []
        for r in reader:
            kl_ao = float(r[kl_ao_col])
            kl_fp = float(r[kl_fp_col]) if kl_fp_col else kl_ao
            h_ao = float(r[h_ao_col])
            h_fp = float(r[h_fp_col]) if h_fp_col else h_ao
            rows.append(
                {
                    "step": int(r["step"]),
                    "tok": (r.get("token_decoded") or "").strip(),
                    "kl_ao": kl_ao,
                    "kl_fp": kl_fp,
                    "h_stu": float(r[h_stu_col]),
                    "h_ao": h_ao,
                    "h_fp": h_fp,
                    "train_single": kl_fp_col is None and kl_ao_col == "kl_ref_minus_student_B_minus_A",
                }
            )
        return rows


def _sentence_end_steps(rows: list[dict], *, start_after_think: bool = True) -> list[int]:
    """Return the step of the last token in each sentence (including the punctuation token)."""
    begin = 0
    if start_after_think:
        for i, r in enumerate(rows):
            t = r["tok"].replace("\u21b5", "")
            if t == ">":
                begin = i + 1
                break
    ends: list[int] = []
    for i in range(begin, len(rows)):
        if _is_sentence_end(rows[i]["tok"]):
            ends.append(rows[i]["step"])
    return ends


def _sentence_ranges(rows: list[dict], end_steps: list[int]) -> list[tuple[int, int, int]]:
    """(sent_id, start_step, end_step) inclusive end."""
    step_to_idx = {r["step"]: i for i, r in enumerate(rows)}
    begin_idx = 0
    for r in rows:
        if r["tok"].replace("\u21b5", "") == ">":
            begin_idx = step_to_idx[r["step"]] + 1
            break
    out: list[tuple[int, int, int]] = []
    cur = begin_idx
    for sid, es in enumerate(end_steps, 1):
        ei = step_to_idx[es]
        out.append((sid, rows[cur]["step"], es))
        cur = ei + 1
    return out


def _avg_note(chunk: list[dict], sid: int) -> str:
    n = len(chunk)
    kl_ao = sum(x["kl_ao"] for x in chunk) / n
    kl_fp = sum(x["kl_fp"] for x in chunk) / n
    h_stu = sum(x["h_stu"] for x in chunk) / n
    h_ao = sum(x["h_ao"] for x in chunk) / n
    h_fp = sum(x["h_fp"] for x in chunk) / n
    if chunk and chunk[0].get("train_single"):
        return (
            f"  # [sent-avg S{sid}] kl_B={kl_ao:+.4f} "
            f"H_stu={h_stu:.4f} H_tea={h_ao:.4f} (n={n})"
        )
    return (
        f"  # [sent-avg S{sid}] kl_ao={kl_ao:+.4f} kl_fp={kl_fp:+.4f} "
        f"H_stu={h_stu:.4f} H_tea_ao={h_ao:.4f} H_tea_fp={h_fp:.4f} (n={n})"
    )


def build_end_step_notes(csv_path: Path) -> dict[int, str]:
    rows = _load_rows(csv_path)
    ends = _sentence_end_steps(rows)
    step_map = {r["step"]: r for r in rows}
    notes: dict[int, str] = {}
    for sid, s0, e0 in _sentence_ranges(rows, ends):
        idx0 = next(i for i, r in enumerate(rows) if r["step"] == s0)
        idx1 = next(i for i, r in enumerate(rows) if r["step"] == e0)
        chunk = rows[idx0 : idx1 + 1]
        notes[e0] = _avg_note(chunk, sid)
    return notes


def _split_body_tail(text: str) -> tuple[str, str]:
    if _SEP in text:
        body, tail = text.split(_SEP, 1)
        return body.rstrip(), _SEP + tail
    if "## Key token analysis" in text:
        idx = text.index("## Key token analysis")
        return text[:idx].rstrip(), text[idx:]
    return text.rstrip(), ""


def apply_to_file(
    target: Path,
    end_notes: dict[int, str],
    csv_path: Path | None = None,
) -> int:
    """Align token lines by CSV row order; append one [sent-avg] at each sentence-end step (stripping existing ones first)."""
    csv_steps: list[int] | None = None
    if csv_path is not None and csv_path.is_file():
        csv_steps = [r["step"] for r in _load_rows(csv_path)]

    text = target.read_text(encoding="utf-8")
    body, tail = _split_body_tail(text)
    out: list[str] = []
    token_idx = 0
    n = 0

    for line in body.splitlines():
        line = _SENT_TAG.sub("", line)
        m = _TOKEN_LINE.match(line)
        if not m:
            m = _TOKEN_LINE_TRAIN.match(line)
        if m:
            csv_step = (
                csv_steps[token_idx]
                if csv_steps is not None and token_idx < len(csv_steps)
                else token_idx
            )
            if csv_step in end_notes:
                line = line.rstrip() + end_notes[csv_step]
                n += 1
            token_idx += 1
        elif re.match(r"^\s+\S+\s+(?:logp_S=[+-]?\d+\.\d+\s+logp_T=[+-]?\d+\.\d+\s+)?kl_(ao|B)=", line):
            token_idx += 1
        out.append(line)

    new_text = "\n".join(out) + ("\n" if out else "")
    if tail:
        new_text += "\n\n" + tail.lstrip("\n")
    target.write_text(new_text, encoding="utf-8")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate per-sentence average metrics at sentence-end tokens.")
    ap.add_argument("--row-dir", type=Path, required=True)
    ap.add_argument("--file", type=Path, action="append", default=None, help="May be specified multiple times; defaults to merged+paste")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    row_dir = args.row_dir.resolve()
    csv_p = args.csv or (row_dir / "priv_kl_plots" / f"{row_dir.name}_per_token.csv")
    if not csv_p.is_file():
        raise FileNotFoundError(csv_p)

    notes = build_end_step_notes(csv_p)
    files = args.file or [
        row_dir / f"{row_dir.name}_kl_entropy_merged.txt",
        row_dir / f"{row_dir.name}_kl_entropy_paste_notes.txt",
    ]
    for f in files:
        f = f.resolve()
        if not f.is_file():
            print(f"SKIP missing {f}")
            continue
        n = apply_to_file(f, notes, csv_p)
        print(f"OK {row_dir.name}: {n} sentence marks -> {f}")


if __name__ == "__main__":
    main()
