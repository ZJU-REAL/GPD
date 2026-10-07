#!/usr/bin/env python3
"""
From directories produced by ``analyze_rollout_student_teacher_logits.py``
(containing ``row_*/logits_summary.npz``), plot per-sample: each token's text
fragment from the student rollout + a per-position "privileged KL proxy"
consistent with the training ``low_var_kl``.

Alignment (consistent with ``verl/trainer/core_algos.py::compute_kl(..., kl_penalty="low_var_kl")``)::
  kl_i = clamp( ref_logp_i - logp_i, -20, 20 )
  where in the offline dump: logp_i ~ A (student prompt + student resp),
  ref_logp_i ~ B (teacher prompt + same student resp).
  kld_i = clamp( exp(kl_i) - kl_i - 1, -10, 10 )

Output:
  - By default, **every token gets an x-tick** (split by ``--tokens-per-panel`` into stacked
    rows: kl on top, kld below; too many rows are split into multiple PNGs)
  - Optional ``--write-csv``: one row per token, suitable for Excel filtering
  - ``--plot-lang auto|zh|en``: language for titles/axis labels; ``auto`` tries system
    Noto/WenQuanYi CJK fonts, falls back to English (avoids DejaVu missing-glyph warnings)
  - ``--x-labels step_human|step_only|piece``: x-axis default ``step_human``
    (step number + ``tokenizer.decode([id])`` fragment); ``piece`` is raw BPE subword for debugging
  - ``--x-tick-rotation``: rotation angle for per-token x-ticks; **default 0 (horizontal)**.
    Try 35 or 90 if labels overlap
  - ``--tokens-per-panel``: number of tokens per horizontal panel block (default 36); the full
    sequence is assembled into a "multi-row" figure
  - ``--max-figure-height-inches``: max height (inches) per PNG; if exceeded, output is split into
    ``*_part00.png``, ``*_part01.png``, ...
  - ``--one-panel-per-file``: each ``--tokens-per-panel`` block in its own PNG (kl top + kld
    bottom), giving taller bars; with multiple blocks filenames are ``*_panelNN_stepA_B.png``
  - ``--panel-figure-height-inches``: only with ``--one-panel-per-file``, total figure height
    (inches, default 32)
  - ``--kl-ylim-pct``: kl red/green plot uses **percentile of |kl|** for the symmetric y-axis
    half-range (default 50 = median), not max|kl|, so outlier tokens do not squash the rest
  - ``--kl-min-half``: minimum y-axis half-range for kl (default 0.008); keeps visible bar height
    even when all |kl| are tiny
  - ``--kl-ylim-max-half``: optional upper bound on kl half-range (e.g. 0.15), further compresses
    axis and raises short bars
  - ``--kl-kld-height-ratio`` / ``--kld-panel-height-ratio``: relative vertical heights of the
    top (kl) and bottom (kld) sub-plots per block (defaults 2.2 / 1.6, kld is taller)
  - ``--kld-ylim-pct`` / ``--no-kld-tight-ylim``: percentile tight-axis for the kld sub-plot
  - kl bars have a thin dark edge by default, making very short bars easier to distinguish
  - ``--dpi`` / ``--high-quality``: increase PNG pixel density; font sizes scale proportionally
    to avoid blurry text at high DPI
  - Performance: ``--reuse-csv-labels``, ``--skip-existing``; ``--plot-lang en`` skips the
    full system font scan

Dependencies: numpy, matplotlib, transformers; no GPU required.
If the server lacks CJK fonts: ``sudo apt install fonts-noto-cjk`` or use ``--plot-lang en``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Plot labels: English requires no CJK font; Chinese needs system Noto/WenQuanYi etc.
_PLOT_I18N = {
    "en": {
        "title_extra": "Green: higher log p under teacher ctx; Red: lower",
        "ylabel_kl": "kl = B \u2212 A\n\u2248 ref_lp \u2212 log_lp",
        "ylabel_kld": "low_var_kld\n\u2248 exp(kl)\u2212kl\u22121 (same form as train low_var_kl)",
        "xlabel": "Position (each bar = one token). Ticks: step + decoded char/slice (CSV: token_decoded)",
    },
    "zh": {
        "title_extra": "Green: teacher context assigns higher log p to this token; Red: lower",
        "ylabel_kl": "kl = B \u2212 A\n\u2248 ref_lp \u2212 log_lp",
        "ylabel_kld": "low_var_kld\n\u2248 exp(kl)\u2212kl\u22121 (same form as train low_var_kl)",
        "xlabel": "Position (each bar = one token). Ticks: step + decoded char/slice (full col in CSV: token_decoded)",
    },
}


def _try_enable_matplotlib_cjk_font() -> str | None:
    """If a sans font containing CJK glyphs is found, set it as default and return its name; else return None.
    Scans the already-registered font list to avoid a full-disk findSystemFonts scan."""
    import matplotlib.font_manager as fm

    keywords = ("CJK", "Noto Sans CJK", "WenQuanYi", "Source Han", "YaHei", "SimHei", "Heiti", "Ming")
    for font in fm.fontManager.ttflist:
        fn = font.name
        if any(k in fn for k in keywords):
            return fn
    return None


def _matplotlib_init(plot_lang: str) -> dict[str, str]:
    """Set up Agg backend and font; return the i18n label dict for this run (keys match _PLOT_I18N)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lang = plot_lang.strip().lower()
    if lang not in ("auto", "en", "zh"):
        lang = "auto"

    use_zh = lang == "zh" or lang == "auto"
    font_name: str | None = None
    if use_zh:
        font_name = _try_enable_matplotlib_cjk_font()
        if font_name is not None:
            plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        elif lang == "zh":
            import sys

            print(
                "Warning: no CJK font found; Chinese labels may appear as boxes. "
                "Install fonts-noto-cjk or use --plot-lang en",
                file=sys.stderr,
            )

    if font_name is None and (lang == "en" or lang == "auto"):
        plt.rcParams["font.family"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return dict(_PLOT_I18N["en"])

    if font_name is not None:
        return dict(_PLOT_I18N["zh"])
    # auto mode and no CJK found: fall back to English to avoid tofu / UserWarning
    return dict(_PLOT_I18N["en"])


def _dpi_scale(dpi: int, ref: int = 100) -> float:
    return max(0.5, float(dpi) / float(ref))


def _scaled_pt(base: float, dpi: int, ref: int = 100, minimum: float = 1.0) -> float:
    return max(minimum, base * _dpi_scale(dpi, ref))


def _figure_dims_for_tokens(seg_n: int, x_tick_rotation: float, dpi: int = 150) -> tuple[float, float, float]:
    """Figure width, tick font size, and bottom padding factor for a horizontal panel of seg_n tokens."""
    seg_n = max(1, int(seg_n))
    if abs(x_tick_rotation) < 22:
        w_in = min(72.0, max(18.0, seg_n * 0.72))
        tick_fs = _scaled_pt(6.0 if seg_n <= 28 else 5.0, dpi)
        bottom_extra = min(0.40, 0.11 + 0.0065 * seg_n)
    else:
        w_in = min(56.0, max(14.0, seg_n * 0.38))
        tick_fs = _scaled_pt(5.0, dpi)
        bottom_extra = 0.14 + (0.18 if abs(x_tick_rotation) > 40 else 0.12)
    return w_in, tick_fs, bottom_extra


def _symmetric_kl_ylim_max_abs(seg_kl: np.ndarray) -> tuple[float, float]:
    """kl y-axis: +/-max|kl| (easily dominated by outliers; only used with --no-kl-tight-ylim)."""
    if seg_kl.size == 0:
        return -1.0, 1.0
    half = max(float(np.max(np.abs(seg_kl.astype(np.float64)))), 1e-12)
    return -half, half


def _symmetric_kl_ylim_tight(
    seg_kl: np.ndarray,
    pct: float,
    min_half: float,
    max_half: float | None,
) -> tuple[float, float, float]:
    """kl y-axis: symmetric half-range = percentile(|kl|, pct), clamped to [min_half, max_half]; returns (y0, y1, half)."""
    if seg_kl.size == 0:
        h = max(min_half, 1e-12)
        return -h, h, h
    absv = np.abs(seg_kl.astype(np.float64))
    if absv.size == 1:
        half = float(absv[0])
    else:
        half = float(np.percentile(absv, pct))
    half = max(half, min_half, 1e-12)
    if max_half is not None and max_half > 0:
        half = min(half, max_half)
    return -half, half, half


def _kld_percentile_ylim(seg_kd: np.ndarray, pct_lo: float, pct_hi: float) -> tuple[float, float]:
    """kld sub-plot: tighten y-axis using a percentile interval + padding, preventing extreme values from dominating."""
    x = np.asarray(seg_kd, dtype=np.float64).ravel()
    if x.size == 0:
        return -1.0, 1.0
    if x.size == 1:
        v = float(x[0])
        return v - 0.05, v + 0.05
    lo = float(np.percentile(x, pct_lo))
    hi = float(np.percentile(x, pct_hi))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span < 1e-12:
        v = (hi + lo) / 2.0
        return v - 0.05, v + 0.05
    pad = 0.08 * span
    return lo - pad, hi + pad


def _png_ihdr_size(path: Path) -> tuple[int, int]:
    """Read width and height from the PNG IHDR chunk; raise on invalid data."""
    import struct

    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG or file too short: {path}")
    return struct.unpack(">II", data[16:24])


def _validate_png(path: Path, max_side_px: int = 16000) -> None:
    w, h = _png_ihdr_size(path)
    if w <= 0 or h <= 0 or w > max_side_px or h > max_side_px:
        raise ValueError(f"Unexpected PNG dimensions {w}x{h} (limit {max_side_px}): {path}")


def _parse_kld_ylim_pct(s: str) -> tuple[float, float]:
    """Parse ``--kld-ylim-pct``, e.g. ``4,96``."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--kld-ylim-pct requires two numbers (low,high percentile), e.g. 4,96, got: {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f"--kld-ylim-pct must satisfy 0<=lo<hi<=100, got {lo}, {hi}")
    return lo, hi


def low_var_kld_from_logp_diff(diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """diff = ref_lp - log_lp (= B - A); returns (kl_clamped, kld_clamped)."""
    kl = np.clip(diff.astype(np.float64), -20.0, 20.0)
    kld = np.exp(kl) - kl - 1.0
    kld = np.clip(kld, -10.0, 10.0)
    return kl, kld


def _format_piece_str(t: str | None, tid: int) -> str:
    if not t:
        return f"<{tid}>"
    return t.replace("\u0121", "\xb7").replace("\u2581", "\xb7").replace("\n", "\u21b5").replace("\r", "")


def _clean_decode_fragment(s: str, fallback: str) -> str:
    s = (s or "").replace("\n", "\u21b5").replace("\r", "").replace("\x00", "")
    return s if s.strip() else fallback


def _labels_from_csv(csv_path: Path, n: int) -> tuple[list[str], list[str]] | None:
    """Read human/piece columns from an existing per_token.csv; length must match n."""
    if not csv_path.is_file():
        return None
    human: list[str] = []
    piece: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = (row.get("token_decoded") or "").strip()
            p = (row.get("token_piece") or "").strip()
            human.append(h if h else p)
            piece.append(p if p else h)
    if len(human) != n or len(piece) != n:
        return None
    if not any(human) and not any(piece):
        return None
    return human, piece


def _token_axis_labels(
    tokenizer,
    ids: np.ndarray,
    mode: str,
    max_chars: int = 8,
    *,
    pre_human: list[str] | None = None,
    pre_piece: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (tick_labels for x-axis, human column, piece column), all of length n."""
    n = int(ids.shape[0])
    mode = (mode or "step_human").strip().lower()
    if mode not in ("step_human", "step_only", "piece"):
        mode = "step_human"
    ids_list = [int(x) for x in ids.astype(np.int64).ravel()]

    if pre_human is not None and pre_piece is not None and len(pre_human) == n and len(pre_piece) == n:
        human = list(pre_human)
        piece = list(pre_piece)
    elif mode == "step_only":
        human = [""] * n
        piece = [""] * n
    else:
        pieces_raw = tokenizer.convert_ids_to_tokens(ids_list)
        piece = [_format_piece_str(t, tid) for t, tid in zip(pieces_raw, ids_list)]
        if mode == "piece":
            human = piece
        else:
            decoded = tokenizer.batch_decode([[tid] for tid in ids_list], skip_special_tokens=False)
            human = [_clean_decode_fragment(s, piece[i]) for i, s in enumerate(decoded)]

    tick: list[str] = []
    for i in range(n):
        h = human[i].replace("\n", "\u21b5")
        if len(h) > max_chars:
            h = h[: max_chars - 1] + "\u2026"
        if mode == "piece":
            p = piece[i]
            if len(p) > max_chars:
                p = p[: max_chars - 1] + "\u2026"
            tick.append(f"{i}\n{p}")
        elif mode == "step_only":
            tick.append(str(i))
        else:
            tick.append(f"{i}\n{h}")
    return tick, human, piece


def _plot_one_row(
    row_dir: Path,
    tokenizer,
    out_png: Path,
    csv_path: Path | None,
    plot_text: dict[str, str],
    x_label_mode: str,
    x_tick_rotation: float,
    tokens_per_panel: int,
    max_figure_height_inches: float,
    label_max_chars: int,
    kl_tight_ylim: bool,
    kl_ylim_pct: float,
    kl_min_half: float,
    kl_ylim_max_half: float | None,
    one_panel_per_file: bool = False,
    panel_figure_height_inches: float = 32.0,
    kl_kld_height_ratio: float = 2.2,
    kld_panel_height_ratio: float = 1.6,
    kld_tight_ylim: bool = True,
    kld_ylim_pct_lo: float = 4.0,
    kld_ylim_pct_hi: float = 96.0,
    title_suffix: str = "",
    dpi: int = 150,
    skip_existing: bool = False,
    reuse_csv_labels: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Plot in blocks, with **one x-tick per token** within each block; returns list of written PNG paths."""
    import matplotlib.pyplot as plt

    zpath = row_dir / "logits_summary.npz"
    if not zpath.is_file():
        raise FileNotFoundError(zpath)

    z = np.load(zpath)
    pre_a = "A_student_prompt_student_resp__"
    pre_b = "B_teacher_prompt_student_resp__"
    ids_key = pre_a + "response_token_ids"
    la_key = pre_a + "logp_selected"
    lb_key = pre_b + "logp_selected"
    ea_key = pre_a + "entropy"
    eb_key = pre_b + "entropy"
    for k in (ids_key, la_key, lb_key):
        if k not in z.files:
            raise KeyError(f"Missing key {k}; ensure the file was generated by analyze_rollout_student_teacher_logits.")

    ids = z[ids_key]
    la = z[la_key].astype(np.float64)
    lb = z[lb_key].astype(np.float64)
    ent_a = z[ea_key].astype(np.float64) if ea_key in z.files else None
    ent_b = z[eb_key].astype(np.float64) if eb_key in z.files else None
    if ent_a is not None and ent_a.shape[0] != ids.shape[0]:
        ent_a = None
    if ent_b is not None and ent_b.shape[0] != ids.shape[0]:
        ent_b = None
    if "student_resp_logp_diff_B_minus_A" in z.files:
        diff = z["student_resp_logp_diff_B_minus_A"].astype(np.float64)
    else:
        diff = lb - la
    n = int(ids.shape[0])
    if la.shape[0] != n or lb.shape[0] != n or diff.shape[0] != n:
        raise ValueError(f"Length mismatch: ids={n}, la={la.shape}, lb={lb.shape}, diff={diff.shape}")

    kl, kld = low_var_kld_from_logp_diff(diff)

    pre_human: list[str] | None = None
    pre_piece: list[str] | None = None
    if csv_path is not None and (reuse_csv_labels or csv_path.is_file()):
        cached = _labels_from_csv(csv_path, n)
        if cached is not None:
            pre_human, pre_piece = cached

    if tokenizer is None and pre_human is None and x_label_mode.strip().lower() != "step_only":
        raise ValueError(
            f"{row_dir.name}: requires tokenizer or existing token_decoded column in {csv_path}; "
            "use --reuse-csv-labels after running --write-csv first, or --x-labels step_only"
        )

    t_lbl = time.perf_counter()
    if tokenizer is None:
        tick_all = [str(i) for i in range(n)]
        human_full = [""] * n
        piece_full = [""] * n
    else:
        tick_all, human_full, piece_full = _token_axis_labels(
            tokenizer,
            ids,
            x_label_mode,
            max_chars=label_max_chars,
            pre_human=pre_human,
            pre_piece=pre_piece,
        )
    if verbose:
        print(f"  [{row_dir.name}] labels: {time.perf_counter() - t_lbl:.3f}s", file=sys.stderr)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
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
            w.writerow(header)
            for i in range(n):
                row_out = [
                    i,
                    int(ids[i]),
                    human_full[i],
                    piece_full[i],
                    f"{la[i]:.6f}",
                    f"{lb[i]:.6f}",
                    f"{kl[i]:.6f}",
                    f"{kld[i]:.6f}",
                ]
                if ent_a is not None:
                    row_out.append(f"{ent_a[i]:.6f}")
                if ent_b is not None:
                    row_out.append(f"{ent_b[i]:.6f}")
                if ent_a is not None and ent_b is not None:
                    row_out.append(f"{(ent_b[i] - ent_a[i]):.6f}")
                w.writerow(row_out)

    chunk = max(4, int(tokens_per_panel))
    ranges: list[tuple[int, int]] = []
    lo = 0
    while lo < n:
        hi = min(lo + chunk, n)
        ranges.append((lo, hi))
        lo = hi

    h_per_pair = 3.35
    max_pairs = max(1, int(max_figure_height_inches // h_per_pair))
    parent = out_png.parent
    stem = out_png.stem
    suf = out_png.suffix or ".png"

    written: list[Path] = []
    npz_mtime = zpath.stat().st_mtime

    def _outputs_fresh(path_out: Path) -> bool:
        if not skip_existing or not path_out.is_file():
            return False
        if path_out.stat().st_mtime < npz_mtime:
            return False
        try:
            _validate_png(path_out)
            return True
        except ValueError:
            return False

    def _draw_batch_to_path(
        batch: list[tuple[int, int]],
        path_out: Path,
        *,
        fig_h: float,
        w_in: float,
        tick_fs: int,
        bottom_extra: float,
        title_on_first: bool,
        emphasize_kl: bool,
        dpi: int,
    ) -> None:
        fs_ann = _scaled_pt(7.0, dpi)
        fs_ylabel = _scaled_pt(9.0, dpi)
        fs_title = _scaled_pt(10.0, dpi)
        fs_xlabel = _scaled_pt(8.0, dpi)
        lw_bar = _scaled_pt(0.35, dpi, minimum=0.35)
        lw_axis = _scaled_pt(0.55, dpi, minimum=0.5)
        n_panels = len(batch)
        use_kl_tall = emphasize_kl and n_panels == 1
        if use_kl_tall:
            fig = plt.figure(figsize=(w_in, fig_h))
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[
                    float(kl_kld_height_ratio),
                    float(kld_panel_height_ratio),
                ],
                hspace=0.30,
            )
            panel_axes: list[tuple] = [(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]))]
        else:
            nrows = 2 * n_panels
            fig, axes = plt.subplots(nrows, 1, figsize=(w_in, fig_h), squeeze=False)
            ax_flat = np.atleast_1d(axes).ravel()
            panel_axes = [(ax_flat[2 * pi], ax_flat[2 * pi + 1]) for pi in range(n_panels)]

        for pi, (a, b) in enumerate(batch):
            xs = np.arange(b - a)
            seg_kl = kl[a:b]
            seg_kd = kld[a:b]
            seg_tick = tick_all[a:b]
            ax0, ax1 = panel_axes[pi]
            colors = np.where(seg_kl >= 0, "#2ca02c", "#d62728")
            # clip_on must be True: after tight ylim, outlier bars still have large data coords;
            # if False, bbox_inches=tight inflates the PNG to hundreds of thousands of pixels
            # and corrupts the file.
            ax0.bar(
                xs,
                seg_kl,
                color=colors,
                width=0.88,
                edgecolor="#1a1a1a",
                linewidth=lw_bar,
                alpha=0.88,
                clip_on=True,
            )
            ax0.axhline(0.0, color="black", linewidth=lw_axis, linestyle="-")
            if kl_tight_ylim:
                y0, y1, half = _symmetric_kl_ylim_tight(
                    seg_kl, kl_ylim_pct, kl_min_half, kl_ylim_max_half
                )
            else:
                y0, y1 = _symmetric_kl_ylim_max_abs(seg_kl)
                half = (y1 - y0) / 2.0
            ax0.set_ylim(y0, y1)
            ax0.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax0.ticklabel_format(style="plain", axis="y", useOffset=False)
            ax0.text(
                0.99,
                0.97,
                f"ylim \u00b1{half:.4g}",
                transform=ax0.transAxes,
                ha="right",
                va="top",
                fontsize=fs_ann,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
            )
            ax0.set_ylabel(plot_text["ylabel_kl"], fontsize=fs_ylabel)
            ax0.tick_params(axis="both", labelsize=_scaled_pt(7.0, dpi))
            show_title = title_on_first and pi == 0
            if show_title:
                ax0.set_title(
                    f"{row_dir.name}{title_suffix}\n{plot_text['title_extra']}",
                    fontsize=fs_title,
                )
            else:
                ax0.set_title("")
            ax0.set_xlim(-0.6, (b - a) - 0.4)
            ax0.set_xticks(xs)
            ax0.set_xticklabels(
                seg_tick,
                rotation=x_tick_rotation,
                fontsize=max(_scaled_pt(5.0, dpi), tick_fs - _scaled_pt(1.0, dpi, minimum=0.5)),
                ha="center",
                va="top",
            )

            ax1.bar(
                xs,
                seg_kd,
                color="#1f77b4",
                width=0.88,
                edgecolor="#1a1a1a",
                linewidth=max(0.25, lw_bar * 0.75),
                alpha=0.88,
                clip_on=True,
            )
            ax1.axhline(0.0, color="black", linewidth=lw_axis, linestyle="-")
            if kld_tight_ylim:
                ky0, ky1 = _kld_percentile_ylim(seg_kd, kld_ylim_pct_lo, kld_ylim_pct_hi)
                ax1.set_ylim(ky0, ky1)
                kld_span = ky1 - ky0
                ax1.text(
                    0.99,
                    0.97,
                    f"kld [{ky0:.4g}, {ky1:.4g}]",
                    transform=ax1.transAxes,
                    ha="right",
                    va="top",
                    fontsize=fs_ann,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
                )
            ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
            ax1.set_ylabel(plot_text["ylabel_kld"], fontsize=fs_ylabel)
            ax1.tick_params(axis="y", labelsize=_scaled_pt(7.0, dpi))
            ax1.set_xlim(-0.6, (b - a) - 0.4)
            ax1.set_xticks(xs)
            ax1.set_xticklabels(
                seg_tick,
                rotation=x_tick_rotation,
                fontsize=tick_fs,
                ha="center",
                va="top",
            )
            sub = f"steps [{a}..{b - 1}]  ({b - a} tokens)"
            ax1.set_xlabel(f"{plot_text['xlabel']}\n{sub}", fontsize=fs_xlabel)
            if pi < len(batch) - 1:
                ax1.xaxis.labelpad = 2.0

        bottom_margin = min(0.52, 0.14 + bottom_extra)
        if not use_kl_tall:
            fig.subplots_adjust(top=0.94, hspace=0.38, bottom=bottom_margin)
        else:
            fig.subplots_adjust(top=0.95, bottom=bottom_margin)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        t_save = time.perf_counter()
        fig.savefig(
            path_out,
            dpi=dpi,
            pad_inches=0.05,
            format="png",
        )
        plt.close(fig)
        try:
            _validate_png(path_out)
        except ValueError as e:
            path_out.unlink(missing_ok=True)
            raise RuntimeError(str(e)) from e
        if verbose:
            print(f"  saved {path_out.name}: {time.perf_counter() - t_save:.2f}s", file=sys.stderr)
        written.append(path_out)

    if one_panel_per_file:
        for pi, (a, b) in enumerate(ranges):
            seg_n = b - a
            w_in, tick_fs, bottom_extra = _figure_dims_for_tokens(seg_n, x_tick_rotation, dpi)
            fig_h = max(22.0, float(panel_figure_height_inches))
            if len(ranges) == 1:
                path_out = out_png
            else:
                path_out = parent / f"{stem}_panel{pi:02d}_step{a}_{b - 1}{suf}"
            if _outputs_fresh(path_out):
                if verbose:
                    print(f"  skip fresh {path_out.name}", file=sys.stderr)
                written.append(path_out)
                continue
            _draw_batch_to_path(
                [(a, b)],
                path_out,
                fig_h=fig_h,
                w_in=w_in,
                tick_fs=tick_fs,
                bottom_extra=bottom_extra,
                title_on_first=True,
                emphasize_kl=True,
                dpi=dpi,
            )
        return written

    total_pairs = len(ranges)
    batch_starts = list(range(0, total_pairs, max_pairs))

    for bi, b0 in enumerate(batch_starts):
        batch = ranges[b0 : b0 + max_pairs]
        w_in, tick_fs, bottom_extra = _figure_dims_for_tokens(chunk, x_tick_rotation, dpi)
        fig_h = max(4.5, min(max_figure_height_inches, h_per_pair * len(batch)))
        if len(batch_starts) == 1:
            path_out = out_png
        else:
            path_out = parent / f"{stem}_part{bi:02d}{suf}"
        if _outputs_fresh(path_out):
            if verbose:
                print(f"  skip fresh {path_out.name}", file=sys.stderr)
            written.append(path_out)
            continue
        _draw_batch_to_path(
            batch,
            path_out,
            fig_h=fig_h,
            w_in=w_in,
            tick_fs=tick_fs,
            bottom_extra=bottom_extra,
            title_on_first=True,
            emphasize_kl=False,
            dpi=dpi,
        )

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cases-dir",
        type=str,
        required=True,
        help="Root directory produced by analyze, e.g. /PATH/TO/WORKSPACE/vsi_logits_10cases",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Same model as used in analyze; required for decoding x-axis/CSV labels. "
             "May be omitted if *_per_token.csv exists and --reuse-csv-labels is set",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots and CSV; defaults to <cases-dir>/priv_kl_plots",
    )
    ap.add_argument("--write-csv", action="store_true", help="Write a per-token CSV for each row")
    ap.add_argument(
        "--only-rows",
        type=str,
        default="",
        help="Process only these subdirectory names, comma-separated, e.g. row_00000,row_00001; "
             "empty means process all row_*",
    )
    ap.add_argument(
        "--plot-lang",
        type=str,
        default="auto",
        choices=("auto", "zh", "en"),
        help="auto: try loading a system CJK font, fall back to English on failure; "
             "zh: prefer CJK font; en: all English labels (no font warnings)",
    )
    ap.add_argument(
        "--x-labels",
        type=str,
        default="step_human",
        choices=("step_human", "step_only", "piece"),
        dest="x_labels",
        help="X-axis tick format: step_human=step+decode (default); step_only=step number only; "
             "piece=step+BPE subword (hard to read)",
    )
    ap.add_argument(
        "--x-tick-rotation",
        type=float,
        default=0.0,
        help="Rotation angle for per-token x-ticks; default 0 (horizontal). Try 35/90 for dense labels",
    )
    ap.add_argument(
        "--tokens-per-panel",
        type=int,
        default=36,
        help="Number of consecutive tokens per horizontal panel block "
             "(each block: kl on top, kld below; multiple blocks stacked vertically)",
    )
    ap.add_argument(
        "--max-figure-height-inches",
        type=float,
        default=96.0,
        help="Max PNG height in inches; if total blocks x block height exceeds this, "
             "output is split into partNN files",
    )
    ap.add_argument(
        "--label-max-chars",
        type=int,
        default=10,
        help="Max characters for decoded token fragment in x-axis tick labels (prevents overflow)",
    )
    ap.add_argument(
        "--kl-ylim-pct",
        type=float,
        default=50.0,
        help="kl red/green plot: symmetric y-axis half-range = this percentile of |kl| "
             "(default 50 = median), not max|kl|; makes short bars visible",
    )
    ap.add_argument(
        "--kl-min-half",
        type=float,
        default=0.008,
        help="Minimum y-axis half-range for kl; ensures minimum visible bar height when all |kl| are very small",
    )
    ap.add_argument(
        "--kl-ylim-max-half",
        type=float,
        default=0.0,
        help="Upper bound on kl y-axis half-range; >0 further compresses axis (e.g. 0.12); 0 = no limit",
    )
    ap.add_argument(
        "--no-kl-tight-ylim",
        action="store_true",
        help="Use +/-max|kl| for kl y-axis (may be dominated by outliers; generally not recommended)",
    )
    ap.add_argument(
        "--kl-kld-height-ratio",
        type=float,
        default=2.2,
        help="Vertical height weight of kl sub-plot when using --one-panel-per-file; default 2.2",
    )
    ap.add_argument(
        "--kld-panel-height-ratio",
        type=float,
        default=1.6,
        help="Vertical height weight of kld sub-plot when using --one-panel-per-file (default 1.6, taller than previous)",
    )
    ap.add_argument(
        "--one-panel-per-file",
        action="store_true",
        help="Save each tokens-per-panel block as a separate PNG (kl+kld only), giving taller bars; "
             "with multiple blocks output files are named *_panelNN_stepA_B.png",
    )
    ap.add_argument(
        "--panel-figure-height-inches",
        type=float,
        default=38.0,
        help="Used with --one-panel-per-file: total figure height in inches; default 38; try 44 if still too short",
    )
    ap.add_argument(
        "--kld-ylim-pct",
        type=str,
        default="8,92",
        help="kld sub-plot tight y-axis: percentile interval (with padding); "
             "default 8,92 raises short bars; mutually exclusive with --no-kld-tight-ylim",
    )
    ap.add_argument(
        "--no-kld-tight-ylim",
        action="store_true",
        help="Do not tighten kld y-axis (matplotlib auto-scale); restores behavior possibly dominated by outliers",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution in DPI (default 150; 200 is sharper but larger and slower)",
    )
    ap.add_argument(
        "--high-quality",
        action="store_true",
        help="High-quality preset: dpi=200, panel height 44 (overrides --dpi / --panel-figure-height-inches)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a PNG if it is not older than its corresponding logits_summary.npz",
    )
    ap.add_argument(
        "--reuse-csv-labels",
        action="store_true",
        help="Prefer reading token text from existing *_per_token.csv in out-dir, skipping tokenizer loading",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stage timing to stderr",
    )
    args = ap.parse_args()
    plot_dpi = int(args.dpi)
    panel_h = float(args.panel_figure_height_inches)
    if args.high_quality:
        plot_dpi = max(plot_dpi, 200)
        panel_h = max(panel_h, 44.0)

    cases = Path(args.cases_dir)
    out_root = Path(args.out_dir) if args.out_dir else (cases / "priv_kl_plots")
    out_root.mkdir(parents=True, exist_ok=True)

    plot_text = _matplotlib_init(args.plot_lang)
    from transformers import AutoTokenizer

    try:
        kld_pct_lo, kld_pct_hi = _parse_kld_ylim_pct(args.kld_ylim_pct)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    row_dirs = sorted(cases.glob("row_*"))
    if not row_dirs:
        # single analyze output may be written directly to cases root
        if (cases / "logits_summary.npz").is_file():
            row_dirs = [cases]
        else:
            raise SystemExit(f"No row_* subdirectories or root logits_summary.npz found: {cases}")

    only = {s.strip() for s in args.only_rows.split(",") if s.strip()}
    if only:
        row_dirs = [d for d in row_dirs if d.name in only]

    model_path = (args.model_path or "").strip()
    if not model_path:
        sj = cases / "manifest.json"
        if sj.is_file():
            try:
                model_path = (json.loads(sj.read_text(encoding="utf-8")).get("model_path") or "").strip()
            except json.JSONDecodeError:
                pass

    x_labels_run = args.x_labels.strip().lower()
    need_tok = x_labels_run != "step_only"
    if args.write_csv:
        need_tok = True
    if args.reuse_csv_labels and need_tok:
        missing_csv: list[str] = []
        bad_csv: list[str] = []
        for rd in row_dirs:
            name = rd.name if rd.name.startswith("row_") else cases.name
            csv_p = out_root / f"{name}_per_token.csv"
            if not csv_p.is_file():
                missing_csv.append(str(csv_p))
                continue
            n = int(
                np.load(rd / "logits_summary.npz")[
                    "A_student_prompt_student_resp__response_token_ids"
                ].shape[0]
            )
            if _labels_from_csv(csv_p, n) is None:
                bad_csv.append(str(csv_p))
        if not missing_csv and not bad_csv:
            need_tok = False
        elif model_path:
            print(
                "Warning: --reuse-csv-labels found no usable CSV; falling back to tokenizer for x-axis token labels.\n"
                f"  missing CSV: {len(missing_csv)}  invalid: {len(bad_csv)}  model={model_path}",
                file=sys.stderr,
            )
            need_tok = True
        else:
            print(
                "Warning: no CSV and no model-path/manifest found; x-axis will show step numbers only. "
                "Use --write-csv or ensure cases-dir/manifest.json contains model_path.",
                file=sys.stderr,
            )
            need_tok = False
            x_labels_run = "step_only"

    tok = None
    if need_tok:
        if not model_path:
            raise SystemExit(
                "--model-path required (or model_path in cases-dir/manifest.json).\n"
                "  Recommended first run: --model-path <model> --write-csv\n"
                "  Step numbers only: --x-labels step_only"
            )
        t_tok = time.perf_counter()
        try:
            tok = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True,
            )
        except (OSError, ValueError):
            tok = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        if args.verbose:
            print(f"load tokenizer: {time.perf_counter() - t_tok:.2f}s", file=sys.stderr)
    elif args.verbose:
        print("skip tokenizer (reuse CSV / step_only)", file=sys.stderr)

    meta_line = ""
    sj = cases / "manifest.json"
    if sj.is_file():
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
            meta_line = f" parquet={meta.get('parquet', '')}"
        except json.JSONDecodeError:
            pass

    for rd in row_dirs:
        name = rd.name if rd.name.startswith("row_") else cases.name
        png = out_root / f"{name}_priv_kl_student_tokens.png"
        csv_p = out_root / f"{name}_per_token.csv"
        if not args.write_csv and not args.reuse_csv_labels:
            csv_p = None
        title_suffix = meta_line
        t_row = time.perf_counter()
        paths = _plot_one_row(
            rd,
            tok,
            png,
            csv_p,
            plot_text,
            x_labels_run,
            x_tick_rotation=args.x_tick_rotation,
            tokens_per_panel=args.tokens_per_panel,
            max_figure_height_inches=args.max_figure_height_inches,
            label_max_chars=args.label_max_chars,
            kl_tight_ylim=not bool(args.no_kl_tight_ylim),
            kl_ylim_pct=float(args.kl_ylim_pct),
            kl_min_half=float(args.kl_min_half),
            kl_ylim_max_half=(
                float(args.kl_ylim_max_half) if float(args.kl_ylim_max_half) > 0 else None
            ),
            one_panel_per_file=bool(args.one_panel_per_file),
            panel_figure_height_inches=panel_h,
            kl_kld_height_ratio=float(args.kl_kld_height_ratio),
            kld_panel_height_ratio=float(args.kld_panel_height_ratio),
            kld_tight_ylim=not bool(args.no_kld_tight_ylim),
            kld_ylim_pct_lo=kld_pct_lo,
            kld_ylim_pct_hi=kld_pct_hi,
            title_suffix=title_suffix,
            dpi=plot_dpi,
            skip_existing=bool(args.skip_existing),
            reuse_csv_labels=bool(args.reuse_csv_labels),
            verbose=bool(args.verbose),
        )
        if args.verbose:
            print(f"row {name} total: {time.perf_counter() - t_row:.2f}s", file=sys.stderr)
        for p in paths:
            print(f"Wrote {p}")
        if args.write_csv and csv_p is not None:
            print(f"Wrote {csv_p}")

    readme = out_root / "README.txt"
    readme.write_text(
        "Column descriptions (CSV)\n"
        "  token_decoded       : tokenizer.decode([id]); more readable than token_piece (BPE subword in token_piece)\n"
        "  logp_student_ctx_A  : log p of the selected token under student prompt + student rollout (context A)\n"
        "  logp_teacher_ctx_B  : log p of the same token under teacher prompt (with privilege) + same student rollout (context B)\n"
        "  kl_ref_minus_student_B_minus_A : first-order quantity aligned with training ref_lp - log_lp\n"
        "  low_var_kld_proxy   : per-token low_var_kl scalar (before kl_mask/kl_coef enters the loss)\n"
        "  entropy_student_ctx_A : Shannon entropy over full vocabulary under student prompt + student rollout\n"
        "  entropy_teacher_ctx_B : Shannon entropy over full vocabulary under teacher prompt + same student rollout\n"
        "  entropy_diff_B_minus_A : difference of the above two (Teacher - Student)\n"
        "Plot: each token is one bar + one x-tick per block; blocks stacked vertically; long sequences split into *_part00.png ...\n"
        "With --one-panel-per-file, each block is a separate figure (named *_panelNN_stepA_B.png) with taller kl/kld sub-plots.\n"
        "The kl red/green plot defaults to --kl-ylim-pct percentile tight-axis (not max|kl|); "
        "single-block figures use --kl-kld-height-ratio + larger --panel-figure-height-inches.\n",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
