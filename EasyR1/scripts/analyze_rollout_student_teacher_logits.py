#!/usr/bin/env python3
"""
Offline comparison of per-token distribution summaries for student/teacher rollouts
under multiple context configurations (single GPU, no Ray).

Background
----------
During training, ``compute_log_prob`` / ``compute_ref_log_probs`` call
``log_probs_from_logits`` for **only the selected response token** per step
(see ``verl/workers/actor/dp_actor.py``); full-vocabulary logits are never saved
to disk (vocab too large).

For a **single parquet sample** this script:
  1) Samples one student rollout using the student prompt;
  2) Samples one teacher rollout using the teacher prompt (containing ``priv_context_text``);
  3) Runs one forward pass per configuration (four total) over the response segment,
     and exports:
       - per-step log p of the **actual token** (the "selected" item under each distribution);
       - per-step Shannon entropy (full vocabulary, computed in chunks to avoid OOM);
       - per-step top-k logits / log_softmax summary (useful for inspecting peaks/tails).

Four forward configurations (corresponding to the distribution differences of interest):
  A. student prompt + **student** response  -> student logits on its own rollout;
  B. teacher prompt + **student** response  -> same construction as the Teacher side of the
     privileged KL during training (teacher context + fixed student tokens, length-aligned);
  C. teacher prompt + **teacher** response  -> teacher logits on its own rollout;
  D. student prompt + **teacher** response  -> how much the student weights score the tokens
     produced by the teacher.

Dependencies: same environment as EasyR1 training (transformers, torch, Qwen3-VL processor).
Sampling defaults match ``examples/gpd_config.yaml`` ``worker.rollout``:
temperature=1.0, top_p=0.95.

To align with the two teacher privilege variants used during training, **run once for each**
(the parquet ``priv_context_text`` and ``--teacher-answer-only`` must be consistent):

  A) **Full privilege (3D + reference answer)**::
      --parquet .../vsi_10k_train.parquet
      do NOT pass ``--teacher-answer-only``  (TEACHER_SYSTEM; user contains scene + reference_answer)

  B) **Answer-only privilege**::
      --parquet .../vsi_10k_answeronly_train.parquet
      MUST pass ``--teacher-answer-only``  (TEACHER_SYSTEM_ANSWER_ONLY; user contains reference_answer only)

  One-shot both: ``bash scripts/run_vsi_logits_10cases_dual.sh``

Multiple consecutive rows (starting at ``--row-index`` for ``--num-rows`` rows;
each row written to ``out_dir/row_XXXXX/``, plus a ``manifest.json``)::

    python scripts/analyze_rollout_student_teacher_logits.py \\
        --model-path /path/to/Qwen3-VL-4B-Instruct \\
        --parquet /path/to/vsi_10k_train.parquet \\
        --row-index 0 --num-rows 10 \\
        --out-dir /path/to/vsi_logits_10cases_full_priv

If samples contain a **video** column rather than images, extend the processor branch
with ``--allow-video`` or modify this script (current default: images only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

# Add EasyR1 root to path (script lives in EasyR1/scripts/)
_EASYR1 = Path(__file__).resolve().parents[1]
if str(_EASYR1) not in sys.path:
    sys.path.insert(0, str(_EASYR1))

from verl.models.transformers.qwen3_vl import get_rope_index  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402
from verl.utils.vsi_zeroshot_prompts import (  # noqa: E402
    build_student_message_list,
    build_teacher_message_list,
    extract_vsi_q_text,
    meta_from_example,
    parse_image_paths,
    resolve_teacher_image_paths,
)


def _entropy_from_logits_rows(logits_2d: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """logits_2d: (L, V) -> per-row entropy (L,) float32 CPU."""
    ent = []
    l = logits_2d.size(0)
    for s in range(0, l, chunk):
        e = logits_2d[s : s + chunk].float()
        logp = F.log_softmax(e, dim=-1)
        p = logp.exp()
        ent.append(-(p * logp).sum(dim=-1))
    return torch.cat(ent, dim=0).cpu()


def _qwen3_position_ids(
    processor,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    second_per_grid_ts: list[float] | None,
) -> torch.Tensor:
    """Consistent with the Qwen3 branch in ``verl/utils/dataset.py``: text row + vision rows -> (4, L)."""
    vision_position_ids = get_rope_index(
        processor,
        input_ids=input_ids_1d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask_1d,
    )
    text_position_ids = torch.arange(len(input_ids_1d), device=input_ids_1d.device).unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=0)


def _processor_prompt(
    processor,
    messages: list[dict[str, Any]],
    processed_images: list,
) -> dict[str, torch.Tensor]:
    try:
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
    return {k: v for k, v in out.items()}


@torch.inference_mode()
def _forward_response_logits(
    model,
    processor,
    prompt_batch: dict[str, torch.Tensor],
    response_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    prompt_batch: processor output (contains input_ids 1xP, attention_mask, pixel_values, image_grid_thw, etc.)
    response_ids: 1-D tensor of length R, continuation tokens (excluding prompt)
    Returns: logits_resp (R, V) float32 CPU, full_input_ids (1, P+R), prompt_len P
    """
    prompt_ids = prompt_batch["input_ids"].to(device)
    p_len = int(prompt_ids.size(1))
    resp = response_ids.to(device).view(1, -1)
    full_ids = torch.cat([prompt_ids, resp], dim=-1)
    full_attn = torch.ones_like(full_ids, dtype=torch.long, device=device)

    mm: dict[str, Any] = {}
    for k, v in prompt_batch.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if isinstance(v, torch.Tensor):
            mm[k] = v.to(device)
        elif k == "second_per_grid_ts" and v is not None:
            mm[k] = v

    second_per_grid_ts = mm.get("second_per_grid_ts")
    s_ts = second_per_grid_ts if isinstance(second_per_grid_ts, list) else None
    # Qwen3-VL: position_ids must be (4, batch, seq) (first row text cache, next three mrope),
    # not (1, 4, seq)
    pos = _qwen3_position_ids(
        processor,
        full_ids[0],
        full_attn[0],
        mm.get("image_grid_thw"),
        mm.get("video_grid_thw"),
        s_ts,
    ).unsqueeze(1)

    out = model(
        input_ids=full_ids,
        attention_mask=full_attn,
        position_ids=pos,
        **mm,
        use_cache=False,
    )
    logits = out.logits[0]  # (P+R, V)
    r = resp.size(1)
    # Consistent with the non-padding_free branch of dp_actor: logits position for predicting response
    sl = logits[p_len - 1 : p_len + r - 1].float().cpu()  # (R, V)
    return sl, full_ids.cpu(), p_len


def _summarize_response_logits(
    logits_resp: torch.Tensor,
    response_ids_1d: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    """logits_resp: (R, V) CPU float; response_ids_1d: (R,) aligned step-by-step with logits."""
    r, v = logits_resp.shape
    assert response_ids_1d.numel() == r
    # Offline CPU logits: avoid verl's log_probs_from_logits (Dynamo/flash-attn may still route
    # CPU tensors through Triton)
    logits_flat = logits_resp.float().reshape(r, v)
    labels_flat = response_ids_1d.long().reshape(-1)
    lp = (-F.cross_entropy(logits_flat, labels_flat, reduction="none")).numpy()
    ent = _entropy_from_logits_rows(logits_resp, chunk=2048).numpy()
    topv, topi = torch.topk(logits_resp, k=min(topk, v), dim=-1)
    return {
        "response_length": int(r),
        "logp_selected": lp.astype(np.float64),
        "entropy": ent.astype(np.float64),
        "topk_indices": topi.numpy().astype(np.int32),
        "topk_logits": topv.numpy().astype(np.float32),
        "topk_logprobs": F.log_softmax(topv.float(), dim=-1).numpy().astype(np.float32),
    }


def _analyze_single_row(
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    row_index: int,
    out_dir: Path,
    processor: Any,
    model: Any,
    device: torch.device,
) -> None:
    """Write summary.json / logits_summary.npz / rollouts_decoded.txt for a single sample to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "images" not in row or not row["images"]:
        raise SystemExit("This script only supports samples with a non-empty ``images`` field; extend the processor branch for video.")

    images_meta = row["images"]
    processed = [process_image(im, args.min_pixels, args.max_pixels) for im in images_meta]
    n_img = len(processed)
    q_text = extract_vsi_q_text(dict(row), "prompt")
    priv = row.get("priv_context_text") or ""
    priv = str(priv).strip()
    has_scene = bool(priv) and "<scene_context>" in priv and priv.strip() != "<scene_context>\n</scene_context>"
    if args.teacher_answer_only and has_scene:
        print(
            f"[row {row_index}] Warning: --teacher-answer-only is set but priv still contains <scene_context>."
            " Use vsi_10k_answeronly_*.parquet; otherwise Teacher still receives 3D text.",
            file=sys.stderr,
        )
    if not args.teacher_answer_only and not has_scene and "<reference_answer>" in priv:
        print(
            f"[row {row_index}] Note: --teacher-answer-only not set and priv has no scene_context; only the answer block is present.",
            file=sys.stderr,
        )

    answer_kind, priv_variant = meta_from_example(dict(row))
    student_paths = parse_image_paths(row.get("images"))
    teacher_paths = resolve_teacher_image_paths(
        dict(row), priv_variant, student_paths, teacher_images_key="teacher_images"
    )
    processed_teacher = [process_image(p, args.min_pixels, args.max_pixels) for p in teacher_paths]
    n_tea = len(processed_teacher)

    stu_messages = build_student_message_list(n_img, q_text, answer_kind=answer_kind)
    tea_messages = build_teacher_message_list(
        n_tea,
        q_text,
        priv,
        answer_kind=answer_kind,
        priv_variant=priv_variant,
        teacher_prompt_answer_only=args.teacher_answer_only,
    )

    stu_prompt = _processor_prompt(processor, stu_messages, processed)
    tea_prompt = _processor_prompt(processor, tea_messages, processed_teacher)

    gen_kw: dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None)
        or processor.tokenizer.eos_token_id,
    )

    def _gen(prompt_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in prompt_batch.items()}
        out = model.generate(**pb, **gen_kw)
        pl = int(pb["input_ids"].size(1))
        return out[0, pl:].cpu()

    stu_resp = _gen(stu_prompt)
    tea_resp = _gen(tea_prompt)

    scenarios: dict[str, tuple[dict[str, torch.Tensor], torch.Tensor, str]] = {
        "A_student_prompt_student_resp": (stu_prompt, stu_resp, "student context + student rollout"),
        "B_teacher_prompt_student_resp": (tea_prompt, stu_resp, "teacher context + student rollout (aligned with KL side)"),
        "C_teacher_prompt_teacher_resp": (tea_prompt, tea_resp, "teacher context + teacher rollout"),
        "D_student_prompt_teacher_resp": (stu_prompt, tea_resp, "student context + teacher rollout"),
    }

    variant = "answer_only" if args.teacher_answer_only else "full_3d_and_answer"
    report: dict[str, Any] = {
        "model_path": args.model_path,
        "parquet": args.parquet,
        "row_index": row_index,
        "variant": variant,
        "teacher_answer_only": bool(args.teacher_answer_only),
        "priv_has_scene_context": has_scene,
        "priv_context_text_preview": priv[:500] + ("..." if len(priv) > 500 else ""),
        "q_text_preview": q_text[:500] + ("..." if len(q_text) > 500 else ""),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "topk": args.topk,
        "scenarios": {},
    }

    npz_parts: dict[str, Any] = {}

    for key, (pb, resp, desc) in scenarios.items():
        logits_r, _, p_len = _forward_response_logits(model, processor, pb, resp, device)
        summ = _summarize_response_logits(logits_r, resp, topk=args.topk)
        report["scenarios"][key] = {
            "description": desc,
            "prompt_len": p_len,
            "response_length": summ["response_length"],
            "mean_logp_selected": float(np.mean(summ["logp_selected"])),
            "mean_entropy": float(np.mean(summ["entropy"])),
        }
        npz_parts[f"{key}__logp_selected"] = summ["logp_selected"]
        npz_parts[f"{key}__entropy"] = summ["entropy"]
        npz_parts[f"{key}__topk_indices"] = summ["topk_indices"]
        npz_parts[f"{key}__topk_logits"] = summ["topk_logits"]
        npz_parts[f"{key}__response_token_ids"] = resp.numpy().astype(np.int32)

    # Alignment B vs A: logp difference on the same student tokens (teacher context - student context)
    la = npz_parts["A_student_prompt_student_resp__logp_selected"]
    lb = npz_parts["B_teacher_prompt_student_resp__logp_selected"]
    if la.shape == lb.shape:
        report["student_rollout_logp_diff_teacher_minus_student"] = {
            "mean": float(np.mean(lb - la)),
            "std": float(np.std(lb - la)),
        }
        npz_parts["student_resp_logp_diff_B_minus_A"] = (lb - la).astype(np.float64)

    if tea_resp.numel() > 0:
        lc = npz_parts["C_teacher_prompt_teacher_resp__logp_selected"]
        ld = npz_parts["D_student_prompt_teacher_resp__logp_selected"]
        if lc.shape == ld.shape:
            report["teacher_rollout_logp_diff_student_ctx_minus_teacher_ctx"] = {
                "mean": float(np.mean(ld - lc)),
                "std": float(np.std(ld - lc)),
            }
            npz_parts["teacher_resp_logp_diff_D_minus_C"] = (ld - lc).astype(np.float64)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    np.savez_compressed(out_dir / "logits_summary.npz", **npz_parts)

    # Human-readable snippet: decode both rollouts
    tok = processor.tokenizer
    with open(out_dir / "rollouts_decoded.txt", "w", encoding="utf-8") as f:
        f.write("=== student rollout ===\n")
        f.write(tok.decode(stu_resp.tolist(), skip_special_tokens=False) + "\n\n")
        f.write("=== teacher rollout ===\n")
        f.write(tok.decode(tea_resp.tolist(), skip_special_tokens=False) + "\n")

    print(f"[row {row_index}] Wrote {out_dir / 'summary.json'}, {out_dir / 'logits_summary.npz'}, {out_dir / 'rollouts_decoded.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument(
        "--num-rows",
        type=int,
        default=1,
        help="Number of consecutive rows to process starting from --row-index (each row in out_dir/row_XXXXX/)",
    )
    ap.add_argument("--teacher-answer-only", action="store_true", help="Use TEACHER_SYSTEM_ANSWER_ONLY for the teacher system prompt")
    ap.add_argument("--min-pixels", type=int, default=196608)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generate; default 1.0 matches examples/gpd_config.yaml worker.rollout.temperature",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help="Nucleus sampling top_p for generate; default 0.95 matches worker.rollout.top_p",
    )
    ap.add_argument("--topk", type=int, default=20, help="Number of top-k logits to retain per step")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = ap.parse_args()

    if args.num_rows < 1:
        raise SystemExit("--num-rows must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_files=args.parquet, split="train")
    n_total = len(ds)
    if args.row_index < 0 or args.row_index >= n_total:
        raise SystemExit(f"--row-index={args.row_index} out of range; dataset has {n_total} rows")

    end_exclusive = min(args.row_index + args.num_rows, n_total)
    if end_exclusive - args.row_index < args.num_rows:
        print(
            f"Warning: requested {args.num_rows} rows but only {end_exclusive - args.row_index} remain from row {args.row_index}; truncated to end of dataset.",
            file=sys.stderr,
        )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if "Qwen3VLProcessor" not in processor.__class__.__name__:
        raise SystemExit(f"This script requires a Qwen3-VL Processor; got {processor.__class__.__name__}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    manifest_rows: list[dict[str, Any]] = []
    for ri in range(args.row_index, end_exclusive):
        torch.manual_seed(args.seed + ri)
        np.random.seed(args.seed + ri)
        # Single-row mode: write to out_dir root for backward compatibility; multi-row: row_XXXXX/
        sub = out_dir if args.num_rows == 1 else out_dir / f"row_{ri:05d}"
        row = ds[ri]
        _analyze_single_row(
            args=args,
            row=row,
            row_index=ri,
            out_dir=sub,
            processor=processor,
            model=model,
            device=device,
        )
        rel = "." if args.num_rows == 1 else str(sub.relative_to(out_dir))
        manifest_rows.append({"row_index": ri, "out_subdir": rel})

    if args.num_rows > 1:
        variant = "answer_only" if args.teacher_answer_only else "full_3d_and_answer"
        manifest = {
            "variant": variant,
            "teacher_prompt_answer_only": bool(args.teacher_answer_only),
            "parquet": args.parquet,
            "model_path": args.model_path,
            "row_index_start": args.row_index,
            "num_rows_requested": args.num_rows,
            "num_rows_written": len(manifest_rows),
            "rows": manifest_rows,
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Wrote manifest: {out_dir / 'manifest.json'} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
