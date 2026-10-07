# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Save privileged KL entropy quantities for incorrect trajectories from training batches as ``logits_summary.npz`` (compatible with offline export scripts)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..protocol import DataProto
from .privileged_kl_case_assets import _normalize_image_paths, write_case_assets


def _seq_correct_mask(
    batch: DataProto,
    *,
    correctness_threshold: float,
) -> torch.Tensor:
    """(B,) bool: sequences whose last-token token_level_rewards >= threshold are True."""
    b = batch.batch
    response_mask = b["response_mask"]
    tlr = b["token_level_rewards"]
    rl = response_mask.sum(dim=-1, keepdim=True).long()
    li = (rl - 1).clamp(min=0)
    seq_r = torch.gather(tlr, dim=1, index=li).squeeze(-1)
    return seq_r >= correctness_threshold


def _valid_response_slice(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    idx: int,
) -> tuple[np.ndarray, int]:
    r = responses[idx]
    m = response_mask[idx]
    if torch.is_tensor(m):
        m = m.bool()
    valid_len = int(m.sum().item())
    if valid_len <= 0:
        return np.array([], dtype=np.int32), 0
    tok = r[:valid_len]
    if torch.is_tensor(tok):
        tok = tok.detach().cpu().numpy()
    return tok.astype(np.int32), valid_len


def _slice_logp(
    logp: torch.Tensor,
    idx: int,
    valid_len: int,
) -> np.ndarray:
    if valid_len <= 0:
        return np.array([], dtype=np.float64)
    row = logp[idx, :valid_len]
    return row.detach().float().cpu().numpy().astype(np.float64)


def _entropy_from_norm(
    ent_norm: Optional[torch.Tensor],
    idx: int,
    valid_len: int,
    ln_vocab: float,
) -> Optional[np.ndarray]:
    if ent_norm is None or valid_len <= 0:
        return None
    row = ent_norm[idx, :valid_len]
    return (row.detach().float().cpu().numpy().astype(np.float64) * ln_vocab)


def _select_unique_prompt_candidates(
    cand: list[int],
    batch: DataProto,
    max_n: int,
) -> list[int]:
    """Deduplicate by ``uid``, keeping the first candidate trajectory per prompt in batch order."""
    seen: set[str] = set()
    selected: list[int] = []
    uids = None
    if batch.non_tensor_batch is not None:
        uids = batch.non_tensor_batch.get("uid")

    for idx in cand:
        if uids is not None and len(uids) > idx:
            key = str(uids[idx])
        else:
            key = f"__batch_index__:{idx}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(idx)
        if len(selected) >= max_n:
            break
    return selected


def dump_incorrect_privileged_kl_entropy(
    batch: DataProto,
    *,
    global_step: int,
    save_checkpoint_path: str | Path,
    tokenizer: Any,
    model_path: str,
    correctness_threshold: float = 0.5,
    max_samples_per_step: int = 8,
    teacher_answer_only: bool = False,
    temperature: float = 1.0,
    incorrect_only: bool = True,
) -> int:
    """Write incorrect trajectories from ``batch`` to ``privileged_kl_entropy/step_XXXXXX/sample_YYYYY/``.

    Each sample directory contains ``logits_summary.npz``, ``summary.json``, and ``rollouts_decoded.txt``.
    Fields are aligned with scenarios A/B in ``scripts/analyze_rollout_student_teacher_logits.py`` to allow
    offline conversion to CSV/paste_notes via ``scripts/export_train_privileged_kl_entropy.py``.

    Returns:
        Number of samples actually written in this step.
    """
    b = batch.batch
    required = ("old_log_probs", "ref_log_probs", "response_mask", "responses", "token_level_rewards")
    if any(k not in b for k in required):
        return 0

    correct = _seq_correct_mask(batch, correctness_threshold=correctness_threshold)
    bs = int(correct.shape[0])
    if incorrect_only:
        cand = [i for i in range(bs) if not bool(correct[i].item())]
    else:
        cand = list(range(bs))

    if not cand:
        return 0

    max_n = max(1, int(max_samples_per_step or 1))
    n_cand_before_dedup = len(cand)
    cand = _select_unique_prompt_candidates(cand, batch, max_n)
    if not cand:
        return 0

    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        vocab_size = int(tokenizer.vocab_size)
    else:
        vocab_size = len(tokenizer)
    ln_vocab = float(math.log(max(vocab_size, 2)))
    step_dir = Path(save_checkpoint_path) / "privileged_kl_entropy" / f"step_{global_step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    stu_ent_norm = b.get("student_entropy_norm")
    tea_ent_norm = b.get("ref_teacher_entropy_norm")

    manifest_rows: list[dict[str, Any]] = []
    n_written = 0

    for j, idx in enumerate(cand):
        tok_ids, valid_len = _valid_response_slice(b["responses"], b["response_mask"], idx)
        if valid_len <= 0:
            continue

        la = _slice_logp(b["old_log_probs"], idx, valid_len)
        lb = _slice_logp(b["ref_log_probs"], idx, valid_len)
        ea = _entropy_from_norm(stu_ent_norm, idx, valid_len, ln_vocab)
        eb = _entropy_from_norm(tea_ent_norm, idx, valid_len, ln_vocab)

        npz: dict[str, Any] = {
            "A_student_prompt_student_resp__response_token_ids": tok_ids,
            "A_student_prompt_student_resp__logp_selected": la.astype(np.float32),
            "B_teacher_prompt_student_resp__response_token_ids": tok_ids.copy(),
            "B_teacher_prompt_student_resp__logp_selected": lb.astype(np.float32),
            "student_resp_logp_diff_B_minus_A": (lb - la).astype(np.float64),
        }
        if ea is not None:
            npz["A_student_prompt_student_resp__entropy"] = ea.astype(np.float32)
        if eb is not None:
            npz["B_teacher_prompt_student_resp__entropy"] = eb.astype(np.float32)

        sample_dir = step_dir / f"sample_{j:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / "logits_summary.npz", **npz)

        resp_text = tokenizer.decode(tok_ids.tolist(), skip_special_tokens=False)
        rollout_txt = (
            "=== student rollout (fixed for privileged KL) ===\n"
            f"{resp_text}\n"
        )
        (sample_dir / "rollouts_decoded.txt").write_text(rollout_txt, encoding="utf-8")

        rl = b["response_mask"]
        last_i = max(int(rl[idx].sum().item()) - 1, 0)
        seq_r = float(b["token_level_rewards"][idx, last_i].item())

        gt = uid = priv_preview = q_preview = q_text = priv_full = ""
        image_paths: list[str] = []
        if batch.non_tensor_batch is not None:
            nt = batch.non_tensor_batch
            if "ground_truth" in nt:
                g = nt["ground_truth"][idx]
                gt = str(g.get("ground_truth", g) if isinstance(g, dict) else g)
            if "uid" in nt:
                uid = str(nt["uid"][idx])
            if "priv_context_text" in nt:
                priv_full = str(nt["priv_context_text"][idx] or "")
                priv_preview = priv_full[:500] + ("..." if len(priv_full) > 500 else "")
            if "vsi_q_text" in nt:
                q_text = str(nt["vsi_q_text"][idx] or "")
                q_preview = q_text[:500]
            elif "raw_prompt" in nt:
                q_preview = str(nt["raw_prompt"][idx])[:500]
            elif "prompt" in nt:
                q_preview = str(nt["prompt"][idx])[:500]
            if "multi_modal_data" in nt:
                mm = nt["multi_modal_data"][idx]
                if isinstance(mm, dict):
                    image_paths = _normalize_image_paths(mm.get("images"))

        if q_text.strip() and image_paths:
            write_case_assets(
                sample_dir,
                q_text=q_text,
                ground_truth=gt,
                image_paths=image_paths,
                priv_context_text=priv_full,
                match_confidence="dump",
                match_note="saved from training batch",
            )

        summary = {
            "source": "train_privileged_kl_entropy_dump",
            "global_step": global_step,
            "batch_index": idx,
            "sample_index_in_step": j,
            "model_path": model_path,
            "variant": "answer_only" if teacher_answer_only else "full_3d_and_answer",
            "teacher_answer_only": bool(teacher_answer_only),
            "correctness_threshold": correctness_threshold,
            "sequence_reward_last_token": seq_r,
            "is_correct": False if incorrect_only else bool(correct[idx].item()),
            "response_length": valid_len,
            "temperature": temperature,
            "ground_truth": gt,
            "uid": uid,
            "priv_context_text_preview": priv_preview,
            "q_text_preview": q_preview,
            "q_text": q_text,
            "n_images": len(image_paths),
            "has_case_assets": bool(q_text.strip() and image_paths),
            "scenarios": {
                "A_student_prompt_student_resp": {
                    "description": "student context + student rollout",
                    "response_length": valid_len,
                    "mean_logp_selected": float(np.mean(la)) if la.size else None,
                    "mean_entropy": float(np.mean(ea)) if ea is not None and ea.size else None,
                },
                "B_teacher_prompt_student_resp": {
                    "description": "teacher context + student rollout (privileged KL alignment)",
                    "response_length": valid_len,
                    "mean_logp_selected": float(np.mean(lb)) if lb.size else None,
                    "mean_entropy": float(np.mean(eb)) if eb is not None and eb.size else None,
                },
            },
            "student_rollout_logp_diff_teacher_minus_student": {
                "mean": float(np.mean(lb - la)) if la.size else None,
                "std": float(np.std(lb - la)) if la.size else None,
            },
        }
        (sample_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "sample_dir": sample_dir.name,
                "batch_index": idx,
                "uid": uid,
                "ground_truth": gt,
                "sequence_reward_last_token": seq_r,
                "response_length": valid_len,
            }
        )
        n_written += 1

    if manifest_rows:
        (step_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "incorrect_only": incorrect_only,
                    "correctness_threshold": correctness_threshold,
                    "dedupe_by_uid": True,
                    "n_incorrect_before_dedup": n_cand_before_dedup,
                    "max_samples_per_step": max_n,
                    "n_samples": n_written,
                    "samples": manifest_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return n_written
