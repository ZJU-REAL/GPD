# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# TIP-style token helpers aligned with OPSD_OnPolicyDistillation/src/opd/losses.py
# (normalized full-vocab student entropy over rows of logits).

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalized_student_entropy_rows_chunked(
    logits: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Per-row Shannon entropy H, normalized by ln(V) to approximately [0, 1].

    Args:
        logits: (N, V) on any device dtype; computed in float32 for stability.
        chunk_size: number of rows per chunk to cap peak memory.

    Returns:
        (N,) same device as logits (dtype default float32 for intermediate; output matches logits dtype).
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits (N, V), got shape {tuple(logits.shape)}")
    n, v = logits.shape[0], logits.shape[1]
    if n == 0:
        return logits.new_zeros((0,))
    ln_v = math.log(v)
    out = logits.new_empty((n,), dtype=torch.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = logits[start:end].float()
        lp = F.log_softmax(chunk, dim=-1)
        p = lp.exp()
        ent = -(p * lp).sum(dim=-1) / ln_v
        out[start:end] = ent.clamp(0.0, 1.0)
        del chunk, lp, p
    return out.to(dtype=logits.dtype)


def masked_median(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scalar median of values where mask > 0.5; if empty, returns 0."""
    m = mask > 0.5
    if m.sum() <= 0:
        return values.new_zeros(())
    flat = values[m]
    return flat.median()


def tip_q1_q3_mask_from_entropy_and_kld(
    h_norm: torch.Tensor,
    kld: torch.Tensor,
    kl_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build (Q1 ∪ Q3) mask on response grid using per-batch medians on kl_mask.

    Q1: high student entropy (H/lnV) & high divergence proxy (kld).
    Q3: low student entropy & high divergence (overconfident-wrong region in TIP).

    Args:
        h_norm: (bs, response_len) normalized entropy in [0, 1].
        kld: (bs, response_len) same shape (e.g. low_var_kl tensor).
        kl_mask: (bs, response_len) non-negative; typically {0,1}.

    Returns:
        tip_mask: same shape as kl_mask, float in {0,1}, subset of kl_mask positions.
        stats: small dict for logging (fractions).
    """
    m = (kl_mask > 1e-6).float()
    if m.sum() < 2:
        tip = m
        return tip, {"actor/priv_kl_tip_skipped_small_batch": 1.0, "actor/priv_kl_tip_coverage": float(m.mean().item())}

    h_det = h_norm.detach()
    d_det = kld.detach()
    med_h = masked_median(h_det, m)
    med_d = masked_median(d_det, m)

    high_h = h_det >= med_h
    low_h = ~high_h
    high_d = d_det >= med_d
    q1 = high_h & high_d & (m > 0.5)
    q3 = low_h & high_d & (m > 0.5)
    tip = (q1 | q3).to(dtype=kl_mask.dtype)
    denom = m.sum().clamp(min=1.0)
    stats = {
        "actor/priv_kl_tip_coverage": float((tip * m).sum().item() / float(denom.item())),
        "actor/priv_kl_tip_q1_frac_of_kl": float((q1.float() * m).sum().item() / float(denom.item())),
        "actor/priv_kl_tip_q3_frac_of_kl": float((q3.float() * m).sum().item() / float(denom.item())),
    }
    return tip * m, stats


def teacher_low_entropy_median_gate_scale(
    h_teacher_norm: torch.Tensor,
    kl_mask: torch.Tensor,
    incorrect_exp: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """TIP-inspired gate: compute the median of Teacher normalized entropy ``H/lnV`` over positions
    in ``kl_mask`` that belong to incorrect sequences;
    **h <= median → retain privileged KL (factor 1)**; **h > median → suppress (factor 0)**;
    correct sequences always have factor 1.

    Similar to ``tip_q1_q3_mask_from_entropy_and_kld``: if the number of valid positions < 2,
    no gating is applied (returns all-ones multiplier) and a skipped scalar is logged.

    Args:
        h_teacher_norm: (bs, response_len), Teacher full-vocab entropy / ln(V), in [0, 1].
        kl_mask: (bs, response_len), current privileged KL validity mask.
        incorrect_exp: (bs, 1) or (bs, response_len) broadcastable; 1 for incorrect sequences.

    Returns:
        scale: (bs, response_len), multiplied element-wise onto ``kl_mask`` (correct sequences always 1).
        stats: log keys under ``actor/priv_kl_teacher_ent_gate_*``.
    """
    inc = incorrect_exp
    if inc.dim() == 2 and inc.size(-1) == 1:
        inc = inc.expand_as(kl_mask)
    m = (kl_mask > 1e-6).float() * inc
    denom = m.sum().clamp(min=1.0)
    if m.sum() < 2:
        scale = torch.ones_like(h_teacher_norm)
        return scale, {
            "actor/priv_kl_teacher_ent_gate_skipped_small_batch": 1.0,
            "actor/priv_kl_teacher_ent_gate_coverage": float(m.mean().item()),
        }
    h_det = h_teacher_norm.detach()
    med_h = masked_median(h_det, m)
    gate_on_m = (h_det <= med_h).float() * m
    scale = inc * gate_on_m + (1.0 - inc)
    cov = float((gate_on_m.sum() / denom).item())
    return scale, {
        "actor/priv_kl_teacher_ent_gate_skipped_small_batch": 0.0,
        "actor/priv_kl_teacher_ent_gate_coverage": cov,
    }
