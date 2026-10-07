# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Line-consistent port of SDAR ``verl/trainer/ppo/sdar_utils.py`` for privileged KL replacement.
"""Confidence-Gated Teacher Distillation (SDAR) — same formulas as SDAR repo."""

from __future__ import annotations

import torch

from .core_algos import agg_loss


def loss_avg_mode_to_sdar_agg_mode(mode: str) -> str:
    """Map EasyR1 ``ActorConfig.loss_avg_mode`` to ``agg_loss`` mode strings."""
    m = (mode or "token").strip().lower()
    if m == "token":
        return "token-mean"
    if m == "seq":
        return "seq-mean-token-mean"
    if m == "maxlen_seq":
        return "maxlen_seq"
    raise ValueError(f"loss_avg_mode={mode!r} has no SDAR agg_loss mapping (use token, seq, maxlen_seq).")


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-Gated Teacher Distillation loss (verbatim structure vs SDAR ``sdar_utils.compute_sdar_loss``).

    L_SDAR = agg( g_t * (log pi_teacher - log pi_student) )
    where g_t = sigmoid(beta * delta_t), delta_t = log pi_teacher - log pi_student.

    The gate g_t is detached so gradients only flow through the student log-probs.
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    gate = torch.sigmoid(gate_beta * delta_t).detach()

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
    }

    return loss, metrics
