# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import numpy as np
import torch

from ..protocol import DataProto
from ..utils import torch_functional as VF
from .core_algos import compute_kl


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }


def compute_privileged_kl_entropy_metrics(
    batch: DataProto,
    correctness_threshold: float = 0.5,
    *,
    incorrect_only: bool = True,
) -> dict[str, Any]:
    """KL/entropy scalars over a training batch (by default only incorrect-trajectory tokens, aligned with privileged_kl_only_on_incorrect)."""
    b = batch.batch
    if "old_log_probs" not in b or "ref_log_probs" not in b or "response_mask" not in b:
        return {}
    old_lp = b["old_log_probs"]
    ref_lp = b["ref_log_probs"]
    response_mask = b["response_mask"]
    if old_lp.shape != ref_lp.shape or old_lp.shape != response_mask.shape:
        return {}

    effective_mask = response_mask
    n_incorrect = n_correct = 0
    if "token_level_rewards" in b:
        correct = _incorrect_seq_mask_from_batch(b, correctness_threshold)
        n_correct = int(correct.sum().item())
        n_incorrect = int((~correct).sum().item())
        if incorrect_only:
            im = (~correct).unsqueeze(-1).to(dtype=response_mask.dtype) * response_mask
            if im.sum() <= 0:
                return {
                    "kl_entropy/stratify_n_correct_samples": float(n_correct),
                    "kl_entropy/stratify_n_incorrect_samples": float(n_incorrect),
                    "kl_entropy/tokens_total": 0.0,
                }
            effective_mask = im

    kl_diff = ref_lp - old_lp
    gap = kl_diff.abs()
    kld = compute_kl(old_lp, ref_lp, kl_penalty="low_var_kl")
    h_stu_s = (-old_lp).clamp(min=0.0)
    h_tea_s = (-ref_lp).clamp(min=0.0)

    def _masked_mean(t: torch.Tensor, m: torch.Tensor) -> Optional[float]:
        den = m.sum()
        if den <= 0:
            return None
        return float((t * m).sum().detach().item() / den.detach().item())

    def _masked_quantile(t: torch.Tensor, m: torch.Tensor, q: float) -> Optional[float]:
        vals = t[m.bool()]
        if vals.numel() == 0:
            return None
        return float(torch.quantile(vals.float(), q).detach().item())

    out: dict[str, Any] = {
        "kl_entropy/tokens_total": float(effective_mask.sum().detach().item()),
        "kl_entropy/kl_teacher_minus_student_mean": float(
            VF.masked_mean(kl_diff, effective_mask).detach().item()
        ),
        "kl_entropy/abs_kl_mean": float(VF.masked_mean(gap, effective_mask).detach().item()),
        "kl_entropy/low_var_kl_mean": float(VF.masked_mean(kld, effective_mask).detach().item()),
        "kl_entropy/h_stu_sampled_mean": float(VF.masked_mean(h_stu_s, effective_mask).detach().item()),
        "kl_entropy/h_tea_sampled_mean": float(VF.masked_mean(h_tea_s, effective_mask).detach().item()),
    }
    if n_correct or n_incorrect:
        out["kl_entropy/stratify_n_correct_samples"] = float(n_correct)
        out["kl_entropy/stratify_n_incorrect_samples"] = float(n_incorrect)

    q95 = _masked_quantile(gap, effective_mask, 0.95)
    if q95 is not None:
        out["kl_entropy/abs_kl_p95"] = q95
    q95_lv = _masked_quantile(kld, effective_mask, 0.95)
    if q95_lv is not None:
        out["kl_entropy/low_var_kl_p95"] = q95_lv

    if "student_entropy_norm" in b:
        m = _masked_mean(b["student_entropy_norm"], effective_mask)
        if m is not None:
            out["kl_entropy/h_stu_full_vocab_norm_mean"] = m
    if "ref_teacher_entropy_norm" in b:
        m = _masked_mean(b["ref_teacher_entropy_norm"], effective_mask)
        if m is not None:
            out["kl_entropy/h_tea_full_vocab_norm_mean"] = m

    return out


def _incorrect_seq_mask_from_batch(b: dict, correctness_threshold: float) -> torch.Tensor:
    response_mask = b["response_mask"]
    tlr = b["token_level_rewards"]
    rl = response_mask.sum(dim=-1, keepdim=True).long()
    li = (rl - 1).clamp(min=0)
    seq_r = torch.gather(tlr, dim=1, index=li).squeeze(-1)
    return seq_r >= correctness_threshold
