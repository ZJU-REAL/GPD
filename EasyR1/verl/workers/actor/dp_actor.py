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
"""
Implement Actor
"""

import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import (
    average_loss,
    compute_forward_kl_full_vocab,
    compute_kl,
    compute_policy_loss,
    reduce_opsd_distillation_loss,
)
from ...trainer.sdar_utils import compute_sdar_loss, loss_avg_mode_to_sdar_agg_mode
from ...utils import torch_functional as VF
from ...utils.peft_utils import is_peft_model, opsd_teacher_disable_adapter_ctx
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.tip_opd import (
    normalized_student_entropy_rows_chunked,
    teacher_low_entropy_median_gate_scale,
    tip_q1_q3_mask_from_entropy_and_kld,
)
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        teacher_module: Optional[nn.Module] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.teacher_module = teacher_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _opsd_teacher_ctx(self):
        if getattr(self.config, "opsd_fixed_teacher", False) and is_peft_model(self.actor_module):
            return opsd_teacher_disable_adapter_ctx(self.actor_module)
        return nullcontext()

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        return_tip_student_entropy_norm: bool = False,
        return_response_logits: bool = False,
        forward_module: Optional[nn.Module] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns:
            log_probs: # (bs, response_len)
            If return_tip_student_entropy_norm=True, returns (log_probs, h_norm) where h_norm is the full-vocab student
            entropy / ln(V) per response position, consistent with normalized entropy in OPSD_OnPolicyDistillation/src/opd/losses.py
            (computed in chunks).
            If return_response_logits=True, also appends (bs, response_len, vocab) logits (already divided by temperature).
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        tip_h: Optional[torch.Tensor] = None
        resp_logits: Optional[torch.Tensor] = None
        ent_chunk = int(getattr(self.config, "privileged_kl_tip_entropy_chunk_size", 256) or 256)
        module = forward_module if forward_module is not None else self.actor_module

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            if return_tip_student_entropy_norm:
                ent_shard = normalized_student_entropy_rows_chunked(
                    logits_rmpad.detach(), ent_chunk
                )
                if self.config.ulysses_size > 1:
                    ent_shard = gather_outputs_and_unpad(
                        ent_shard,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                full_ent = pad_input(
                    hidden_states=ent_shard.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                tip_h = full_ent.squeeze(-1)[:, -response_length - 1 : -1]

            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
            if return_response_logits:
                logits_for_pad = logits_rmpad
                if self.config.ulysses_size > 1:
                    logits_for_pad = gather_outputs_and_unpad(
                        logits_for_pad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                full_logits = pad_input(
                    hidden_states=logits_for_pad, indices=indices, batch=batch_size, seqlen=seqlen
                )
                resp_logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            if return_tip_student_entropy_norm:
                tip_h = normalized_student_entropy_rows_chunked(
                    logits.detach().reshape(-1, logits.size(-1)),
                    ent_chunk,
                ).view(logits.shape[0], logits.shape[1])
            if return_response_logits:
                resp_logits = logits
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        if return_tip_student_entropy_norm and return_response_logits:
            assert tip_h is not None and resp_logits is not None
            return log_probs, tip_h, resp_logits
        if return_tip_student_entropy_norm:
            assert tip_h is not None
            return log_probs, tip_h
        if return_response_logits:
            assert resp_logits is not None
            return log_probs, resp_logits
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @torch.no_grad()
    def compute_log_prob_with_entropy_norm(self, data: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        """Same forward pass as ``compute_log_prob``, additionally returning the full-vocab Shannon entropy / ln(V)
        of the Teacher at each response position, shape (bs, response_len).

        Used for per-token correction weights in privileged KL ('strengthen at low entropy, weaken at high entropy');
        not called on the main path by default.
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst: list[torch.Tensor] = []
        ent_lst: list[torch.Tensor] = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs + teacher H", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            lp, h_norm = self._forward_micro_batch(
                model_inputs, temperature=temperature, return_tip_student_entropy_norm=True
            )
            log_probs_lst.append(lp)
            ent_lst.append(h_norm)

        log_probs = torch.concat(log_probs_lst, dim=0)
        h_all = torch.concat(ent_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            h_all = restore_dynamic_batch(h_all, batch_idx_list)

        return log_probs, h_all

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        _opsd_kl_only = getattr(self.config, "opsd_privileged_kl_only_no_grpo", False)
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        if self.config.use_action_weight:
            select_keys.extend(["response_action_mask", "response_grounding_mask"])

        if not _opsd_kl_only:
            select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        if getattr(self.config, "use_frozen_base_kl", False) and float(
            getattr(self.config, "frozen_base_kl_coef", 0.0) or 0.0
        ) > 0:
            select_keys.append("frozen_base_ref_log_probs")
        if bool(getattr(self.config, "privileged_kl_teacher_entropy_reweight", False)) and (
            "ref_teacher_entropy_norm" in data.batch
        ):
            select_keys.append("ref_teacher_entropy_norm")
        if self.config.use_kl_loss and "token_level_rewards" in data.batch:
            select_keys.append("token_level_rewards")
        if bool(getattr(self.config, "privileged_kl_full_vocab", False)):
            select_keys.extend(
                ["teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"]
            )
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    if not _opsd_kl_only:
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                    pg_loss_weighted_response_mask = response_mask
                    kl_loss_weighted_response_mask = response_mask
                    pg_token_weight: Optional[torch.Tensor] = None
                    if self.config.use_action_weight and "response_action_mask" in model_inputs and "response_grounding_mask" in model_inputs:
                        # print("Calculating action and grounding weighted policy masks ...")
                        action_mask = model_inputs.pop("response_action_mask")
                        grounding_mask = model_inputs.pop("response_grounding_mask")
                        action_mask = action_mask * (1 - grounding_mask)   # remove overlap
                        temp_response_mask = response_mask.clone()
                        temp_response_mask = temp_response_mask * (1 - ((action_mask + grounding_mask) > 0).to(temp_response_mask.dtype)) # remove overlap
                        # build the weighted response mask using weight coefficients
                        pg_loss_weighted_response_mask = (
                            temp_response_mask
                            + action_mask * self.config.pg_loss_action_weight_coef
                            + grounding_mask * self.config.pg_loss_grounding_weight_coef
                        )
                        kl_loss_weighted_response_mask = (
                            temp_response_mask
                            + action_mask
                            + grounding_mask * self.config.kl_loss_grounding_weight_coef
                        )

                    if self.config.use_action_weight:
                        pg_token_weight = pg_loss_weighted_response_mask
                    if _opsd_kl_only:
                        metrics["actor/opsd_kl_only_no_grpo"].append(1.0)
                        if getattr(self.config, "opsd_fixed_teacher", False):
                            if self.teacher_module is not None:
                                metrics["actor/opsd_fixed_teacher_frozen_ref"].append(1.0)
                            elif is_peft_model(self.actor_module):
                                metrics["actor/opsd_fixed_teacher_lora"].append(1.0)
                    # GRPO+OPD split: correct sequences use PG (GRPO advantage); incorrect sequences skip PG and use only privileged KL (requires KL-on-incorrect in algorithm config)
                    elif getattr(self.config, "split_grpo_pg_opd_kl", False):
                        if "token_level_rewards" not in model_inputs:
                            raise RuntimeError(
                                "split_grpo_pg_opd_kl=True requires batch to contain token_level_rewards (correct/incorrect mask)."
                            )
                        tlr = model_inputs["token_level_rewards"]
                        rl = response_mask.sum(dim=-1, keepdim=True).long()
                        li = (rl - 1).clamp(min=0)
                        seq_rewards = torch.gather(tlr, dim=1, index=li).squeeze(-1)
                        thr = float(getattr(self.config, "opd_correct_reward_threshold", 0.5) or 0.5)
                        correct_1d = (seq_rewards >= thr).float().unsqueeze(-1)
                        advantages = advantages * correct_1d
                        base_pg = pg_token_weight if pg_token_weight is not None else response_mask
                        pg_token_weight = base_pg * correct_1d
                        metrics["actor/opd_pg_seq_correct_ratio"].append(correct_1d.mean().item())

                    # all return: (bsz, response_length)
                    need_tip_entropy = (
                        bool(getattr(self.config, "privileged_kl_tip_q1_q3_mask", False))
                        and getattr(self.config, "privileged_kl", False)
                        and self.config.use_kl_loss
                    )
                    need_full_vocab_kl = (
                        bool(getattr(self.config, "privileged_kl_full_vocab", False))
                        and getattr(self.config, "privileged_kl", False)
                        and self.config.use_kl_loss
                    )
                    if need_tip_entropy and need_full_vocab_kl:
                        log_probs, h_norm, student_resp_logits = self._forward_micro_batch(
                            model_inputs,
                            temperature,
                            return_tip_student_entropy_norm=True,
                            return_response_logits=True,
                        )
                    elif need_tip_entropy:
                        log_probs, h_norm = self._forward_micro_batch(
                            model_inputs, temperature, return_tip_student_entropy_norm=True
                        )
                        student_resp_logits = None
                    elif need_full_vocab_kl:
                        log_probs, student_resp_logits = self._forward_micro_batch(
                            model_inputs, temperature, return_response_logits=True
                        )
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature)
                        student_resp_logits = None

                    if _opsd_kl_only:
                        pg_metrics = {
                            "ppo_kl": 0.0,
                            "entropy_loss": 0.0,
                            "pg_clipfrac_higher": 0.0,
                            "pg_clipfrac_lower": 0.0,
                        }
                    else:
                        pg_loss, pg_metrics = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                            token_weight_mask=pg_token_weight,
                        )
                    if self.config.use_kl_loss and (
                        _opsd_kl_only or "ref_log_probs" in model_inputs
                    ):
                        eff_kl_coef_override = None
                        ref_log_probs = model_inputs.get("ref_log_probs")
                        # compute kl loss (default: sampled tokens; privileged_kl_full_vocab: full-vocab KL(π_T‖π_S))
                        if not _opsd_kl_only:
                            kld_sampled = compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kld = kld_sampled
                        if need_full_vocab_kl or _opsd_kl_only:
                            if student_resp_logits is None:
                                raise RuntimeError(
                                    "privileged_kl_full_vocab=True requires student response logits, but they were not returned in the forward pass."
                                )
                            if "teacher_input_ids" not in model_inputs:
                                raise RuntimeError(
                                    "privileged_kl_full_vocab=True requires batch to contain teacher_input_ids etc.; "
                                    "ensure the driver has union-merged Teacher sequences during the ref phase."
                                )
                            teacher_inputs = {
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                                "responses": model_inputs["responses"],
                            }
                            if "multi_modal_inputs" in model_inputs:
                                teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                            fv_chunk = int(
                                getattr(self.config, "privileged_kl_full_vocab_chunk_size", 256) or 256
                            )
                            teacher_module = (
                                self.teacher_module
                                if _opsd_kl_only and self.teacher_module is not None
                                else None
                            )
                            with torch.no_grad():
                                if teacher_module is not None:
                                    _, teacher_resp_logits = self._forward_micro_batch(
                                        teacher_inputs,
                                        temperature,
                                        return_response_logits=True,
                                        forward_module=teacher_module,
                                    )
                                else:
                                    with self._opsd_teacher_ctx():
                                        _, teacher_resp_logits = self._forward_micro_batch(
                                            teacher_inputs, temperature, return_response_logits=True
                                        )
                            kld = compute_forward_kl_full_vocab(
                                student_resp_logits,
                                teacher_resp_logits,
                                chunk_size=fv_chunk,
                            )
                            metrics["actor/privileged_kl_full_vocab"].append(1.0)
                        kl_mask = kl_loss_weighted_response_mask if self.config.use_action_weight else response_mask
                        if _opsd_kl_only:
                            kl_mask = response_mask
                        # Privileged path: BiPS default applies KL only on correct trajectories;
                        # privileged_kl_only_on_correct=False includes incorrect ones;
                        # privileged_kl_only_on_incorrect=True applies KL only on incorrect ones (takes priority)
                        elif getattr(self.config, "privileged_kl", False) and "token_level_rewards" in model_inputs:
                            token_level_rewards = model_inputs["token_level_rewards"]
                            response_lengths = response_mask.sum(dim=-1, keepdim=True).long()
                            last_token_idx = (response_lengths - 1).clamp(min=0)
                            seq_rewards = torch.gather(token_level_rewards, dim=1, index=last_token_idx).squeeze(-1)
                            correct_mask = (seq_rewards >= 0.5).float().unsqueeze(-1)  # (bsz, 1)
                            metrics["actor/kl_correct_ratio"].append(correct_mask.mean().item())
                            only_inc = getattr(self.config, "privileged_kl_only_on_incorrect", False)
                            only_cor = getattr(self.config, "privileged_kl_only_on_correct", True)
                            if only_inc:
                                kl_mask = kl_mask * (1.0 - correct_mask)
                            elif only_cor:
                                kl_mask = kl_mask * correct_mask
                            # Length-aware: zero out privileged KL for sequences shorter than threshold to block 'short answer = low KL shortcut'
                            short_thr = int(getattr(self.config, "priv_kl_short_threshold", 0) or 0)
                            if short_thr > 0:
                                seq_len = response_mask.sum(dim=-1)  # (bs,) actual response token count
                                keep = (seq_len >= float(short_thr)).to(dtype=kl_mask.dtype).unsqueeze(-1)
                                kl_mask = kl_mask * keep
                                # proportion of samples in this micro-batch whose privileged KL was disabled due to short sequences
                                disabled = 1.0 - keep.squeeze(-1).mean()
                                metrics["actor/priv_kl_short_disabled_ratio"].append(
                                    float(disabled.detach().item())
                                )
                            # Incorrect-trajectory adaptive scaling: mask=per-token multiplier; kl_coef=binary mask, gap determines eff_kl_coef within this micro-batch
                            if only_inc and getattr(
                                self.config, "privileged_kl_adaptive_incorrect_low_var", False
                            ):
                                adapt_style = (
                                    getattr(
                                        self.config, "privileged_kl_adaptive_incorrect_style", "mask"
                                    )
                                    or "mask"
                                ).lower()
                                denom_per_seq = kl_mask.sum(dim=-1)  # (bs,)
                                has_kl = denom_per_seq > 1e-6
                                seq_mean = torch.zeros(
                                    denom_per_seq.shape[0], dtype=kld.dtype, device=kld.device
                                )
                                seq_mean[has_kl] = (kld * kl_mask).sum(dim=-1)[has_kl] / denom_per_seq[has_kl]
                                tau = float(self.config.privileged_kl_incorrect_low_var_tau)
                                smin = float(
                                    getattr(self.config, "privileged_kl_adaptive_incorrect_scale_min", 0.0)
                                    or 0.0
                                )
                                smax = float(
                                    getattr(self.config, "privileged_kl_adaptive_incorrect_scale_max", 1.0)
                                    or 1.0
                                )
                                incorrect_1d = (correct_mask.squeeze(-1) < 0.5).float()
                                inc_has = (incorrect_1d > 0.5) & has_kl

                                if adapt_style == "kl_coef":
                                    if inc_has.any():
                                        gap_bar = seq_mean[inc_has].mean()
                                        gb = float(gap_bar.detach().item())
                                    else:
                                        gb = float(tau)
                                    if getattr(self.config, "privileged_kl_incorrect_low_var_step_mode", False):
                                        t_val = 1.0 if gb > tau else 0.0
                                    else:
                                        cap = float(self.config.privileged_kl_incorrect_low_var_cap)
                                        span = max(cap - tau, 1e-8)
                                        t_val = max(0.0, min(1.0, (gb - tau) / span))
                                    eff_kl_coef_override = smin + t_val * (smax - smin)
                                    metrics["actor/priv_kl_adapt_scale_mean_on_incorrect"].append(
                                        float(eff_kl_coef_override)
                                    )
                                    metrics["actor/priv_kl_adapt_seq_kl_mean_on_incorrect"].append(
                                        gb if inc_has.any() else 0.0
                                    )
                                else:
                                    if getattr(self.config, "privileged_kl_incorrect_low_var_step_mode", False):
                                        scale = (seq_mean > tau).to(dtype=kld.dtype)
                                    else:
                                        cap = float(self.config.privileged_kl_incorrect_low_var_cap)
                                        span = max(cap - tau, 1e-8)
                                        scale = torch.clamp((seq_mean - tau) / span, 0.0, 1.0)
                                    scale = smin + scale * (smax - smin)
                                    scale = scale * incorrect_1d * has_kl.float()
                                    kl_mask = kl_mask * scale.unsqueeze(-1)
                                    inc_idx = incorrect_1d > 0.5
                                    if inc_idx.any():
                                        metrics["actor/priv_kl_adapt_scale_mean_on_incorrect"].append(
                                            scale[inc_idx].mean().detach().item()
                                        )
                                        metrics["actor/priv_kl_adapt_seq_kl_mean_on_incorrect"].append(
                                            seq_mean[inc_idx].mean().detach().item()
                                        )
                                    else:
                                        metrics["actor/priv_kl_adapt_scale_mean_on_incorrect"].append(0.0)
                                        metrics["actor/priv_kl_adapt_seq_kl_mean_on_incorrect"].append(0.0)
                        if (
                            bool(getattr(self.config, "privileged_kl_tip_q1_q3_mask", False))
                            and getattr(self.config, "privileged_kl", False)
                            and need_tip_entropy
                        ):
                            tip_mask, tip_stats = tip_q1_q3_mask_from_entropy_and_kld(
                                h_norm, kld_sampled, kl_mask
                            )
                            kl_mask = kl_mask * tip_mask
                            append_to_dict(metrics, tip_stats)
                        # Teacher full-vocab entropy (ref): mimics TIP — on incorrect × current kl_mask positions, take median of H/lnV;
                        # h<=median retains privileged KL; h>median disables (high-entropy positions are not tightly constrained).
                        if (
                            bool(getattr(self.config, "privileged_kl_teacher_entropy_reweight", False))
                            and getattr(self.config, "privileged_kl", False)
                            and "ref_teacher_entropy_norm" in model_inputs
                            and "token_level_rewards" in model_inputs
                        ):
                            tlr_te = model_inputs["token_level_rewards"]
                            rl_te = response_mask.sum(dim=-1, keepdim=True).long()
                            li_te = (rl_te - 1).clamp(min=0)
                            seq_rewards_te = torch.gather(tlr_te, dim=1, index=li_te).squeeze(-1)
                            correct_mask_te = (seq_rewards_te >= 0.5).float().unsqueeze(-1).to(
                                dtype=kld.dtype, device=kld.device
                            )
                            incorrect_exp = 1.0 - correct_mask_te
                            h_t = model_inputs["ref_teacher_entropy_norm"].to(
                                device=kld.device, dtype=kld.dtype
                            )
                            scale_resp, te_stats = teacher_low_entropy_median_gate_scale(
                                h_t, kl_mask, incorrect_exp
                            )
                            kl_mask = kl_mask * scale_resp
                            append_to_dict(metrics, te_stats)
                            inc_m = incorrect_exp * response_mask
                            if inc_m.sum() > 1e-6:
                                append_to_dict(
                                    metrics,
                                    {
                                        "actor/priv_kl_teacher_ent_h_mean_on_incorrect": float(
                                            (h_t.clamp(0.0, 1.0).detach() * inc_m)
                                            .sum()
                                            .div(inc_m.sum().clamp(min=1.0))
                                            .item()
                                        ),
                                    },
                                )
                            else:
                                append_to_dict(
                                    metrics,
                                    {"actor/priv_kl_teacher_ent_h_mean_on_incorrect": 0.0},
                                )
                        if _opsd_kl_only:
                            if not need_full_vocab_kl:
                                raise RuntimeError(
                                    "opsd_privileged_kl_only_no_grpo=True requires privileged_kl_full_vocab "
                                    "(should be automatically enabled by config in OPSD mode)."
                                )
                            _token_clip = float(getattr(self.config, "opsd_jsd_token_clip", 0.05) or 0.0)
                            kl_loss = reduce_opsd_distillation_loss(
                                kld,
                                kl_mask,
                                token_clip=_token_clip if _token_clip > 0 else None,
                                mode=self.config.loss_avg_mode,
                            )
                            kl_term_coef = 1.0
                            loss = kl_loss
                            append_to_dict(
                                metrics,
                                {
                                    "actor/opsd_jsd_token_clip": _token_clip,
                                    "actor/opsd_pg_skipped": 1.0,
                                    "actor/kl_loss": kl_loss.detach().item(),
                                    "actor/kl_coef": kl_term_coef,
                                },
                            )
                        else:
                            use_priv_sdar_loss = bool(
                                getattr(self.config, "privileged_kl_sdar_sigmoid_gate", False)
                            ) and getattr(self.config, "privileged_kl", False)
                            # SRPO-style: exp(β·ref_lp) — the more confident the Teacher is in the current sampled token, the larger the KL weight.
                            beta_tc = float(getattr(self.config, "kl_teacher_confidence_weight_beta", 0.0) or 0.0)
                            gamma_tc = float(
                                getattr(self.config, "kl_teacher_disconfidence_weight_gamma", 0.0) or 0.0
                            )
                            incorrect_seq_exp = None
                            if gamma_tc > 0.0 and "token_level_rewards" in model_inputs:
                                tlr_g = model_inputs["token_level_rewards"]
                                rl_g = response_mask.sum(dim=-1, keepdim=True).long()
                                li_g = (rl_g - 1).clamp(min=0)
                                sr_g = torch.gather(tlr_g, dim=1, index=li_g).squeeze(-1)
                                incorrect_seq_exp = (sr_g < 0.5).float().unsqueeze(-1).to(
                                    dtype=kld.dtype, device=kld.device
                                )

                            if use_priv_sdar_loss:
                                b_sd = float(getattr(self.config, "privileged_kl_sdar_gate_beta", 5.0) or 5.0)
                                sdar_agg = loss_avg_mode_to_sdar_agg_mode(self.config.loss_avg_mode)
                                kl_loss, sdar_m = compute_sdar_loss(
                                    student_log_probs=log_probs,
                                    teacher_log_probs=ref_log_probs,
                                    response_mask=kl_mask,
                                    gate_beta=b_sd,
                                    loss_agg_mode=sdar_agg,
                                )
                                append_to_dict(
                                    metrics,
                                    {
                                        "actor/priv_kl_sdar_gate_mean": sdar_m["sdar/gate_mean"],
                                        "actor/priv_kl_sdar_delta_mean": sdar_m["sdar/teacher_gap_mean"],
                                        "actor/priv_kl_sdar_gate_active_ratio": sdar_m["sdar/gate_active_ratio"],
                                        "actor/priv_kl_sdar_loss_unweighted": sdar_m["sdar/loss"],
                                    },
                                )
                            elif beta_tc > 0.0 or gamma_tc > 0.0:
                                ref_lp = ref_log_probs.detach()
                                rw = kl_mask.to(dtype=ref_lp.dtype)
                                if beta_tc > 0.0:
                                    rw = rw * torch.exp(beta_tc * ref_lp)
                                if gamma_tc > 0.0 and incorrect_seq_exp is not None:
                                    rw = rw * (
                                        incorrect_seq_exp * torch.exp(-gamma_tc * ref_lp)
                                        + (1.0 - incorrect_seq_exp)
                                    )
                                num = kl_mask.sum().clamp(min=1e-8)
                                if num.item() <= 0:
                                    kl_loss = kld.sum() * 0.0
                                else:
                                    denom = rw.sum().clamp(min=1e-8)
                                    w = rw * (num / denom)
                                    kl_loss = (kld * w).sum() / num
                                if beta_tc > 0.0:
                                    append_to_dict(
                                        metrics,
                                        {
                                            "actor/kl_teacher_conf_raw_mean": VF.masked_mean(
                                                torch.exp(beta_tc * ref_lp), kl_mask
                                            ).detach().item()
                                        },
                                    )
                                if gamma_tc > 0.0 and incorrect_seq_exp is not None:
                                    dis_raw = torch.exp(-gamma_tc * ref_lp)
                                    m_inc = kl_mask * incorrect_seq_exp
                                    append_to_dict(
                                        metrics,
                                        {
                                            "actor/kl_teacher_disconf_raw_mean": VF.masked_mean(
                                                dis_raw, m_inc
                                            ).detach().item()
                                        },
                                    )
                            else:
                                kl_loss = average_loss(kld, kl_mask, mode=self.config.loss_avg_mode)
                            kl_clip = getattr(self.config, "kl_clip", None)
                            kl_clip_sym = bool(getattr(self.config, "kl_clip_symmetric", False))
                            if kl_clip is not None and getattr(self.config, "privileged_kl", False):
                                if kl_clip_sym:
                                    kl_loss = torch.clamp(kl_loss, min=-kl_clip, max=kl_clip)
                                else:
                                    kl_loss = torch.clamp(kl_loss, max=kl_clip)
                            kl_term_coef = float(self.config.kl_coef)
                            if eff_kl_coef_override is not None:
                                kl_term_coef = float(eff_kl_coef_override)
                            loss = pg_loss + kl_loss * kl_term_coef
                            append_to_dict(
                                metrics,
                                {
                                    "actor/kl_loss": kl_loss.detach().item(),
                                    "actor/kl_coef": kl_term_coef,
                                },
                            )
                            if kl_clip is not None and getattr(self.config, "privileged_kl", False):
                                append_to_dict(
                                    metrics,
                                    {
                                        "actor/kl_clip": float(kl_clip),
                                        "actor/kl_clip_symmetric": 1.0 if kl_clip_sym else 0.0,
                                    },
                                )

                    # KL relative to the frozen initial ref (Student trajectory; complementary to ref_log_probs on Teacher sequences)
                    fb_coef = float(getattr(self.config, "frozen_base_kl_coef", 0.0) or 0.0)
                        if (
                            fb_coef > 0
                            and getattr(self.config, "use_frozen_base_kl", False)
                            and "frozen_base_ref_log_probs" in model_inputs
                        ):
                            kld_fb = compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=model_inputs["frozen_base_ref_log_probs"],
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_base_loss = average_loss(
                                kld_fb, response_mask, mode=self.config.loss_avg_mode
                            )
                            loss = loss + fb_coef * kl_base_loss
                            append_to_dict(
                                metrics,
                                {
                                    "actor/frozen_base_kl_loss": kl_base_loss.detach().item(),
                                    "actor/frozen_base_kl_coef": fb_coef,
                                },
                            )
                    else:
                        loss = pg_loss

                    # Simplified DVRP spirit: penalize mean(-log π) to suppress continuously rising surprisal/entropy during training (full DVRP requires mask/noise dual views)
                    ep_coef = float(getattr(self.config, "entropy_penalty_coef", 0.0) or 0.0)
                    if ep_coef > 0:
                        ent_surrogate = average_loss(-log_probs, response_mask, mode=self.config.loss_avg_mode)
                        loss = loss + ep_coef * ent_surrogate
                        append_to_dict(
                            metrics,
                            {"actor/entropy_penalty_term": (ep_coef * ent_surrogate).detach().item()},
                        )

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    if _opsd_kl_only:
                        batch_metrics["actor/pg_loss"] = 0.0
                    else:
                        batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
