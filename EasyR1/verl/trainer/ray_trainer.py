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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import importlib.util
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from pathlib import Path


from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.dataset import process_image
from ..utils.vsi_zeroshot_prompts import build_teacher_message_list
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_privileged_kl_entropy_metrics,
    reduce_metrics,
)
from .privileged_kl_entropy_dump import dump_incorrect_privileged_kl_entropy
import torch.nn.functional as _F


def _align_batch_seqlen(batch1: "DataProto", batch2: "DataProto", pad_token_id: int = 0):
    """Right-pad tensors so the sequence dimension (last dim) matches before ``DataProto.concat``.

    Used when ``mini_rollout_batch_size`` is set: each mini-batch is collated to its own max
    length, so accumulated chunks may differ by a few tokens (e.g. multimodal image grids).

    - ``input_ids`` / ``responses``: pad with ``pad_token_id``
    - bool masks (e.g. ``response_mask``): pad with ``False``
    - other 2+D tensors (``attention_mask``, ``position_ids``, mrope ``(B,4,L)``, etc.): pad with ``0``
    """
    for key in set(batch1.batch.keys()) & set(batch2.batch.keys()):
        t1, t2 = batch1.batch[key], batch2.batch[key]
        if t1.dim() < 2:
            continue
        len1, len2 = t1.shape[-1], t2.shape[-1]
        if len1 == len2:
            continue
        if t1.dtype == torch.bool:
            pad_val: bool | int = False
        elif key in ("input_ids", "responses"):
            pad_val = pad_token_id
        else:
            pad_val = 0
        if len1 < len2:
            batch1.batch[key] = _F.pad(t1, (0, len2 - len1), value=pad_val)
        else:
            batch2.batch[key] = _F.pad(t2, (0, len1 - len2), value=pad_val)
    return batch1, batch2


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        # Runner and similar actors may be scheduled on nodes without GPUs; using available_resources()
        # from inside such an actor often falsely reports GPU=0. Use cluster_resources for total cluster GPUs.
        gpus_cluster = ray.cluster_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_cluster < gpus_required:
            _torch_n = -1
            try:
                import torch

                _torch_n = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            except Exception:
                pass
            raise ValueError(
                f"Cluster total GPUs (Ray) {gpus_cluster} < desired {gpus_required}. "
                f"available_resources['GPU']={ray.available_resources().get('GPU', 0)}; "
                f"torch.cuda.device_count()={_torch_n}. "
                "Ray cluster has no registered GPUs: run on a compute node with allocated GPUs; "
                "for Slurm add #SBATCH --gres=gpu:...; "
                "verify locally with `python -c \"import ray; ray.init(); print(ray.cluster_resources())\"` "
                "to confirm GPUs are visible; or start with `ray start --head --num-gpus=N` and export RAY_ADDRESS."
            )


def _build_teacher_batch_vsi_zeroshot(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    *,
    image_dir: Optional[str] = None,
    teacher_prompt_answer_only: bool = False,
) -> DataProto:
    """Teacher consistent with vsi_text_cot_eval; recomputes prompt input_ids / position_ids using the processor."""
    from tensordict import TensorDict

    device = batch.batch["input_ids"].device
    responses = batch.batch["responses"]
    B, _resp_len = responses.shape
    total_len = batch.batch["input_ids"].size(1)
    prompt_len = total_len - _resp_len
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    has_mrope = position_ids.dim() == 3
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    q_texts = batch.non_tensor_batch["vsi_q_text"]
    priv_texts = batch.non_tensor_batch.get("priv_context_text")
    mm = batch.non_tensor_batch["multi_modal_data"]

    if "Qwen3VLProcessor" in processor.__class__.__name__:
        from ..models.transformers.qwen3_vl import get_rope_index
    else:
        from ..models.transformers.qwen2_vl import get_rope_index

    new_prompt_ids_list: list[torch.Tensor] = []
    new_prompt_mask_list: list[torch.Tensor] = []
    new_prompt_pos_list: list[torch.Tensor] = []
    new_multi_modal_data: list[dict[str, Any]] = []
    delta_list: list[int] = []

    for i in range(B):
        q_text = str(q_texts[i])
        if priv_texts is None:
            priv = ""
        else:
            priv = priv_texts[i]
            priv = "" if priv is None else str(priv)

        answer_kind = "choice"
        if batch.non_tensor_batch.get("vsi_answer_kind") is not None:
            answer_kind = str(batch.non_tensor_batch["vsi_answer_kind"][i])
        priv_variant = "pure_grpo"
        if batch.non_tensor_batch.get("vsi_priv_variant") is not None:
            priv_variant = str(batch.non_tensor_batch["vsi_priv_variant"][i])

        student_paths = mm[i]["images"]
        teacher_paths = list(student_paths)
        if priv_variant in ("image_routed", "image_full"):
            from ..utils.vsi_zeroshot_prompts import parse_image_paths

            t_raw = batch.non_tensor_batch.get("teacher_images")
            if t_raw is not None and t_raw[i] is not None:
                parsed = parse_image_paths(t_raw[i], image_dir=image_dir)
                if parsed:
                    teacher_paths = parsed

        images = teacher_paths
        processed_images = [process_image(im, min_pixels, max_pixels) for im in images]
        n_img = len(processed_images)

        messages = build_teacher_message_list(
            n_img,
            q_text,
            priv,
            answer_kind=answer_kind,  # type: ignore[arg-type]
            priv_variant=priv_variant,
            teacher_prompt_answer_only=teacher_prompt_answer_only,
        )
        try:
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        model_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        tid = model_inputs.pop("input_ids")[0]
        tmask = model_inputs.pop("attention_mask")[0]

        if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(
                processor,
                input_ids=tid,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=tmask,
            )
            text_position_ids = torch.arange(len(tid), device=tid.device).unsqueeze(0)
            tpos = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            tpos = torch.clip(tmask.cumsum(dim=0) - 1, min=0, max=None)

        tid = tid.to(device)
        tmask = tmask.to(device)
        tpos = tpos.to(device)

        stu_prompt_tokens = int(attention_mask[i, :prompt_len].sum().item())
        tea_prompt_tokens = int(tmask.sum().item())
        delta_list.append(tea_prompt_tokens - stu_prompt_tokens)

        new_prompt_ids_list.append(tid)
        new_prompt_mask_list.append(tmask)
        new_prompt_pos_list.append(tpos)
        new_multi_modal_data.append({"images": images})

    max_tp_len = max(t.size(0) for t in new_prompt_ids_list)
    teacher_prompt_ids = torch.full((B, max_tp_len), pad_id, dtype=torch.long, device=device)
    teacher_prompt_mask = torch.zeros(B, max_tp_len, dtype=attention_mask.dtype, device=device)
    if has_mrope:
        n_pos_rows = position_ids.size(1)
        teacher_prompt_pos = torch.zeros(B, n_pos_rows, max_tp_len, dtype=position_ids.dtype, device=device)
    else:
        teacher_prompt_pos = torch.zeros(B, max_tp_len, dtype=position_ids.dtype, device=device)

    for i, (p_ids, p_mask, p_pos) in enumerate(
        zip(new_prompt_ids_list, new_prompt_mask_list, new_prompt_pos_list)
    ):
        L = p_ids.size(0)
        pad_amt = max_tp_len - L
        teacher_prompt_ids[i, pad_amt:] = p_ids
        teacher_prompt_mask[i, pad_amt:] = p_mask
        if has_mrope:
            teacher_prompt_pos[i, :, pad_amt:] = p_pos
        else:
            teacher_prompt_pos[i, pad_amt:] = p_pos

    orig_resp_pos = position_ids[:, :, prompt_len:] if has_mrope else position_ids[:, prompt_len:]
    delta_t = torch.tensor(delta_list, device=device, dtype=orig_resp_pos.dtype)
    if has_mrope:
        delta_t = delta_t.view(B, 1, 1).expand_as(orig_resp_pos)
    else:
        delta_t = delta_t.view(B, 1).expand_as(orig_resp_pos)
    teacher_resp_pos = orig_resp_pos + delta_t

    teacher_input_ids = torch.cat([teacher_prompt_ids, responses], dim=-1)
    teacher_attn_mask = torch.cat(
        [teacher_prompt_mask, batch.batch["response_mask"].to(teacher_prompt_mask.dtype)], dim=-1
    )
    teacher_position_ids = torch.cat([teacher_prompt_pos, teacher_resp_pos], dim=-1)

    new_td_dict = {k: v for k, v in batch.batch.items()}
    new_td_dict["input_ids"] = teacher_input_ids
    new_td_dict["attention_mask"] = teacher_attn_mask
    new_td_dict["position_ids"] = teacher_position_ids

    new_non_tensor_batch = dict(batch.non_tensor_batch)
    new_non_tensor_batch["multi_modal_data"] = np.array(new_multi_modal_data, dtype=object)

    new_td = TensorDict(new_td_dict, batch_size=[B])
    return DataProto(batch=new_td, non_tensor_batch=new_non_tensor_batch, meta_info=batch.meta_info)


def _build_teacher_batch_priv_insert(batch: DataProto, tokenizer) -> DataProto:
    """
    GPD Asymmetric Privileged GRPO — Teacher batch construction.

    Aligns with VSI zero-shot structure (vsi_text_cot_eval.py):
      Student: [image patches | query | response]
      Teacher: [image patches | priv_text | query | response]

    priv_text is inserted AFTER the last <|vision_end|> token in the student
    prompt, i.e., between the image region and the query text — matching the
    zero-shot layout where 3D context follows all images and precedes the question.

    Position IDs (mrope, shape B x 4 x seq_len):
      - image region: unchanged (visual grid positions preserved)
      - priv_text: sequential, continuing from last image region position
      - query + response: shifted by +P (P = priv token count for that sample)
    """
    from tensordict import TensorDict

    priv_texts = batch.non_tensor_batch.get("priv_context_text", None)
    if priv_texts is None or not any(t for t in priv_texts):
        return batch

    # <|vision_end|> marks the boundary between image patches and query text
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if vision_end_id == tokenizer.unk_token_id:
        vision_end_id = None  # model has no vision tokens; fall back gracefully

    device = batch.batch["input_ids"].device
    input_ids   = batch.batch["input_ids"]       # (B, total_len)
    attention_mask = batch.batch["attention_mask"]  # (B, total_len)
    position_ids   = batch.batch["position_ids"]    # (B, 4, total_len) mrope or (B, total_len)
    responses      = batch.batch["responses"]       # (B, resp_len)

    B         = input_ids.size(0)
    resp_len  = responses.size(1)
    total_len = input_ids.size(1)
    prompt_len = total_len - resp_len
    has_mrope  = position_ids.dim() == 3           # Qwen3-VL: (B, 4, seq_len)
    n_pos_rows = position_ids.size(1) if has_mrope else 1
    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    new_prompt_ids_list  = []
    new_prompt_mask_list = []
    new_prompt_pos_list  = []
    P_list = []  # priv token count per sample

    for i in range(B):
        priv_text     = priv_texts[i] if priv_texts[i] else ""
        priv_token_ids = tokenizer.encode(priv_text, add_special_tokens=False) if priv_text else []
        P = len(priv_token_ids)
        P_list.append(P)

        # Strip left-padding to get real prompt tokens: (S,)
        prompt_mask_i  = attention_mask[i, :prompt_len].bool()
        real_prompt_ids = input_ids[i, :prompt_len][prompt_mask_i]
        S = real_prompt_ids.size(0)

        if P == 0:
            # No privileged text for this sample → teacher == student
            teacher_prompt      = real_prompt_ids
            teacher_prompt_mask = torch.ones(S, dtype=attention_mask.dtype, device=device)
            if has_mrope:
                orig_prompt_pos = position_ids[i, :, :prompt_len][:, prompt_mask_i]  # (4, S)
                teacher_prompt_pos = orig_prompt_pos
            else:
                teacher_prompt_pos = torch.arange(S, device=device, dtype=position_ids.dtype)
        else:
            priv_t = torch.tensor(priv_token_ids, dtype=torch.long, device=device)  # (P,)

            # Find split point: right after last <|vision_end|> in real prompt
            if vision_end_id is not None:
                ve_positions = (real_prompt_ids == vision_end_id).nonzero(as_tuple=True)[0]
            else:
                ve_positions = torch.tensor([], dtype=torch.long)

            if len(ve_positions) > 0:
                # Insert priv_text between image region and query text
                split = ve_positions[-1].item() + 1  # exclusive end of image region
                img_region = real_prompt_ids[:split]   # (split,)
                qry_region = real_prompt_ids[split:]   # (S - split,)
                teacher_prompt = torch.cat([img_region, priv_t, qry_region], dim=0)  # (S+P,)
                teacher_prompt_mask = torch.ones(S + P, dtype=attention_mask.dtype, device=device)

                if has_mrope:
                    orig_prompt_pos = position_ids[i, :, :prompt_len][:, prompt_mask_i]  # (4, S)
                    img_pos = orig_prompt_pos[:, :split]    # (4, split)
                    qry_pos = orig_prompt_pos[:, split:]    # (4, S-split)

                    # priv_text: all 4 rows continue sequentially from last image position
                    # <|vision_end|> is a text token so all 4 rows have the same value there
                    base = img_pos[:, -1:]                  # (4, 1)
                    priv_pos = base + torch.arange(1, P + 1, device=device,
                                                   dtype=position_ids.dtype).unsqueeze(0)  # (4, P)
                    shifted_qry_pos = qry_pos + P           # (4, S-split)
                    teacher_prompt_pos = torch.cat([img_pos, priv_pos, shifted_qry_pos], dim=-1)  # (4, S+P)
                else:
                    teacher_prompt_pos = torch.arange(S + P, device=device, dtype=position_ids.dtype)
            else:
                # No vision tokens found (text-only sample) — fall back: append priv after prompt
                teacher_prompt      = torch.cat([real_prompt_ids, priv_t], dim=0)
                teacher_prompt_mask = torch.ones(S + P, dtype=attention_mask.dtype, device=device)
                if has_mrope:
                    orig_prompt_pos = position_ids[i, :, :prompt_len][:, prompt_mask_i]
                    base    = orig_prompt_pos[:, -1:]
                    priv_pos = base + torch.arange(1, P + 1, device=device,
                                                   dtype=position_ids.dtype).unsqueeze(0)
                    teacher_prompt_pos = torch.cat([orig_prompt_pos, priv_pos], dim=-1)
                else:
                    teacher_prompt_pos = torch.arange(S + P, device=device, dtype=position_ids.dtype)

        new_prompt_ids_list.append(teacher_prompt)
        new_prompt_mask_list.append(teacher_prompt_mask)
        new_prompt_pos_list.append(teacher_prompt_pos)

    # Left-pad all teacher prompts to the same length
    max_tp_len = max(p.size(0) for p in new_prompt_ids_list)

    teacher_prompt_ids  = torch.full((B, max_tp_len), pad_id, dtype=torch.long, device=device)
    teacher_prompt_mask = torch.zeros(B, max_tp_len, dtype=attention_mask.dtype, device=device)
    if has_mrope:
        teacher_prompt_pos = torch.zeros(B, n_pos_rows, max_tp_len, dtype=position_ids.dtype, device=device)
    else:
        teacher_prompt_pos = torch.zeros(B, max_tp_len, dtype=position_ids.dtype, device=device)

    for i, (p_ids, p_mask, p_pos) in enumerate(
        zip(new_prompt_ids_list, new_prompt_mask_list, new_prompt_pos_list)
    ):
        L   = p_ids.size(0)
        pad = max_tp_len - L
        teacher_prompt_ids[i, pad:]  = p_ids
        teacher_prompt_mask[i, pad:] = p_mask
        if has_mrope:
            teacher_prompt_pos[i, :, pad:] = p_pos
        else:
            teacher_prompt_pos[i, pad:] = p_pos

    # Response positions: shift by per-sample P (priv inserted before query)
    orig_resp_pos = position_ids[:, :, prompt_len:] if has_mrope else position_ids[:, prompt_len:]
    P_tensor = torch.tensor(P_list, device=device, dtype=orig_resp_pos.dtype)
    if has_mrope:
        P_shifts = P_tensor.view(B, 1, 1).expand_as(orig_resp_pos)
    else:
        P_shifts = P_tensor.view(B, 1).expand_as(orig_resp_pos)
    teacher_resp_pos = orig_resp_pos + P_shifts

    # Assemble full teacher sequence: [teacher_prompt | response]
    teacher_input_ids = torch.cat([teacher_prompt_ids, responses], dim=-1)
    teacher_attn_mask = torch.cat(
        [teacher_prompt_mask, batch.batch["response_mask"].to(teacher_prompt_mask.dtype)], dim=-1
    )
    teacher_position_ids = torch.cat([teacher_prompt_pos, teacher_resp_pos], dim=-1)

    new_td_dict = {k: v for k, v in batch.batch.items()}
    new_td_dict["input_ids"]      = teacher_input_ids
    new_td_dict["attention_mask"] = teacher_attn_mask
    new_td_dict["position_ids"]   = teacher_position_ids

    new_td = TensorDict(new_td_dict, batch_size=[B])
    return DataProto(batch=new_td, non_tensor_batch=batch.non_tensor_batch, meta_info=batch.meta_info)


def build_teacher_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    use_vsi_zeroshot_prompts: bool = False,
    teacher_prompt_answer_only: bool = False,
    image_dir: Optional[str] = None,
) -> DataProto:
    if use_vsi_zeroshot_prompts:
        if processor is None:
            raise ValueError("build_teacher_batch(..., use_vsi_zeroshot_prompts=True) requires processor")
        return _build_teacher_batch_vsi_zeroshot(
            batch,
            tokenizer,
            processor,
            min_pixels,
            max_pixels,
            image_dir=image_dir,
            teacher_prompt_answer_only=teacher_prompt_answer_only,
        )
    return _build_teacher_batch_priv_insert(batch, tokenizer)


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, **kwargs):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    adv_inputs.update(kwargs)
    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.algorithm.privileged_rl:
            print(
                "RL mode: privileged asymmetric (teacher ref + `use_kl_loss` per config; "
                "see algorithm.privileged_rl / use_kl_loss)."
            )
            if getattr(config.worker, "shared_ref_student_weights", False):
                print(
                    "RL mode: shared_ref_student_weights=True — ref_log_probs from current actor (no frozen ref copy)."
                )
            if getattr(config.algorithm, "use_frozen_base_kl", False) and float(
                getattr(config.algorithm, "frozen_base_kl_coef", 0.0) or 0.0
            ) > 0:
                print(
                    "RL mode: use_frozen_base_kl — extra KL(π‖π_ref_frozen) on Student trajectory "
                    "(frozen ref FSDP; in addition to privileged Teacher ref_log_probs)."
                )
            if getattr(config.algorithm, "opsd_privileged_kl_only_no_grpo", False):
                print(
                    "RL mode: opsd_privileged_kl_only_no_grpo=True — no GRPO policy gradient; "
                    "privileged KL on all sequences (OPSD-style)."
                )
                if getattr(config.worker.actor.model, "use_lora", False):
                    print(
                        "RL mode: OPSD LoRA fixed teacher — Student=Base+LoRA (train LoRA only); "
                        "Teacher ref=disable_adapter (Base frozen at step 0)."
                    )
                elif getattr(config.worker.actor, "opsd_fixed_teacher", False):
                    print(
                        "RL mode: OPSD full fine-tune — Student=trainable actor; "
                        "Teacher=frozen ref FSDP (initial weights, privileged 3D input)."
                    )
                _opsd_clip = getattr(config.algorithm, "opsd_jsd_token_clip", 0.05)
                print(
                    f"RL mode: OPSD loss — full-vocab KL(π_T‖π_S), per-token clip={_opsd_clip}, "
                    "mean over response (kl_coef=1, no batch kl_clip)."
                )
            if getattr(config.algorithm, "privileged_kl_full_vocab", False):
                print(
                    "RL mode: privileged_kl_full_vocab=True — privileged KL uses full-vocabulary "
                    "KL(teacher||student) per response token (update step dual forward)."
                )
        else:
            if config.algorithm.disable_kl:
                print(
                    "RL mode: GRPO without ref (privileged_rl=false, disable_kl=true; no KL term)."
                )
            elif config.algorithm.use_kl_loss:
                print(
                    "RL mode: standard GRPO + β·KL(π‖π_ref) in actor loss (privileged_rl=false; ref=student trajectory)."
                )
            else:
                print(
                    "RL mode: GRPO + KL penalty in reward (privileged_rl=false, use_kl_loss=false; apply_kl_penalty)."
                )

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
            and not getattr(config.algorithm, "opsd_privileged_kl_only_no_grpo", False)
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)
    
    def _log_val_samples_to_disk(self, inputs: List[str], outputs: List[str], scores: List[float], ground_truths: List[Any]) -> str:
        """
        Persist all validation samples (prompt, output, reward) for offline inspection.
        Returns the saved file path.
        """
        # Choose a stable base dir: reuse the checkpoint path for convenience
        base_dir = Path(self.config.trainer.save_checkpoint_path) 
        out_dir = base_dir / "val_samples" 
        out_dir.mkdir(parents=True, exist_ok=True)

        # One file per validation call, keyed by global_step
        filename = out_dir / f"step{self.global_step}.jsonl"

        with filename.open("w", encoding="utf-8") as f:
            for prompt, output, reward, ground_truth in zip(inputs, outputs, scores, ground_truths):
                rec = {
                    "step": self.global_step,
                    "prompt": prompt,
                    "output": output,
                    "reward": float(reward),
                    "ground_truth": ground_truth,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(filename)

    _gpd_reward_module = None  # lazy cache: module or False if unavailable

    @classmethod
    def _get_gpd_reward_module(cls):
        if cls._gpd_reward_module is False:
            return None
        if cls._gpd_reward_module is not None:
            return cls._gpd_reward_module
        root = Path(__file__).resolve().parents[2]
        path = root / "examples" / "reward_function" / "gpd.py"
        if not path.is_file():
            cls._gpd_reward_module = False
            return None
        try:
            spec = importlib.util.spec_from_file_location("gpd_reward_train_dbg", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            cls._gpd_reward_module = mod
            return mod
        except Exception:
            cls._gpd_reward_module = False
            return None

    def _print_train_rollout_samples(self, batch: DataProto) -> None:
        """Print decoded train-batch samples to stdout (for debugging format / reward)."""
        n_log = int(getattr(self.config.trainer, "train_rollout_samples_to_log", 0) or 0)
        if n_log <= 0:
            return
        if batch.batch is None or "prompts" not in batch.batch or "responses" not in batch.batch:
            return
        if "token_level_scores" not in batch.batch:
            return

        bs = batch.batch["prompts"].shape[0]
        k = min(n_log, bs)
        rng = np.random.RandomState(42 + int(self.global_step))
        indices = sorted(int(i) for i in rng.choice(bs, size=k, replace=False).tolist())

        prompts_ids = batch.batch["prompts"]
        resp_ids = batch.batch["responses"]
        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()

        gt_raw = None
        uid_raw = None
        if batch.non_tensor_batch is not None:
            if "ground_truth" in batch.non_tensor_batch:
                gt_raw = batch.non_tensor_batch["ground_truth"]
            if "uid" in batch.non_tensor_batch:
                uid_raw = batch.non_tensor_batch["uid"]

        def _norm_gt(g: Any) -> str:
            if g is None:
                return ""
            if isinstance(g, dict):
                v = g.get("ground_truth", g)
                return str(v) if not isinstance(v, dict) else str(v)
            return str(g)

        gpd_reward = self._get_gpd_reward_module()

        print(f"\n========== train_rollout_samples step={self.global_step} ({len(indices)}/{bs}) ==========")
        for j, i in enumerate(indices):
            p_ids = prompts_ids[i]
            r_ids = resp_ids[i]
            if torch.is_tensor(p_ids):
                p_ids = p_ids.cpu()
            if torch.is_tensor(r_ids):
                r_ids = r_ids.cpu()
            inp_text = self.tokenizer.decode(p_ids, skip_special_tokens=True)
            out_text = self.tokenizer.decode(r_ids, skip_special_tokens=True)
            gt_s = _norm_gt(gt_raw[i]) if gt_raw is not None else ""

            print(f"--- sample {j + 1}/{len(indices)}  batch_index={i} ---")
            print(f"[prompt]\n{inp_text}\n")
            print(f"[output]\n{out_text}\n")
            if gt_raw is not None:
                print(f"[ground_truth] {gt_s}\n")
            if uid_raw is not None:
                u = uid_raw[i]
                print(f"[uid] {u if isinstance(u, str) else str(u)}\n")
            print(f"[reward token_level sum] {float(scores[i])}")
            if gpd_reward is not None:
                try:
                    fmt = float(gpd_reward.format_reward(out_text))
                    acc = float(gpd_reward.accuracy_reward(out_text, gt_s))
                    ex = gpd_reward.extract_final_answer(out_text)
                    print(
                        f"[gpd_debug] format={fmt} accuracy={acc} extracted_answer={ex!r} "
                        f"(1.0 need <redacted_thinking>...</> + <answer>X</answer>)"
                    )
                except Exception as e:
                    print(f"[gpd_debug] unavailable: {e}")
            print()
        print(f"========== end train_rollout_samples step={self.global_step} ==========\n")

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)

        # NEW: best-of-group per uid (store only reward-related info)
        # best_reward_by_uid = {}         # uid -> best reward (float)

        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # test_batch.non_tensor_batch["uid"] = np.array(
            #     [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            # )
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

            # # NEW: best-of-group per uid
            # uids = test_batch.non_tensor_batch["uid"]
            # if torch.is_tensor(uids):
            #     uids = uids.detach().cpu().tolist()
            # elif hasattr(uids, "tolist"):
            #     uids = uids.tolist()
            
            # for uid, reward in zip(uids, scores):
            #     if uid not in best_reward_by_uid or reward > best_reward_by_uid[uid]:
            #         best_reward_by_uid[uid] = reward
            

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        if self.global_step % (self.config.trainer.val_freq * 2) == 0:
            self._log_val_samples_to_disk(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, ground_truths=sample_labels)

        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        # val_reward_metrics["val/best_of_group_reward"] = np.mean(list(best_reward_by_uid.values()))
        print(val_reward_metrics)
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            # Align sequence lengths before concat: different mini-batches may have
            # different max seq_len (due to variable-length image token counts), so
            # we right-pad the shorter one to match the longer one before torch.cat.
            if batch is not None:
                batch, new_batch = _align_batch_seqlen(
                    batch, new_batch,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
                )
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward (OPSD pure distillation: skip reward computation, advantages set to zero later)
                _opsd_kl_only = getattr(self.config.algorithm, "opsd_privileged_kl_only_no_grpo", False)
                reward_ref = None
                if not _opsd_kl_only and "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs (not needed for OPSD pure KL; student/teacher logits are recomputed at update)
                if not _opsd_kl_only:
                    _log_kl_ent = bool(getattr(self.config.trainer, "log_privileged_kl_entropy", False))
                    if _log_kl_ent:
                        batch.meta_info["compute_student_entropy_norm"] = True
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                # compute ref_log_probs / teacher sequence (OPSD only needs teacher_input_ids; ref forward done at update)
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        if bool(getattr(self.config.trainer, "log_privileged_kl_entropy", False)):
                            batch.meta_info["compute_ref_teacher_entropy_norm"] = True
                        # Sequence widths for tuning worker.rollout.max_model_len (logged to experiment_log.jsonl)
                        metrics["seq/student_input_ids_width"] = int(batch.batch["input_ids"].shape[1])

                        if self.config.algorithm.privileged_rl:
                            _has_priv = (
                                batch.non_tensor_batch is not None
                                and "priv_context_text" in batch.non_tensor_batch
                                and any(t for t in batch.non_tensor_batch["priv_context_text"])
                            )
                            if _has_priv:
                                priv_texts = batch.non_tensor_batch["priv_context_text"]
                                priv_lens = [
                                    len(self.tokenizer.encode(t or "", add_special_tokens=False))
                                    for t in priv_texts
                                ]
                                metrics["seq/teacher_priv_token_max_batch"] = int(max(priv_lens))
                                metrics["seq/teacher_priv_token_mean_batch"] = float(sum(priv_lens) / len(priv_lens))

                            if self.config.data.use_vsi_zeroshot_prompts:
                                if self.processor is None:
                                    raise ValueError(
                                        "data.use_vsi_zeroshot_prompts requires a processor (multimodal)."
                                    )
                                ref_batch = build_teacher_batch(
                                    batch,
                                    self.tokenizer,
                                    processor=self.processor,
                                    min_pixels=self.config.data.min_pixels,
                                    max_pixels=self.config.data.max_pixels,
                                    use_vsi_zeroshot_prompts=True,
                                    teacher_prompt_answer_only=bool(
                                        getattr(self.config.data, "teacher_prompt_answer_only", False)
                                    ),
                                    image_dir=self.config.data.image_dir,
                                )
                            elif _has_priv:
                                ref_batch = build_teacher_batch(batch, self.tokenizer)
                            else:
                                ref_batch = batch
                        else:
                            # GRPO-only baseline: ref on student trajectory (no teacher / 3D-priv forward)
                            ref_batch = batch
                        metrics["seq/teacher_ref_input_ids_width"] = int(ref_batch.batch["input_ids"].shape[1])
                        metrics["seq/teacher_ref_attn_sum_max"] = int(
                            ref_batch.batch["attention_mask"].sum(dim=-1).max().item()
                        )

                        if not _opsd_kl_only:
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(ref_batch)
                            batch = batch.union(ref_log_probs)

                        if getattr(self.config.algorithm, "privileged_kl_full_vocab", False) or _opsd_kl_only:
                            batch = batch.union(
                                DataProto.from_dict(
                                    tensors={
                                        "teacher_input_ids": ref_batch.batch["input_ids"],
                                        "teacher_attention_mask": ref_batch.batch["attention_mask"],
                                        "teacher_position_ids": ref_batch.batch["position_ids"],
                                    }
                                )
                            )

                        # Frozen base KL: independent ref FSDP computes log probs on the Student trajectory
                        # (separate from ref_log_probs computed on the privileged Teacher sequence)
                        _fb_kl = bool(getattr(self.config.algorithm, "use_frozen_base_kl", False))
                        _fb_coef = float(getattr(self.config.algorithm, "frozen_base_kl_coef", 0.0) or 0.0)
                        _need_frozen_base = _fb_kl and _fb_coef > 0.0
                        if _need_frozen_base:
                            _has_priv_text = (
                                batch.non_tensor_batch is not None
                                and "priv_context_text" in batch.non_tensor_batch
                                and any(t for t in batch.non_tensor_batch["priv_context_text"])
                            )
                            teacher_ref_used_for_kl = self.config.data.use_vsi_zeroshot_prompts or _has_priv_text
                            if not teacher_ref_used_for_kl:
                                raise ValueError(
                                    "algorithm.use_frozen_base_kl requires Teacher and Student inputs to differ: "
                                    "enable data.use_vsi_zeroshot_prompts or provide non-empty priv_context_text "
                                    "in the samples."
                                )
                            with timer("ref_frozen_base", timing_raw):
                                if getattr(self.config.worker, "shared_ref_student_weights", False):
                                    frozen_base_lp = self.actor_rollout_ref_wg.compute_frozen_base_ref_log_probs(
                                        batch
                                    )
                                else:
                                    frozen_base_lp = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(
                                DataProto.from_dict(
                                    tensors={
                                        "frozen_base_ref_log_probs": frozen_base_lp.batch["ref_log_probs"],
                                    }
                                )
                            )

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if _opsd_kl_only:
                        rm = batch.batch["response_mask"]
                        z = torch.zeros_like(rm, dtype=torch.float32)
                        batch.batch["token_level_scores"] = z
                        batch.batch["token_level_rewards"] = z
                        batch.batch["advantages"] = z
                        batch.batch["returns"] = z
                        metrics["reward/opsd_skipped"] = 1.0
                    elif "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    if not _opsd_kl_only:
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            weighted_posneg_base=self.config.algorithm.weighted_posneg_base,
                            weighted_posneg_coef=self.config.algorithm.weighted_posneg_coef,
                        )

                    self._print_train_rollout_samples(batch)

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            _log_kl_ent = bool(getattr(self.config.trainer, "log_privileged_kl_entropy", False))
            _log_every = max(
                1, int(getattr(self.config.trainer, "log_privileged_kl_entropy_every_n_steps", 1) or 1)
            )
            if (
                _log_kl_ent
                and self.global_step % _log_every == 0
                and not getattr(self.config.algorithm, "opsd_privileged_kl_only_no_grpo", False)
                and self.use_reference_policy
                and "ref_log_probs" in batch.batch
                and "old_log_probs" in batch.batch
            ):
                _thr = float(getattr(self.config.algorithm, "opd_correct_reward_threshold", 0.5) or 0.5)
                _inc_only = bool(
                    getattr(self.config.trainer, "log_privileged_kl_entropy_incorrect_only", True)
                )
                metrics.update(
                    compute_privileged_kl_entropy_metrics(
                        batch,
                        correctness_threshold=_thr,
                        incorrect_only=_inc_only,
                    )
                )
                _max_dump = int(
                    getattr(self.config.trainer, "log_privileged_kl_entropy_max_samples_per_step", 64) or 64
                )
                _n_dump = dump_incorrect_privileged_kl_entropy(
                    batch,
                    global_step=self.global_step,
                    save_checkpoint_path=self.config.trainer.save_checkpoint_path,
                    tokenizer=self.tokenizer,
                    model_path=self.config.worker.actor.model.model_path,
                    correctness_threshold=_thr,
                    max_samples_per_step=_max_dump,
                    teacher_answer_only=bool(
                        getattr(self.config.data, "teacher_prompt_answer_only", False)
                    ),
                    temperature=float(self.config.worker.rollout.temperature),
                    incorrect_only=_inc_only,
                )
                if _n_dump > 0:
                    metrics["kl_entropy/dumped_incorrect_samples"] = float(_n_dump)
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
