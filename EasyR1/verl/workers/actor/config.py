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
Actor config
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    use_lora: bool = False
    """True: attach LoRA to the actor; OPSD mode (opsd_privileged_kl_only_no_grpo) enables this automatically."""
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)

        if isinstance(self.lora_target_modules, list):
            self.lora_target_modules = tuple(self.lora_target_modules)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)
    """True: GPD privileged KL (correct-trajectory only by default, see dp_actor); False: standard β·KL(π‖π_ref) over the entire response"""
    privileged_kl: bool = field(default=False, init=False)
    kl_clip: Optional[float] = field(default=None, init=False)
    kl_clip_symmetric: bool = field(default=False, init=False)
    privileged_kl_only_on_correct: bool = field(default=True, init=False)
    privileged_kl_only_on_incorrect: bool = field(default=False, init=False)
    privileged_kl_adaptive_incorrect_low_var: bool = field(default=False, init=False)
    privileged_kl_incorrect_low_var_tau: float = field(default=0.10, init=False)
    privileged_kl_incorrect_low_var_cap: float = field(default=0.28, init=False)
    privileged_kl_incorrect_low_var_step_mode: bool = field(default=False, init=False)
    privileged_kl_adaptive_incorrect_scale_min: float = field(default=0.0, init=False)
    privileged_kl_adaptive_incorrect_scale_max: float = field(default=1.0, init=False)
    privileged_kl_adaptive_incorrect_style: str = field(default="mask", init=False)
    split_grpo_pg_opd_kl: bool = field(default=False, init=False)
    opsd_privileged_kl_only_no_grpo: bool = field(default=False, init=False)
    opsd_fixed_teacher: bool = field(default=False, init=False)
    """True: Teacher uses step-0 fixed weights (full fine-tune: independent ref FSDP; LoRA: disable_adapter Base frozen). Requires OPSD mode."""
    opsd_jsd_token_clip: float = field(default=0.05, init=False)
    """OPSD mode: per-token KL clip upper bound (official jsd_token_clip); <=0 means no clip."""
    opd_correct_reward_threshold: float = field(default=0.5, init=False)
    entropy_penalty_coef: float = field(default=0.0, init=False)
    kl_teacher_confidence_weight_beta: float = field(default=0.0, init=False)
    kl_teacher_disconfidence_weight_gamma: float = field(default=0.0, init=False)
    # TIP (arXiv:2604.14084) + OPSD-style: privileged KL computed only on Q1∪Q3 tokens (see verl/utils/tip_opd.py)
    privileged_kl_tip_q1_q3_mask: bool = field(default=False, init=False)
    privileged_kl_tip_entropy_chunk_size: int = field(default=256, init=False)
    privileged_kl_teacher_entropy_reweight: bool = field(default=False, init=False)
    privileged_kl_sdar_sigmoid_gate: bool = field(default=False, init=False)
    privileged_kl_sdar_gate_beta: float = field(default=5.0, init=False)
    privileged_kl_full_vocab: bool = field(default=False, init=False)
    privileged_kl_full_vocab_chunk_size: int = field(default=256, init=False)
    use_frozen_base_kl: bool = field(default=False, init=False)
    frozen_base_kl_coef: float = field(default=0.0, init=False)
    priv_kl_short_threshold: int = field(default=0, init=False)
    # action weight and grounding weight
    use_action_weight: bool = field(default=False, init=False)
    pg_loss_action_weight_coef: float = field(default=1.0, init=False)
    pg_loss_grounding_weight_coef: float = field(default=1.0, init=False)
    kl_loss_grounding_weight_coef: float = field(default=1.0, init=False)



@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
