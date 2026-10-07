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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    use_vsi_zeroshot_prompts: bool = False
    """True: Student=vsi_zeroshot_eval baseline, Teacher=vsi_text_cot_eval + priv_context_text
    (requires parquet with conversations or a parseable prompt field)"""
    teacher_prompt_answer_only: bool = False
    """True: Teacher system prompt omits 3D references; uses only video frames + ground-truth answer block
    (paired with ``prepare_data --priv_context_mode answer``).
    Only takes effect when ``use_vsi_zeroshot_prompts=true``; disabled by default."""

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None

        if self.teacher_prompt_answer_only and not self.use_vsi_zeroshot_prompts:
            print(
                "[config] data.teacher_prompt_answer_only=True but use_vsi_zeroshot_prompts=False: "
                "Teacher system switch only takes effect on the zeroshot Teacher path; "
                "non-zeroshot still goes through priv_insert."
            )
        if self.teacher_prompt_answer_only and self.use_vsi_zeroshot_prompts:
            print(
                "[config] data.teacher_prompt_answer_only=True — Teacher system uses prompt text "
                "without 3D references (verl/utils/vsi_zeroshot_prompts.TEACHER_SYSTEM_ANSWER_ONLY)."
            )


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    privileged_rl: bool = True
    """True: asymmetric privileged — Teacher/ref uses 3D privileged sequences; KL is the privileged KL (may differ
    from standard). False: standard GRPO — ref shares the Student trajectory; if use_kl_loss is set, standard
    beta*KL is added to the actor loss."""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """True: KL(pi||pi_ref) added to actor loss (not deducted from reward). When False and disable_kl=False,
    applies KL penalty inside the reward (apply_kl_penalty)."""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    kl_clip: Optional[float] = None
    """clip privileged KL loss before multiplying kl_coef; None means no clip"""
    kl_clip_symmetric: bool = False
    """True: clip low_var_kl to [-kl_clip, +kl_clip] before entering the loss;
    False (default) clips only the upper bound max=kl_clip"""
    privileged_kl_only_on_correct: bool = True
    """Whether privileged KL applies only to correct trajectories; when False, incorrect trajectories
    also receive the KL penalty (differs from BiPS 'high-reward only')."""
    privileged_kl_only_on_incorrect: bool = False
    """Whether privileged KL applies only to incorrect trajectories; when True, mutually exclusive with
    privileged_kl_only_on_correct (this flag takes priority). Disabled by default."""
    privileged_kl_adaptive_incorrect_low_var: bool = False
    """True: on incorrect trajectories under privileged KL, apply per-token mask weighting scaled by the
    mean kld of each incorrect sequence's response (consistent with algorithm.kl_penalty, typically low_var_kl).
    Default behavior linearly scales in [tau, cap] to full weight; if privileged_kl_incorrect_low_var_step_mode=true,
    sequences with mean > tau get full weight, others get 0. Requires privileged_kl_only_on_incorrect=true
    (or split_grpo_pg_opd_kl which auto-enables it)."""
    privileged_kl_incorrect_low_var_tau: float = 0.10
    """Adaptive privileged KL: if the mean kld of an incorrect sequence is below this threshold, the privileged KL
    weight for that sequence is 0 (GRPO side unaffected). Reference from analysis: post-training incorrect
    low_var_kl ~0.07-0.074, baseline incorrect ~0.335."""
    privileged_kl_incorrect_low_var_cap: float = 0.28
    """Linear mode: when the sequence mean reaches this value or above, privileged KL weight is 1; linearly
    interpolated between tau and cap. Requires cap > tau in linear mode. In step mode
    (privileged_kl_incorrect_low_var_step_mode=true) this field is not used for scaling and can be ignored."""
    privileged_kl_incorrect_low_var_step_mode: bool = False
    """True: privileged KL weight is 1 when the incorrect sequence kld mean is strictly greater than tau,
    otherwise 0 (no linear segment; cap is not used)."""
    privileged_kl_adaptive_incorrect_scale_min: float = 0.0
    """Together with scale_max: when style=mask, mapped to the mask multiplier; when style=kl_coef, defines
    the coefficient range [min, max] for kl_loss at this step."""
    privileged_kl_adaptive_incorrect_scale_max: float = 1.0
    """See scale_min. In kl_coef mode, algorithm.kl_coef is no longer used for this term;
    gap linearly interpolates within this range."""
    privileged_kl_adaptive_incorrect_style: str = "mask"
    """mask: scale the incorrect token mask by the sequence gap (original behavior). kl_coef: mask is 0/1 only;
    gap determines eff_kl_coef in [scale_min, max] multiplied by kl_loss for this micro-batch."""
    split_grpo_pg_opd_kl: bool = False
    """True: correct trajectories use GRPO only (policy gradient / clip); incorrect trajectories skip PG and use
    privileged KL only (OPD). Requires privileged_rl + use_kl_loss. Enabling this auto-sets
    privileged_kl_only_on_incorrect=true and privileged_kl_only_on_correct=false."""
    opsd_privileged_kl_only_no_grpo: bool = False
    """True: no GRPO policy gradient for any trajectory; optimize only privileged KL (OPSD-style sequence
    distillation); auto-sets privileged KL mask to full response. Auto-enables actor LoRA + fixed teacher
    (Teacher uses disable_adapter/Base at step-0, Student updates LoRA only). Loss aligned with official OPSD:
    full-vocab KL(pi_T||pi_S), per-token jsd_token_clip, kl_coef=1 (loss equals distillation mean).
    Requires privileged_rl + use_kl_loss; mutually exclusive with split_grpo_pg_opd_kl."""
    opsd_jsd_token_clip: float = 0.05
    """When ``opsd_privileged_kl_only_no_grpo=true``, per-token KL upper clip bound
    (corresponds to official ``--jsd_token_clip``); <=0 disables clipping."""
    opsd_rollout_top_k: int = 20
    """Rollout sampling top_k when ``opsd_privileged_kl_only_no_grpo=true`` (official ``--top_k 20``);
    <=0 means no restriction in vLLM."""
    shared_ref_student_weights: bool = False
    """True: ref/Teacher shares the same trainable weights as the student (compute_ref_log_probs uses actor FSDP,
    no second ref copy); consistent with OPSD dynamic mode."""
    use_frozen_base_kl: bool = False
    """True: in addition to the privileged KL (Teacher sequence), add an extra KL(pi||pi_ref_frozen) term on the
    Student trajectory. When shared_ref_student_weights=false, uses the original ref FSDP; when true, an
    additional frozen ref_frozen is loaded exclusively for this term."""
    frozen_base_kl_coef: float = 0.01
    """KL coefficient relative to the frozen base ref (Student input); only takes effect when
    use_frozen_base_kl=true and coefficient > 0."""
    opd_correct_reward_threshold: float = 0.5
    """For split_grpo_pg_opd_kl: a sequence is considered correct if the last-position token_level_rewards >=
    this threshold, in which case it participates in PG; otherwise only KL."""
    entropy_penalty_coef: float = 0.0
    """Coefficient for entropy penalty term mean(-log pi) on the currently sampled tokens; 0 disables it."""
    kl_teacher_confidence_weight_beta: float = 0.0
    """Per-token weight for privileged KL: w ~ exp(beta*ref_logprob), normalized within the batch;
    0 means uniform weighting (logprob proxy for the SRPO teacher entropy idea)."""
    kl_teacher_disconfidence_weight_gamma: float = 0.0
    """Symmetric with beta: when the batch contains token_level_rewards, incorrect sequences are further weighted
    by exp(-gamma*ref_logprob) (factor is 1 for correct sequences); the less the Teacher favors the sampled token,
    the larger its privileged KL weight; gamma=0 disables it."""
    privileged_kl_tip_q1_q3_mask: bool = False
    """TIP Q1 union Q3: privileged KL computed only on 'high-entropy + high-divergence' or
    'low-entropy + high-divergence' tokens (student entropy is full-vocabulary H/lnV, chunked as in OPSD;
    divergence axis uses per-token kld under current kl_penalty as a proxy). Requires privileged_rl.
    Disabled by default."""
    privileged_kl_tip_entropy_chunk_size: int = 256
    """Chunk size (in rows) for computing student entropy row-by-row when privileged_kl_tip_q1_q3_mask is true."""
    privileged_kl_teacher_entropy_reweight: bool = False
    """True: mimics TIP — on incorrect trajectories at valid positions of the current privileged kl_mask, computes
    the median of Teacher full-vocabulary entropy H/lnV; h<=median keeps privileged KL, h>median disables it;
    correct sequences are not gated. No gating when fewer than 2 valid positions.
    Requires privileged_rl + use_kl_loss. Disabled by default."""
    privileged_kl_sdar_sigmoid_gate: bool = False
    """SDAR confidence-gated distillation (see SDAR ``verl/trainer/ppo/sdar_utils.py``): when true,
    replaces the privileged KL scalar with ``compute_sdar_loss`` (``agg(g*(log pi_T - log pi_S))``,
    ``g=sigmoid(beta*Delta)``); the ``kld`` from ``compute_kl`` is no longer used for this term.
    Requires ``algorithm.privileged_rl=true`` (``worker.actor.privileged_kl`` derived from it);
    mutually exclusive with ``kl_teacher_confidence_weight_beta`` / ``kl_teacher_disconfidence_weight_gamma``.
    Disabled by default."""
    privileged_kl_sdar_gate_beta: float = 5.0
    """beta for ``privileged_kl_sdar_sigmoid_gate=true``; larger values produce a steeper gate
    (consistent with SDAR default magnitude)."""
    privileged_kl_full_vocab: bool = False
    """True: privileged KL computes KL(pi_T||pi_S) over the full vocabulary distribution at each response step
    (OPSD-style); False (default): uses only the sampled token log pi for low_var_kl etc."""
    privileged_kl_full_vocab_chunk_size: int = 256
    """Softmax chunk size (in rows) when ``privileged_kl_full_vocab=true``;
    defaults to the same as TIP entropy chunking."""
    weighted_posneg_base: float = 0.5
    """base term for negative score weight in grpo_weighted_positive_negative: clamp(base + coef * mean, 0, 1)"""
    weighted_posneg_coef: float = 1.5
    """coefficient for negative score weight in grpo_weighted_positive_negative: clamp(base + coef * mean, 0, 1)"""
    priv_kl_short_threshold: int = 0
    """Length-aware privileged KL: sequences with response length < this threshold (in tokens) have their
    privileged KL zeroed out entirely; used to block the 'short answer = low KL shortcut' that causes length
    collapse. <=0 disables this (default). Compatible with split_grpo_pg_opd_kl / OPSD /
    adaptive_incorrect_low_var; only active on the use_kl_loss + privileged_rl path."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    train_rollout_samples_to_log: int = 0
    """each training step, print this many decoded (prompt, output, reward) to stdout; 0 disables"""
    log_privileged_kl_entropy: bool = False
    """True: write incorrect trajectories to disk at
    ``privileged_kl_entropy/step_*/sample_*/logits_summary.npz`` (compatible with offline export);
    also writes kl_entropy/* scalars to the experiment log (incorrect trajectory tokens only).
    Disabled by default."""
    log_privileged_kl_entropy_every_n_steps: int = 1
    """When log_privileged_kl_entropy=true, logging is performed only when global_step is a multiple
    of this value."""
    log_privileged_kl_entropy_max_samples_per_step: int = 64
    """Maximum number of incorrect trajectories to write to disk per step
    (deduplicated by uid, at most 1 per prompt)."""
    log_privileged_kl_entropy_incorrect_only: bool = True
    """True: only incorrect trajectories (consistent with privileged_kl_only_on_incorrect;
    threshold is opd_correct_reward_threshold)."""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        if self.algorithm.use_kl_loss and self.algorithm.disable_kl:
            raise ValueError(
                "algorithm.use_kl_loss=true requires a reference policy (set algorithm.disable_kl=false)."
            )
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.privileged_kl = self.algorithm.privileged_rl
        # privileged KL may use kl_clip; standard beta*KL generally does not clip the loss (consistent with TRL etc.)
        if self.algorithm.privileged_rl and getattr(self.algorithm, "kl_clip", None) is not None:
            self.worker.actor.kl_clip = self.algorithm.kl_clip
            self.worker.actor.kl_clip_symmetric = bool(
                getattr(self.algorithm, "kl_clip_symmetric", False)
            )
        else:
            self.worker.actor.kl_clip = None
            self.worker.actor.kl_clip_symmetric = False
        opsd_kl_only = getattr(self.algorithm, "opsd_privileged_kl_only_no_grpo", False)
        split_pg = getattr(self.algorithm, "split_grpo_pg_opd_kl", False)
        if opsd_kl_only and split_pg:
            raise ValueError(
                "algorithm.opsd_privileged_kl_only_no_grpo=true and algorithm.split_grpo_pg_opd_kl=true "
                "are mutually exclusive."
            )
        if opsd_kl_only and (not self.algorithm.privileged_rl or not self.algorithm.use_kl_loss):
            raise ValueError(
                "algorithm.opsd_privileged_kl_only_no_grpo=true requires algorithm.privileged_rl=true "
                "and algorithm.use_kl_loss=true."
            )
        self.worker.actor.opsd_privileged_kl_only_no_grpo = opsd_kl_only
        self.worker.actor.split_grpo_pg_opd_kl = split_pg
        self.worker.actor.opd_correct_reward_threshold = float(
            getattr(self.algorithm, "opd_correct_reward_threshold", 0.5) or 0.5
        )
        if opsd_kl_only:
            self.worker.actor.privileged_kl_only_on_correct = False
            self.worker.actor.privileged_kl_only_on_incorrect = False
            # Official OPSD fixed teacher: Student full fine-tuning; Teacher = independent ref FSDP
            # (step-0 weights, not updated)
            self.worker.actor.model.use_lora = False
            self.worker.actor.opsd_fixed_teacher = True
            if self.worker.shared_ref_student_weights or getattr(
                self.algorithm, "shared_ref_student_weights", False
            ):
                print(
                    "[config] opsd_privileged_kl_only_no_grpo=True — forcing shared_ref_student_weights=false "
                    "(Teacher uses independent frozen ref; Student trains with full fine-tuning)."
                )
            self.algorithm.shared_ref_student_weights = False
            self.worker.shared_ref_student_weights = False
            if getattr(self.algorithm, "use_frozen_base_kl", False):
                print(
                    "[config] opsd_privileged_kl_only_no_grpo=True — disabling use_frozen_base_kl "
                    "(fixed teacher already provided by independent ref; no extra frozen_base KL needed)."
                )
            self.algorithm.use_frozen_base_kl = False
            self.worker.actor.use_frozen_base_kl = False
            self.worker.actor.frozen_base_kl_coef = 0.0
            self.algorithm.privileged_kl_full_vocab = True
            self.worker.actor.privileged_kl_full_vocab = True
            _opsd_clip = float(getattr(self.algorithm, "opsd_jsd_token_clip", 0.05) or 0.05)
            self.worker.actor.opsd_jsd_token_clip = _opsd_clip
            self.worker.actor.kl_clip = None
            self.worker.actor.kl_clip_symmetric = False
            if float(getattr(self.algorithm, "kl_coef", 1.0) or 1.0) != 1.0:
                print(
                    "[config] opsd_privileged_kl_only_no_grpo=True — ignoring algorithm.kl_coef; "
                    "loss is the mean of per-token clipped KL (same as official JSD main loss, coefficient=1)."
                )
            self.algorithm.kl_coef = 1.0
            self.worker.actor.kl_coef = 1.0
            if int(getattr(self.worker.rollout, "n", 1) or 1) != 1:
                print(
                    "[config] opsd_privileged_kl_only_no_grpo=True — forcing worker.rollout.n=1 "
                    "(one on-policy trajectory per prompt, aligned with official OPSD)."
                )
            self.worker.rollout.n = 1
            self.worker.rollout.top_k = int(getattr(self.algorithm, "opsd_rollout_top_k", 20) or 20)
            for _flag, _name in (
                (getattr(self.algorithm, "privileged_kl_sdar_sigmoid_gate", False), "privileged_kl_sdar_sigmoid_gate"),
                (getattr(self.algorithm, "privileged_kl_tip_q1_q3_mask", False), "privileged_kl_tip_q1_q3_mask"),
                (
                    getattr(self.algorithm, "privileged_kl_adaptive_incorrect_low_var", False),
                    "privileged_kl_adaptive_incorrect_low_var",
                ),
                (float(getattr(self.algorithm, "kl_teacher_confidence_weight_beta", 0) or 0) > 0, "kl_teacher_confidence_weight_beta"),
                (float(getattr(self.algorithm, "kl_teacher_disconfidence_weight_gamma", 0) or 0) > 0, "kl_teacher_disconfidence_weight_gamma"),
                (float(getattr(self.algorithm, "entropy_penalty_coef", 0) or 0) > 0, "entropy_penalty_coef"),
            ):
                if _flag:
                    raise ValueError(
                        f"algorithm.opsd_privileged_kl_only_no_grpo=true is incompatible with "
                        f"algorithm.{_name} (official OPSD uses pure distillation loss)."
                    )
            print(
                "[config] OPSD fixed teacher: Student full fine-tuning; "
                "Teacher = independent ref FSDP (initial weights, not updated)."
            )
            print(
                f"[config] OPSD loss: full-vocab KL(pi_T||pi_S) + per-token clip={_opsd_clip} + mean "
                "(no batch kl_clip / kl_coef)."
            )
            print(
                f"[config] OPSD rollout: n=1, top_k={self.worker.rollout.top_k}; "
                "training step skips reward/advantage (pure distillation)."
            )
        elif split_pg:
            self.worker.actor.privileged_kl_only_on_correct = False
            self.worker.actor.privileged_kl_only_on_incorrect = True
        else:
            self.worker.actor.privileged_kl_only_on_correct = getattr(
                self.algorithm, "privileged_kl_only_on_correct", True
            )
            self.worker.actor.privileged_kl_only_on_incorrect = getattr(
                self.algorithm, "privileged_kl_only_on_incorrect", False
            )
        if split_pg and not opsd_kl_only:
            if not self.algorithm.privileged_rl or not self.algorithm.use_kl_loss:
                print(
                    "[config] split_grpo_pg_opd_kl=true is recommended together with privileged_rl=true "
                    "and use_kl_loss=true; otherwise incorrect trajectories may have no KL gradient."
                )
        self.worker.actor.entropy_penalty_coef = getattr(self.algorithm, "entropy_penalty_coef", 0.0)
        self.worker.shared_ref_student_weights = getattr(self.algorithm, "shared_ref_student_weights", False)
        self.worker.actor.kl_teacher_confidence_weight_beta = getattr(
            self.algorithm, "kl_teacher_confidence_weight_beta", 0.0
        )
        self.worker.actor.kl_teacher_disconfidence_weight_gamma = getattr(
            self.algorithm, "kl_teacher_disconfidence_weight_gamma", 0.0
        )
        self.worker.actor.privileged_kl_tip_q1_q3_mask = bool(
            getattr(self.algorithm, "privileged_kl_tip_q1_q3_mask", False)
        )
        self.worker.actor.privileged_kl_tip_entropy_chunk_size = int(
            getattr(self.algorithm, "privileged_kl_tip_entropy_chunk_size", 256) or 256
        )
        _tre = bool(getattr(self.algorithm, "privileged_kl_teacher_entropy_reweight", False))
        self.worker.actor.privileged_kl_teacher_entropy_reweight = _tre
        if _tre:
            if not self.algorithm.use_kl_loss or self.algorithm.disable_kl:
                raise ValueError(
                    "algorithm.privileged_kl_teacher_entropy_reweight=true requires "
                    "algorithm.use_kl_loss=true and a reference policy (algorithm.disable_kl=false)."
                )
            if not self.algorithm.privileged_rl:
                raise ValueError(
                    "algorithm.privileged_kl_teacher_entropy_reweight=true requires "
                    "algorithm.privileged_rl=true (Teacher entropy is computed on the Teacher sequence)."
                )
            print(
                "[config] privileged_kl_teacher_entropy_reweight=True — "
                "on incorrect trajectories within the current kl_mask, compute median of Teacher H/lnV: "
                "h<=median keeps privileged KL, h>median disables it; "
                "skips gating when fewer than 2 valid positions; "
                "ref forward additionally computes full-vocabulary entropy."
            )
        self.worker.actor.privileged_kl_sdar_sigmoid_gate = bool(
            getattr(self.algorithm, "privileged_kl_sdar_sigmoid_gate", False)
        )
        self.worker.actor.privileged_kl_sdar_gate_beta = float(
            getattr(self.algorithm, "privileged_kl_sdar_gate_beta", 5.0) or 5.0
        )
        if self.worker.actor.privileged_kl_sdar_sigmoid_gate:
            if not self.algorithm.use_kl_loss or self.algorithm.disable_kl:
                raise ValueError(
                    "algorithm.privileged_kl_sdar_sigmoid_gate=true requires "
                    "algorithm.use_kl_loss=true and a reference policy (algorithm.disable_kl=false)."
                )
            if not self.algorithm.privileged_rl:
                raise ValueError(
                    "algorithm.privileged_kl_sdar_sigmoid_gate=true requires "
                    "algorithm.privileged_rl=true (worker.actor.privileged_kl is derived from privileged_rl; "
                    "this flag only replaces the privileged KL path)."
                )
            b_srpo = float(getattr(self.algorithm, "kl_teacher_confidence_weight_beta", 0.0) or 0.0)
            g_srpo = float(getattr(self.algorithm, "kl_teacher_disconfidence_weight_gamma", 0.0) or 0.0)
            if b_srpo > 0.0 or g_srpo > 0.0:
                raise ValueError(
                    "algorithm.privileged_kl_sdar_sigmoid_gate=true is mutually exclusive with SRPO weights "
                    "(kl_teacher_confidence_weight_beta / kl_teacher_disconfidence_weight_gamma > 0); "
                    "please set them to 0."
                )
            print(
                "[config] privileged_kl_sdar_sigmoid_gate=True — "
                "privileged KL replaced by SDAR compute_sdar_loss (agg(g*(log pi_T - log pi_S))), "
                f"beta={self.worker.actor.privileged_kl_sdar_gate_beta}; "
                "aggregation aligned with worker.actor.loss_avg_mode."
            )
        _adapt_inc = bool(getattr(self.algorithm, "privileged_kl_adaptive_incorrect_low_var", False))
        self.worker.actor.privileged_kl_adaptive_incorrect_low_var = _adapt_inc
        self.worker.actor.privileged_kl_incorrect_low_var_tau = float(
            getattr(self.algorithm, "privileged_kl_incorrect_low_var_tau", 0.10) or 0.10
        )
        self.worker.actor.privileged_kl_incorrect_low_var_cap = float(
            getattr(self.algorithm, "privileged_kl_incorrect_low_var_cap", 0.28) or 0.28
        )
        self.worker.actor.privileged_kl_incorrect_low_var_step_mode = bool(
            getattr(self.algorithm, "privileged_kl_incorrect_low_var_step_mode", False)
        )
        _smin = float(getattr(self.algorithm, "privileged_kl_adaptive_incorrect_scale_min", 0.0) or 0.0)
        _smax = float(getattr(self.algorithm, "privileged_kl_adaptive_incorrect_scale_max", 1.0) or 1.0)
        if _smax < _smin:
            raise ValueError(
                "algorithm.privileged_kl_adaptive_incorrect_scale_max must be >= "
                "algorithm.privileged_kl_adaptive_incorrect_scale_min."
            )
        self.worker.actor.privileged_kl_adaptive_incorrect_scale_min = _smin
        self.worker.actor.privileged_kl_adaptive_incorrect_scale_max = _smax
        _pst = str(getattr(self.algorithm, "privileged_kl_adaptive_incorrect_style", "mask") or "mask").lower()
        if _pst not in ("mask", "kl_coef"):
            raise ValueError(
                "algorithm.privileged_kl_adaptive_incorrect_style must be 'mask' or 'kl_coef', "
                f"got {_pst!r}."
            )
        self.worker.actor.privileged_kl_adaptive_incorrect_style = _pst
        if _adapt_inc:
            if not self.worker.actor.privileged_kl_only_on_incorrect:
                raise ValueError(
                    "algorithm.privileged_kl_adaptive_incorrect_low_var=true requires privileged KL to be "
                    "computed on incorrect trajectories only: set algorithm.privileged_kl_only_on_incorrect=true, "
                    "or enable algorithm.split_grpo_pg_opd_kl=true."
                )
            if opsd_kl_only:
                raise ValueError(
                    "algorithm.privileged_kl_adaptive_incorrect_low_var is incompatible with "
                    "opsd_privileged_kl_only_no_grpo."
                )
            _tau = self.worker.actor.privileged_kl_incorrect_low_var_tau
            _cap = self.worker.actor.privileged_kl_incorrect_low_var_cap
            _step = self.worker.actor.privileged_kl_incorrect_low_var_step_mode
            if not _step and _cap <= _tau:
                raise ValueError(
                    "In linear adaptive mode, algorithm.privileged_kl_incorrect_low_var_cap must be > tau; "
                    "to use 'full weight above tau', set algorithm.privileged_kl_incorrect_low_var_step_mode=true."
                )
            _style = self.worker.actor.privileged_kl_adaptive_incorrect_style
            _style_note = (
                "mask: incorrect token mask multiplier; kl_coef: mask is 0/1, privileged KL coefficient "
                f"in [{_smin}, {_smax}] varies with gap (algorithm.kl_coef not used for this term)."
            )
            if _step:
                print(
                    "[config] privileged_kl_adaptive_incorrect_low_var=True — "
                    f"style={_style} ({_style_note}) "
                    f"STEP: kld mean > tau={_tau} -> upper bound (mask=scale_max; kl_coef=upper coef), "
                    "otherwise lower bound."
                )
            else:
                print(
                    "[config] privileged_kl_adaptive_incorrect_low_var=True — "
                    f"style={_style} ({_style_note}) "
                    f"LINEAR: tau={_tau}, cap={_cap}; gap mapped to [{_smin}, {_smax}]."
                )
        self.worker.actor.use_frozen_base_kl = bool(getattr(self.algorithm, "use_frozen_base_kl", False))
        self.worker.actor.frozen_base_kl_coef = float(getattr(self.algorithm, "frozen_base_kl_coef", 0.0) or 0.0)
        _full_vocab = bool(getattr(self.algorithm, "privileged_kl_full_vocab", False))
        self.worker.actor.privileged_kl_full_vocab = _full_vocab
        self.worker.actor.privileged_kl_full_vocab_chunk_size = int(
            getattr(self.algorithm, "privileged_kl_full_vocab_chunk_size", 256) or 256
        )
        if _full_vocab:
            if not self.algorithm.privileged_rl or not self.algorithm.use_kl_loss:
                raise ValueError(
                    "algorithm.privileged_kl_full_vocab=true requires algorithm.privileged_rl=true "
                    "and algorithm.use_kl_loss=true."
                )
            if self.worker.actor.privileged_kl_sdar_sigmoid_gate:
                raise ValueError(
                    "algorithm.privileged_kl_full_vocab=true is mutually exclusive with "
                    "algorithm.privileged_kl_sdar_sigmoid_gate=true."
                )
            print(
                "[config] privileged_kl_full_vocab=True — privileged KL uses full-vocabulary KL(pi_T||pi_S); "
                f"softmax chunk size={self.worker.actor.privileged_kl_full_vocab_chunk_size}; "
                "frozen base KL still uses sampled tokens."
            )
        self.worker.actor.priv_kl_short_threshold = int(
            getattr(self.algorithm, "priv_kl_short_threshold", 0) or 0
        )
        if self.worker.actor.priv_kl_short_threshold > 0:
            print(
                "[config] priv_kl_short_threshold="
                f"{self.worker.actor.priv_kl_short_threshold} — "
                "sequences shorter than this length will have their privileged KL zeroed out "
                "(prevents length collapse)."
            )
        _fb = self.worker.actor.use_frozen_base_kl and self.worker.actor.frozen_base_kl_coef > 0
        if _fb:
            if self.algorithm.disable_kl:
                raise ValueError(
                    "algorithm.use_frozen_base_kl requires a reference policy (algorithm.disable_kl=false)."
                )
            if not self.algorithm.use_kl_loss:
                raise ValueError(
                    "algorithm.use_frozen_base_kl must be used together with algorithm.use_kl_loss=true "
                    "(shares the same loss path as privileged KL)."
                )
            if not self.algorithm.privileged_rl:
                raise ValueError(
                    "algorithm.use_frozen_base_kl adds an extra Student-trajectory frozen KL term alongside "
                    "the privileged Teacher sequence; set privileged_rl=true."
                )
            if self.worker.shared_ref_student_weights:
                print(
                    "[config] use_frozen_base_kl + shared_ref_student_weights=True: "
                    "will load an additional frozen ref FSDP used only for KL(pi||pi_frozen) on the Student "
                    "trajectory; privileged KL still uses current actor weights + Teacher sequence."
                )

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
