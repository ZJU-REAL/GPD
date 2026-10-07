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

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager, BatchFunctionRewardGroundingManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        elif config.worker.reward.reward_type == "batch_grounding":
            RewardManager = BatchFunctionRewardGroundingManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        import os
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            # With multiple Ray workers on the same node + NCCL, RAS binds the same local port by default,
            # which easily triggers "Address already in use"; disabled by default, override via environment variable.
            "NCCL_RAS_ENABLE": os.environ.get("NCCL_RAS_ENABLE", "0"),
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        }
        # Propagate CUDA_VISIBLE_DEVICES to all Ray workers so they respect the
        # GPU selection from the launch script (Ray workers don't inherit shell env).
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        # Ray workers may not have nvcc in their default PATH; vLLM + FlashInfer JIT invokes nvcc.
        if "PATH" in os.environ:
            env_vars["PATH"] = os.environ["PATH"]
        if "LD_LIBRARY_PATH" in os.environ:
            env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        # System CUDA headers (e.g. curand.h under /usr/local/cuda); FlashInfer JIT needs them in Ray workers.
        for _cuda_key in ("CUDA_HOME", "CUDA_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"):
            if _cuda_key in os.environ:
                env_vars[_cuda_key] = os.environ[_cuda_key]
        if "NCCL_DEBUG_SUBSYS" in os.environ:
            env_vars["NCCL_DEBUG_SUBSYS"] = os.environ["NCCL_DEBUG_SUBSYS"]
        runtime_env = {"env_vars": env_vars}
        ray.init(runtime_env=runtime_env)

    # Ray registers GPUs via NVML; if nvidia-smi reports a Driver/library version mismatch,
    # cluster_resources may show 0 GPUs even though torch.cuda.device_count() > 0,
    # causing the GPU check in init_workers to fail.
    try:
        import torch

        _cr = ray.cluster_resources()
        if float(_cr.get("GPU", 0) or 0) == 0 and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(
                "[verl][WARN] Ray cluster_resources reports no GPUs, but torch.cuda.device_count()="
                f"{torch.cuda.device_count()}. Fix the local NVML first (must match nvidia-smi): "
                "common cause is a mismatch between the kernel driver and the user-space libnvidia-ml "
                "(driver upgraded without reboot, or conda/LD_LIBRARY_PATH overrides an incompatible "
                "libnvidia-ml). Try: `LD_LIBRARY_PATH= nvidia-smi`; until fixed, Ray cannot register "
                "GPUs for training."
            )
    except Exception:
        pass

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
