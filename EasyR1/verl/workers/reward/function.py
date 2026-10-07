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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]

BatchRewardFunctionGrounding = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto, step=None) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# class SequentialFunctionRewardManager(FunctionRewardManager):
#     reward_fn: SequentialRewardFunction

#     def compute_reward(self, data: DataProto, step=None) -> Tuple[torch.Tensor, dict[str, list[float]]]:
#         responses = data.batch["responses"]
#         response_mask = data.batch["response_mask"]

#         reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
#         reward_metrics = defaultdict(list)

#         # ------------------ hyperparams ------------------
#         warmup_steps = 200
#         bonus = 0.1
#         mean_high = 0.8
#         mean_low  = 0.2

#         # ------------------ helpers ------------------
#         def is_valid_for_balance(score: dict) -> bool:
#             """Only use well-formatted samples to estimate/apply balancing."""
#             return score.get("format", 0.0) == 1.0 and ("use_reasoning" in score)

#         def get_target_reasoning(batch_mean: float) -> Optional[float]:
#             """
#             Decide which use_reasoning value to encourage based on batch mean.
#             - mean > mean_high => encourage 0.0
#             - mean < mean_low  => encourage 1.0
#             - else             => no shaping
#             """
#             if batch_mean > mean_high:
#                 return 0.0
#             if batch_mean < mean_low:
#                 return 1.0
#             return None

#         # ------------------ pass 1: compute scores ------------------
#         lengths = response_mask.sum(dim=-1).tolist()  # list[int], faster than per-sample .item()
#         cached = []  # list of dicts: {"i": int, "len": int, "score": dict}

#         for i, cur_len in enumerate(lengths):
#             cur_len = int(cur_len)
#             valid_ids = responses[i][:cur_len]
#             response_str = self.tokenizer.decode(
#                 valid_ids, skip_special_tokens=self.config.skip_special_tokens
#             )

#             score = self.reward_fn(
#                 {
#                     "response": response_str,
#                     "response_length": cur_len,
#                     "ground_truth": data.non_tensor_batch["ground_truth"][i],
#                 }
#             )
#             cached.append({"i": i, "len": cur_len, "score": score})

#         # ------------------ pass 2: batch-level balancing (optional) ------------------
#         # (step is not None) and (step < warmup_steps) and 
#         do_balance = (bonus != 0.0)

#         batch_mean = None
#         target_reasoning = None
#         if do_balance:
#             use_r_vals = [
#                 float(item["score"]["use_reasoning"])
#                 for item in cached
#                 if is_valid_for_balance(item["score"])
#             ]
#             if use_r_vals:
#                 batch_mean = sum(use_r_vals) / len(use_r_vals)
#                 target_reasoning = get_target_reasoning(batch_mean)

#                 if target_reasoning is not None:
#                     for item in cached:
#                         s = item["score"]
#                         if is_valid_for_balance(s) and float(s["use_reasoning"]) == float(target_reasoning):
#                             s["overall"] = float(s["overall"]) + bonus
#                             s["use_reasoning_balance_bonus"] = bonus
#                         else:
#                             s["use_reasoning_balance_bonus"] = 0.0

#         # ------------------ pass 3: write rewards + metrics ------------------
#         for item in cached:
#             i, cur_len, score = item["i"], item["len"], item["score"]

#             # place scalar reward at the last token position
#             last_idx = cur_len - 1
#             if last_idx >= 0:
#                 reward_tensor[i, last_idx] = float(score["overall"])

#             # log batch-level info per-sample for easy aggregation later
#             if batch_mean is not None:
#                 score["use_reasoning_batch_mean"] = float(batch_mean)
#             score["use_reasoning_target"] = float(target_reasoning) if target_reasoning is not None else -1.0

#             for k, v in score.items():
#                 reward_metrics[k].append(float(v) if isinstance(v, (int, float)) else v)

#         return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardGroundingManager(FunctionRewardManager):
    reward_fn: BatchRewardFunctionGrounding

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "image": data.non_tensor_batch["multi_modal_data"][i]['images'][0],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
