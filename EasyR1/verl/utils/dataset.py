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

import math
import os
import csv
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union
import random
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .vsi_zeroshot_prompts import (
    build_student_message_list,
    extract_vsi_q_text,
    meta_from_example,
    parse_image_paths,
)


def make_rlhf_collate_fn(pad_token_id: int):
    """input_ids / attention_mask / position_ids within a batch may have different lengths (when multimodal full-length is preserved); pad before stacking."""

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)
        for feature in features:
            for key, value in feature.items():
                if isinstance(value, torch.Tensor):
                    tensors[key].append(value)
                else:
                    non_tensors[key].append(value)

        seq_pad_keys = ("input_ids", "attention_mask", "position_ids")
        for key in seq_pad_keys:
            if key not in tensors:
                continue
            lst = tensors[key]
            max_len = max(x.size(-1) for x in lst)
            pad_val = pad_token_id if key == "input_ids" else 0
            tensors[key] = torch.stack(
                [VF.pad_sequence_to_length(x, max_len, pad_val, left_pad=True) for x in lst],
                dim=0,
            )

        for key, lst in tensors.items():
            if key in seq_pad_keys:
                continue
            tensors[key] = torch.stack(lst, dim=0)

        for key, value in non_tensors.items():
            # np.array(list_of_lists, dtype=object) becomes 2D when inner lists share
            # the same length (e.g. teacher_images with 3 vs 16 paths). mini_rollout
            # concat then fails with mismatched ndim across batches.
            arr = np.empty(len(value), dtype=object)
            arr[:] = value
            non_tensors[key] = arr

        return {**tensors, **non_tensors}

    return collate_fn


# default pad=0; for training use the collate_fn constructed from the tokenizer inside data_loader
collate_fn = make_rlhf_collate_fn(0)


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        if "path" in image:
            image = Image.open(image["path"])
        else:
            image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        teacher_image_key: str = "teacher_images",
        use_vsi_zeroshot_prompts: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.teacher_image_key = teacher_image_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.use_vsi_zeroshot_prompts = use_vsi_zeroshot_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        # if os.path.isdir(data_path):
        #     # when we use dataset builder, we should always refer to the train split
        #     file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
        #     self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        # elif os.path.isfile(data_path):
        #     file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
        #     self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        # else:
        #     # load remote dataset from huggingface hub
        #     self.dataset = load_dataset(data_path, split=data_split)
        def _load_one(data_path: str, data_split: str):
            if os.path.isdir(data_path):
                file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
                return load_dataset(file_type, data_dir=data_path, split=data_split)
            elif os.path.isfile(data_path):
                file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
                print(f"Loading dataset from {data_path} with file type {file_type}")
                return load_dataset(file_type, data_files=data_path, split=data_split)
            else:
                return load_dataset(data_path, split=data_split)
        
        if isinstance(data_path, str) and ("," in data_path):
            # csv parsing: allows paths like "/path/a","/path/with,comma"
            paths = [p.strip() for p in next(csv.reader([data_path], skipinitialspace=True)) if p.strip()]
            if len(paths) > 1:
                ds0 = _load_one(paths[0], data_split)
                rest = [_load_one(p, data_split).cast(ds0.features) for p in paths[1:]]
                self.dataset = concatenate_datasets([ds0] + rest)
            else:
                self.dataset = _load_one(paths[0], data_split)
        else:
            self.dataset = _load_one(data_path, data_split)


        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages_from_preformatted(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Handle GPD pre-formatted messages where user content is a string with <image> placeholders."""
        result = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                content_list = []
                for i, part in enumerate(msg["content"].split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})
                    if part:
                        content_list.append({"type": "text", "text": part})
                result.append({"role": "user", "content": content_list})
            else:
                result.append(msg)
        return result

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if '<image>' in prompt_str:
            prompt_str = prompt_str.replace(f'<image>Please generate the next move according to the UI screenshot', f'<image>Please generate the next move according to the UI screenshot (original image size {example["answer"]["image_width"]}x{example["answer"]["image_height"]})')
            prompt_str = prompt_str.replace('<image>', '')
            
            
        grounding_data = False
        if "system_prompt" in example:
            system_prompt =[{"role": "system", "content": example["system_prompt"]}]
            if 'write' not in example["system_prompt"].lower():
                grounding_data = True
        else:
            system_prompt = []

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        
        if grounding_data:
            prompt_str = prompt_str + """\n\nThe response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "Click",
  "value": "None",
  "point_2d": [x, y]
}
</answer>
"""
        else:
            # qwen3 base model has issue when parsing the <think> </think> tags
            if 'qwen3' in self.tokenizer.name_or_path.lower():
                prompt_str = prompt_str + """\n\nThe response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}
</answer>
"""
            else:
                prompt_str = prompt_str + """\n\nThe response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
<think> Your step-by-step thought process here... </think>
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}
</answer>
"""

        # if random.random() < 0.1:
        #     print("Prompt after formatting:", prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return  system_prompt + [{"role": "user", "content": content_list}], grounding_data
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return  system_prompt + [{"role": "user", "content": content_list}], grounding_data
        else:
            return  system_prompt + [{"role": "user", "content": prompt_str}], grounding_data

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        prompt_val = example[self.prompt_key]
        if self.use_vsi_zeroshot_prompts and self.image_key in example and self.processor is not None:
            q_text = extract_vsi_q_text(example, self.prompt_key)
            answer_kind, _ = meta_from_example(example)
            messages = build_student_message_list(
                len(example[self.image_key]), q_text, answer_kind=answer_kind
            )
        elif isinstance(prompt_val, list):
            messages = self._build_messages_from_preformatted(prompt_val)
        else:
            messages, _ = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        prompt_val = example[self.prompt_key]
        if self.use_vsi_zeroshot_prompts and self.image_key in example and self.processor is not None:
            q_text = extract_vsi_q_text(example, self.prompt_key)
            answer_kind, priv_variant = meta_from_example(example)
            messages = build_student_message_list(
                len(example[self.image_key]), q_text, answer_kind=answer_kind
            )
            is_grounding = False
            example["vsi_q_text"] = q_text
            example["vsi_answer_kind"] = answer_kind
            example["vsi_priv_variant"] = priv_variant
        elif isinstance(prompt_val, list):
            # GPD: prompt is already a list of chat message dicts
            messages = self._build_messages_from_preformatted(prompt_val)
            is_grounding = False
        else:
            messages, is_grounding = self._build_messages(example)
        example.pop(self.prompt_key, None)

        # Truncating multimodal sequences at max_prompt_length creates inconsistency with worker-side pixel recomputation (image token count != patch count)
        multimodal_keep_full_len = False
        if self.image_key in example:
            multimodal_keep_full_len = True
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            image_width_new, image_height_new = processed_images[-1].size
            # When GUI ground_truth is a dict, write the resized dimensions into it; for GPD and similar tasks the answer is a str and cannot be subscript-assigned
            _ans = example.get(self.answer_key)
            if isinstance(_ans, dict):
                _ans["image_width_new"] = image_width_new
                _ans["image_height_new"] = image_height_new

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            multimodal_keep_full_len = True
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        post_max_length = self.max_prompt_length
        if multimodal_keep_full_len:
            post_max_length = max(self.max_prompt_length, len(input_ids))

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=post_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        ground_truth = example.pop(self.answer_key)
        if isinstance(ground_truth, dict):
            ground_truth['is_grounding'] = is_grounding
        example["ground_truth"] = ground_truth
        return example
