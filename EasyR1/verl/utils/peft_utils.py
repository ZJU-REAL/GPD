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

"""PEFT / LoRA helpers for OPSD-style fixed teacher (disable_adapter on Teacher forward)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def unwrap_fsdp_module(module: nn.Module) -> nn.Module:
    while isinstance(module, FSDP):
        module = module._fsdp_wrapped_module
    return module


def is_peft_model(module: nn.Module) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(unwrap_fsdp_module(module), PeftModel)


def opsd_teacher_disable_adapter_ctx(module: nn.Module) -> AbstractContextManager:
    """Teacher forward: Base only (LoRA off), matching official OPSD ``fixed_teacher``."""
    inner = unwrap_fsdp_module(module)
    if hasattr(inner, "disable_adapter"):
        return inner.disable_adapter()
    return nullcontext()


def _normalize_param_name(name: str) -> str:
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("base_model.model.", "").replace("base_model.", "")
    return name.replace(".base_layer", "")


def _is_lora_param_name(name: str) -> bool:
    return ("lora_" in name) or (".adapter_" in name)


def normalize_peft_param_name(params: dict[str, Any]) -> dict[str, Any]:
    """Strip PEFT / FSDP prefixes so vLLM/HF can load merged weights."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        norm_key = _normalize_param_name(key)
        if _is_lora_param_name(norm_key):
            continue
        out[norm_key] = value
    return out


def _merge_lora_layers_(module: nn.Module, merge: bool) -> None:
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, LoraLayer):
                is_merged = getattr(layer, "merged", False)
                if merge and not is_merged:
                    layer.merge()
                elif (not merge) and is_merged:
                    layer.unmerge()


def _clean_merged_lora_state_(module: nn.Module) -> None:
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, LoraLayer):
                merged_adapters = getattr(layer, "merged_adapters", None)
                if merged_adapters:
                    layer.merged_adapters = []


def _backup_lora_base_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    from peft.tuners.lora import LoraLayer

    backups: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, layer in module.named_modules():
            if isinstance(layer, LoraLayer):
                base = layer.get_base_layer()
                backups[name] = base.weight.data.clone()
    return backups


def _restore_lora_base_weights(module: nn.Module, backups: dict[str, torch.Tensor]) -> None:
    from peft.tuners.lora import LoraLayer

    with torch.no_grad():
        for name, layer in module.named_modules():
            if isinstance(layer, LoraLayer) and name in backups:
                base = layer.get_base_layer()
                base.weight.data.copy_(backups[name])
        _clean_merged_lora_state_(module)


def _tensor_from_state_value(val: Any) -> torch.Tensor:
    if hasattr(val, "full_tensor"):
        return val.full_tensor().detach()
    return val.detach()


def _merge_lora_state_dict(raw_sd: dict[str, Any], peft_inner: nn.Module) -> dict[str, torch.Tensor]:
    """Merge LoRA A/B tensors into base weights in a copied state dict (no model mutation)."""
    peft_cfg = peft_inner.peft_config.get("default")
    if peft_cfg is None:
        raise RuntimeError("collect_merged_hf_state_dict expects a default peft config.")
    scaling = float(peft_cfg.lora_alpha) / float(peft_cfg.r)

    merged: dict[str, torch.Tensor] = {}
    lora_a: dict[str, torch.Tensor] = {}
    lora_b_keys: dict[str, str] = {}

    for key, val in raw_sd.items():
        if "_flat_param" in key:
            continue
        if ".lora_A." in key:
            lora_a[key.split(".lora_A.")[0]] = _tensor_from_state_value(val)
            continue
        if ".lora_B." in key:
            lora_b_keys[key.split(".lora_B.")[0]] = key
            continue
        if "lora_" in key:
            continue
        clean_name = _normalize_param_name(key)
        if _is_lora_param_name(clean_name):
            continue
        merged[clean_name] = _tensor_from_state_value(val).clone()

    for prefix, a_weight in lora_a.items():
        b_key = lora_b_keys.get(prefix)
        if b_key is None:
            continue
        b_weight = _tensor_from_state_value(raw_sd[b_key])
        delta = (b_weight @ a_weight) * scaling
        base_clean = _normalize_param_name(f"{prefix}.base_layer.weight")
        if base_clean not in merged:
            base_clean = _normalize_param_name(f"{prefix}.weight")
        if base_clean not in merged:
            continue
        merged[base_clean] = merged[base_clean] + delta.to(dtype=merged[base_clean].dtype)
    return merged


def collect_merged_hf_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Export merged HF weights for vLLM via FSDP FULL_STATE_DICT (never summon_full_params)."""
    inner = unwrap_fsdp_module(module)
    if not hasattr(inner, "base_model"):
        raise TypeError("collect_merged_hf_state_dict expects a PeftModel.")

    root = module
    if isinstance(root, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(root, StateDictType.FULL_STATE_DICT, state_cfg):
            raw_sd = root.state_dict()
    else:
        raw_sd = root.state_dict()
    return _merge_lora_state_dict(raw_sd, inner)


def peft_base_pretrained_model(module: nn.Module) -> nn.Module:
    """Underlying HF model inside FSDP / PeftModel (for checkpoint key mapping)."""
    inner = unwrap_fsdp_module(module)
    base = getattr(inner, "base_model", inner)
    return getattr(base, "model", base)


def merge_lora_adapter(module: nn.Module) -> bool:
    """Deprecated: in-place merge is unsafe under FSDP use_orig_params; use collect_merged_hf_state_dict."""
    inner = unwrap_fsdp_module(module)
    if not hasattr(inner, "base_model"):
        return False
    _merge_lora_layers_(module if not isinstance(module, FSDP) else unwrap_fsdp_module(module), merge=True)
    return True


def unmerge_lora_adapter(module: nn.Module) -> bool:
    inner = unwrap_fsdp_module(module)
    if not hasattr(inner, "base_model"):
        return False
    _merge_lora_layers_(module if not isinstance(module, FSDP) else unwrap_fsdp_module(module), merge=False)
    return True


def apply_lora_to_model(
    model: nn.Module,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Any,
    torch_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    modules = list(target_modules) if target_modules is not None else []
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if getattr(getattr(model, "config", None), "use_cache", None) is not None:
        model.config.use_cache = False
    # PEFT defaults LoRA to fp32; FSDP flatten requires uniform dtype within a layer
    if torch_dtype is not None:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=torch_dtype)
    return model
