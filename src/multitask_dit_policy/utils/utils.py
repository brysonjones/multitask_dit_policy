#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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
Utilities for the Multi-Task Diffusion Transformer (DiT) policy.

Credit is given to the https://github.com/huggingface/lerobot project
for which many of these util functions are adapted from.
"""

import json
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

# Import from lerobot where available


# NormalizationMode is not available in lerobot, so we define it here
class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


def populate_queues(queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def normalize_tensor(tensor: torch.Tensor, stats: dict, mode: NormalizationMode, eps: float = 1e-8) -> torch.Tensor:
    """Apply normalization transformation to a tensor.

    Uses PyTorch broadcasting - stats are used as-is without reshaping.
    """
    if mode == NormalizationMode.IDENTITY or not stats:
        return tensor

    if mode == NormalizationMode.MIN_MAX:
        min_val = stats["min"]
        max_val = stats["max"]
        denom = max_val - min_val
        # Avoid division by zero
        denom = torch.where(denom == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), denom)
        # Map from [min, max] to [-1, 1]
        return 2 * (tensor - min_val) / denom - 1
    elif mode == NormalizationMode.MEAN_STD:
        mean = stats["mean"]
        std = stats["std"]
        # Avoid division by zero
        denom = std + eps
        return (tensor - mean) / denom
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")


def unnormalize_tensor(tensor: torch.Tensor, stats: dict, mode: NormalizationMode, eps: float = 1e-8) -> torch.Tensor:
    """Apply unnormalization (inverse normalization) transformation to a tensor.

    Uses PyTorch broadcasting - stats are used as-is without reshaping.
    """
    if mode == NormalizationMode.IDENTITY or not stats:
        return tensor

    if mode == NormalizationMode.MIN_MAX:
        min_val = stats["min"]
        max_val = stats["max"]
        denom = max_val - min_val
        # Avoid division by zero
        denom = torch.where(denom == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), denom)
        # Map from [-1, 1] back to [min, max]
        return (tensor + 1) / 2 * denom + min_val
    elif mode == NormalizationMode.MEAN_STD:
        mean = stats["mean"]
        std = stats["std"]
        return tensor * std + mean
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")


def normalize_batch(
    batch: dict[str, torch.Tensor],
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    normalization_mapping: dict[FeatureType, NormalizationMode],
    stats_tensors: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Normalize a batch of data according to policy configuration and dataset statistics.

    - Visual: (B, T, C, H, W) where B can be 1 for inference
    - State/Action: (B, T, D) where B can be 1 for inference

    Args:
        batch: Dictionary mapping feature keys to tensors (e.g., {"observation.image": tensor, "action": tensor})
        input_features: Dictionary mapping feature keys to PolicyFeature objects for input features
        output_features: Dictionary mapping feature keys to PolicyFeature objects for output features
        normalization_mapping: Dictionary mapping FeatureType to NormalizationMode
        stats_tensors: Dictionary mapping feature keys to their statistics tensors (mean, std, min, max, etc.)

    Returns:
        Dictionary with normalized tensors, preserving keys that don't need normalization (e.g., "task").
    """
    normalized_batch = {}
    for key, val in batch.items():
        # Skip non-tensor keys (e.g., "task" which is a list)
        if not isinstance(val, torch.Tensor):
            normalized_batch[key] = val
            continue

        # Check if we have stats and a normalization mode for this key
        mode = None
        if key in input_features:
            ft_type = input_features[key].type
            mode = normalization_mapping.get(ft_type)
        elif key in output_features:
            ft_type = output_features[key].type
            mode = normalization_mapping.get(ft_type)

        if mode and key in stats_tensors:
            curr_stats = stats_tensors[key]
            # Use stats directly - PyTorch broadcasting handles the rest
            normalized_batch[key] = normalize_tensor(val.float(), curr_stats, mode)
        else:
            normalized_batch[key] = val

    return normalized_batch


def unnormalize_batch(
    batch: dict[str, torch.Tensor],
    output_features: dict[str, PolicyFeature],
    normalization_mapping: dict[FeatureType, NormalizationMode],
    stats_tensors: dict[str, dict[str, torch.Tensor]],
    feature_keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Unnormalize a batch of data according to policy configuration and dataset statistics.

    - Visual: (B, T, C, H, W) where B can be 1 for inference
    - State/Action: (B, T, D) where B can be 1 for inference

    Args:
        batch: Dictionary mapping feature keys to normalized tensors
        output_features: Dictionary mapping feature keys to PolicyFeature objects for output features
        normalization_mapping: Dictionary mapping FeatureType to NormalizationMode
        stats_tensors: Dictionary mapping feature keys to their statistics tensors (mean, std, min, max, etc.)
        feature_keys: Optional list of keys to unnormalize. If None, unnormalizes all keys in output_features.

    Returns:
        Dictionary with unnormalized tensors, preserving keys that don't need unnormalization.
    """
    if feature_keys is None:
        feature_keys = list(output_features.keys())

    unnormalized_batch = {}
    for key, val in batch.items():
        # Skip non-tensor keys
        if not isinstance(val, torch.Tensor):
            unnormalized_batch[key] = val
            continue

        # Only unnormalize specified feature keys
        if key not in feature_keys:
            unnormalized_batch[key] = val
            continue

        # Check if we have stats and a normalization mode for this key
        mode = None
        if key in output_features:
            ft_type = output_features[key].type
            mode = normalization_mapping.get(ft_type)

        if mode and key in stats_tensors:
            curr_stats = stats_tensors[key]
            # Use stats directly - PyTorch broadcasting handles the rest
            unnormalized_batch[key] = unnormalize_tensor(val.float(), curr_stats, mode)
        else:
            unnormalized_batch[key] = val

    return unnormalized_batch


def move_to_device(data: Any, device: str, non_blocking: bool = None) -> Any:
    """
    Recursively move all tensors to the specified device with optimizations.

    Args:
        data: Can be a tensor, dict, list, or any other type
        device: Target device string (e.g., "cpu", "cuda", "cuda:0")
        non_blocking: If None, auto-detects based on device type (True for CUDA)

    Returns:
        Data with all tensors moved to the target device
    """
    target_device = torch.device(device)

    # Auto-detect non_blocking for CUDA
    if non_blocking is None:
        non_blocking = target_device.type == "cuda"

    if isinstance(data, torch.Tensor):
        # Skip move if already on target device
        if data.device == target_device:
            return data
        return data.to(target_device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device, non_blocking) for key, value in data.items()}
    elif isinstance(data, list | tuple):
        return type(data)(move_to_device(item, device, non_blocking) for item in data)
    else:
        # For non-tensor types (strings, ints, etc.), return as-is
        return data


def save_policy(
    policy: Any,
    save_directory: str | Path,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Save a policy model and optionally its associated dataset statistics.

    This function saves the policy (config and weights) and optionally the dataset statistics
    needed for inference. The stats are saved as JSON in a format compatible with
    lerobot conventions.

    Args:
        policy: The policy model instance (must have a .save() method)
        save_directory: Directory path where to save the policy and stats
        dataset_stats: Optional dictionary mapping feature keys to their statistics dictionaries.
                      Each stat dict contains keys like "mean", "std", "min", "max", etc.
                      Values can be numpy arrays, tensors, or other JSON-serializable types.
                      If None, only the policy is saved.
    """
    save_directory = Path(save_directory)

    # Save the policy (config and model weights)
    policy.save(save_directory)

    # Optionally save stats if provided
    if dataset_stats is not None:
        # Convert stats to JSON-serializable format (matching lerobot convention)
        # Convert numpy arrays and tensors to lists for JSON serialization
        stats_json = {}
        for key, stat_dict in dataset_stats.items():
            stats_json[key] = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stat_dict.items()}

        # Save stats JSON file
        with open(save_directory / "dataset_stats.json", "w") as f:
            json.dump(stats_json, f, indent=2)
