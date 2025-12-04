#!/usr/bin/env python

# Copyright 2025 Bryson Jones. All rights reserved.
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
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STR

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.utils import (
    move_to_device,
    normalize_batch,
    unnormalize_batch,
)

# Suppress Pydantic warnings from draccus ChoiceRegistry union types
# This is an interaction with draccus that we can't control
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


@dataclass
class InferenceConfig:
    """Configuration for running inference."""

    checkpoint_dir: str = ""
    dataset_path: str = ""
    device: str = "cuda"
    seed: int = 17


def inference(cfg: InferenceConfig):
    checkpoint_dir = Path(cfg.checkpoint_dir)
    dataset_path = Path(cfg.dataset_path) if cfg.dataset_path else None
    device = cfg.device

    if not torch.cuda.is_available() and device == "cuda":
        logging.warning("=" * 80)
        logging.warning("WARNING: No CUDA available, using CPU. Inference will be significantly slower.")
        logging.warning("=" * 80)
        device = "cpu"
    device = torch.device(device)

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading model from {checkpoint_dir} on {device}...")
    policy = MultiTaskDiTPolicy.load(checkpoint_dir)
    policy.to(device)
    policy.eval()

    # Load stats from JSON (matching lerobot convention)
    stats_path = checkpoint_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. "
            f"Dataset statistics are required for proper normalization. "
            f"Please ensure the checkpoint directory contains 'dataset_stats.json'."
        )

    with open(stats_path) as f:
        stats_json = json.load(f)
    stats_tensors = {
        key: {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in stat_dict.items()}
        for key, stat_dict in stats_json.items()
    }

    # Load dataset
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {cfg.dataset_path}\n" f"Please provide a valid dataset path using --dataset_path"
        )

    print(f"Loading dataset from {dataset_path}...")
    repo_id = dataset_path.name

    # For inference, we only need a single frame (no temporal stacking)
    # Load dataset without delta_timestamps to get single frames
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        video_backend="torchcodec",
    )

    policy_config = policy.config

    # Get a random sample from the dataset (single frame, no temporal stacking)
    dataset_size = len(dataset)
    random_idx = random.randint(0, dataset_size - 1)
    print(f"Loading random sample {random_idx} from dataset (size: {dataset_size})...")
    sample = dataset[random_idx]

    # Extract images for visualization before preprocessing
    print("Preparing sample for inference...")
    all_images = []

    if policy_config.image_features:
        image_keys = list(policy_config.image_features.keys())
        for key in image_keys:
            if key in sample:
                img_tensor = sample[key]
                if isinstance(img_tensor, torch.Tensor):
                    # Single frame: (C, H, W)
                    if len(img_tensor.shape) == 3:  # (C, H, W)
                        all_images.append(img_tensor.cpu().numpy())

    # Reset policy
    policy.reset()

    # Use prepare_observation_for_inference and select_action
    # The sample is a single frame, select_action will handle queue population
    print("Generating predicted actions from policy...")

    with torch.no_grad():
        # Keep only observation or task keys in `sample`
        sample = {k: v for k, v in sample.items() if k.startswith("observation") or k == "task"}
        
        # Move sample tensors to device (stats_tensors are already on device)
        sample = move_to_device(sample, device, non_blocking=False)
        
        normalized_obs = normalize_batch(
            sample,
            policy_config.input_features,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
        )

        for name in normalized_obs:
            if "task" in name:
                continue
            if "image" in name:
                normalized_obs[name] = normalized_obs[name].type(torch.float32) / 255
                normalized_obs[name] = normalized_obs[name].permute(2, 0, 1).contiguous()
            normalized_obs[name] = normalized_obs[name].unsqueeze(0)
            normalized_obs[name] = normalized_obs[name].to(device)

        # Get the predicted action
        # select_action will populate queues (copying the observation if needed) and return an action
        predicted_action_tensor = policy.select_action(normalized_obs)

        # Unnormalize the action (add batch dimension if needed)
        action_with_batch = (
            predicted_action_tensor.unsqueeze(0) if len(predicted_action_tensor.shape) == 1 else predicted_action_tensor
        )
        action_dict = {ACTION: action_with_batch}
        unnormalized_action_dict = unnormalize_batch(
            action_dict,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
            feature_keys=[ACTION],
        )
        predicted_action = unnormalized_action_dict[ACTION]
        # Remove batch dimension if it was added
        if len(predicted_action.shape) > 1:
            predicted_action = predicted_action[0]
        predicted_action = predicted_action.cpu().numpy()
        predicted_actions = [predicted_action]

    # Convert actions to numpy arrays
    predicted_actions_array = np.array(predicted_actions)  # (1, action_dim)
    action_dim = (
        predicted_actions_array.shape[1] if len(predicted_actions_array.shape) > 1 else len(predicted_actions_array)
    )

    # Create visualization
    print("Creating visualization...")

    # Determine layout: images on top (if available), actions below
    num_image_cols = min(5, max(1, len(all_images))) if all_images else 0
    num_image_rows = (len(all_images) + num_image_cols - 1) // num_image_cols if all_images else 0

    # Total rows: images + 1 for actions
    total_rows = num_image_rows + 1 if all_images else 1

    plt.figure(figsize=(16, 4 + 2 * total_rows))

    # Plot images if available
    if all_images:
        for idx, img in enumerate(all_images):
            ax = plt.subplot(total_rows, num_image_cols, idx + 1)
            # Convert from (C, H, W) to (H, W, C) for display
            if len(img.shape) == 3 and img.shape[0] == 3:  # RGB (C, H, W)
                img_display = np.transpose(img, (1, 2, 0))
                # Normalize to [0, 1] if needed
                if img_display.max() > 1.0:
                    img_display = img_display / 255.0
                ax.imshow(np.clip(img_display, 0, 1))
            elif len(img.shape) == 3 and img.shape[0] == 1:  # Grayscale (1, H, W)
                img_display = img[0]
                ax.imshow(img_display, cmap="gray")
            elif len(img.shape) == 2:  # Already (H, W)
                ax.imshow(img, cmap="gray")
            else:
                # Fallback: try to display first channel
                ax.imshow(img[0] if len(img.shape) == 3 else img, cmap="gray")
            ax.set_title(f"Frame {idx}", fontsize=8)
            ax.axis("off")

    # Plot predicted action trajectory
    ax = plt.subplot(total_rows, 1, total_rows)

    # Ensure predicted_actions_array is 2D
    if len(predicted_actions_array.shape) == 1:
        predicted_actions_array = predicted_actions_array.reshape(1, -1)

    # Plot predicted action trajectory
    timesteps = np.arange(len(predicted_actions_array))
    for dim in range(action_dim):
        ax.plot(
            timesteps,
            predicted_actions_array[:, dim],
            label=f"Dim {dim}",
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=8,
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Action Value", fontsize=12)
    ax.set_title("Predicted Action Trajectory", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Visualization complete!")
    print(f"  Predicted action: {action_dim} dimensions")
    print(f"  Images displayed: {len(all_images)}")


if __name__ == "__main__":
    # Use draccus.parse() directly with explicit config class to avoid auto-discovery
    cfg = draccus.parse(InferenceConfig)
    inference(cfg)
