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

    dataset_size = len(dataset)
    random_idx = random.randint(0, dataset_size - 1)
    print(f"Loading random sample {random_idx} from dataset (size: {dataset_size})...")
    sample = dataset[random_idx]
    # filter out everything except the observation and task keys
    # in practice, you would just built the observation dict with only these keys
    # this is necessary here because we are loading from the dataset, which has all keys
    sample = {k: v for k, v in sample.items() if k.startswith("observation") or k == "task"}
    sample = move_to_device(sample, device, non_blocking=False)
        
    for name in sample:
        if "task" in name:
            continue
        sample[name] = sample[name].unsqueeze(0)

    policy.reset()

    print("Generating predicted actions from policy...")

    with torch.no_grad():

        normalized_obs = normalize_batch(
            sample,
            policy_config.input_features,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
        )

        # Get the full action trajectory by calling select_action until queue is empty
        # NOTE: This while loop is just to get the full action trajectory, in practice 
        #       you would just call select_action per loop iteration.
        predicted_actions_list = []
        while True:
            predicted_action_tensor = policy.select_action(normalized_obs)

            if len(predicted_action_tensor.shape) > 1:
                predicted_action_tensor = predicted_action_tensor.squeeze()
            predicted_actions_list.append(predicted_action_tensor)
            # Stop when queue is empty (after we've collected all actions from the chunk)
            if len(policy._queues[ACTION]) == 0:
                break

        predicted_action_trajectory = torch.stack(predicted_actions_list, dim=0)

        # Unnormalize the full action trajectory
        action_dict = {ACTION: predicted_action_trajectory.unsqueeze(0)}
        unnormalized_action_dict = unnormalize_batch(
            action_dict,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
            feature_keys=[ACTION],
        )
        predicted_actions_trajectory = unnormalized_action_dict[ACTION]
        
        # Remove batch dimension and any extra dimensions
        predicted_actions_trajectory = predicted_actions_trajectory.squeeze(0)
        
        predicted_actions = predicted_actions_trajectory.cpu().numpy()
    
    action_dim = predicted_actions.shape[1]
    horizon = predicted_actions.shape[0]
    print(f"Generated action trajectory: shape={predicted_actions.shape} (horizon={horizon}, action_dim={action_dim})")

    print("Creating visualization...")

    plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 1, 1)

    # predicted_actions_array should already be 2D (horizon, action_dim) from above
    timesteps = np.arange(len(predicted_actions))
    for dim in range(action_dim):
        ax.plot(
            timesteps,
            predicted_actions[:, dim],
            label=f"Action State {dim}",
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


if __name__ == "__main__":
    # Use draccus.parse() directly with explicit config class to avoid auto-discovery
    cfg = draccus.parse(InferenceConfig)
    inference(cfg)
