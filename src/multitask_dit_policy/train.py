#!/usr/bin/env python

# Copyright 2025 Bryson Jones and the HuggingFace Team. All rights reserved.
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
Training script for Multi-Task Diffusion Transformer (DiT) policy.

Credit is given to the https://github.com/huggingface/lerobot project
for which this training script is adapted from.
"""

import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig
from multitask_dit_policy.utils.utils import move_to_device, normalize_batch, save_policy

# Suppress Pydantic warnings from draccus ChoiceRegistry union types
# This is an interaction with draccus that we can't control
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


@dataclass
class TrainConfig:
    # Dataset parameters
    dataset_path: str

    # Training parameters
    batch_size: int = 16
    num_workers: int = 4
    train_steps: int = 40000
    save_freq: int = 1000
    log_freq: int = 1
    output_dir: str = "outputs/train_multi_task_dit"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    seed: int = 17

    # Checkpoint loading
    checkpoint_path: str | None = None

    # Policy parameters
    policy: MultiTaskDiTConfig = field(default_factory=MultiTaskDiTConfig)


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.device == "cpu" or cfg.device.startswith("cpu"):
        logging.warning("=" * 80)
        logging.warning("WARNING: Config device is set to CPU")
        logging.warning("Training will be significantly slower than on GPU")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if wandb logging is enabled
    use_wandb = "WANDB_API_KEY" in os.environ
    if use_wandb:
        wandb.init(
            project="multitask-dit-policy",
            name=output_dir.name,
            config=asdict(cfg),
            dir=str(output_dir),
        )

    # Extract repo_id from the final directory name of dataset_path
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {cfg.dataset_path}\n"
            f"Please ensure the dataset directory exists at the specified path."
        )
    repo_id = dataset_path.name

    # need metadata to get features and stats
    ds_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=cfg.dataset_path)

    # Load or create policy
    if cfg.checkpoint_path is not None:
        logging.info(f"Loading policy from checkpoint: {cfg.checkpoint_path}")
        policy = MultiTaskDiTPolicy.load(cfg.checkpoint_path)
        # Use the checkpoint's config for feature setup
        policy_config = policy.config
        logging.info("Policy loaded successfully from checkpoint")
    else:
        policy_config = cfg.policy
        policy = MultiTaskDiTPolicy(policy_config, dataset_metadata=ds_metadata)

    policy_config.device = cfg.device
    policy.to(cfg.device)
    policy.train()

    delta_indices = {}
    # For observation keys
    for key in policy_config.input_features:
        delta_indices[key] = policy_config.observation_delta_indices
    # For action keys
    for key in policy_config.output_features:
        delta_indices[key] = policy_config.action_delta_indices

    delta_timestamps = {key: [i / ds_metadata.fps for i in indices] for key, indices in delta_indices.items()}

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=cfg.dataset_path,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",  #  set torchcodec explicitly
    )

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=cfg.policy.drop_n_last_frames,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.device == "cuda" else False,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    optimizer_config = policy_config.get_optimizer_preset()
    optimizer_params = policy.get_optim_params()
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )

    use_amp = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logging.info("Using Automatic Mixed Precision (AMP) for training")

    stats_tensors = {}
    for key, stat in ds_metadata.stats.items():
        stats_tensors[key] = {k: torch.tensor(v, device=cfg.device, dtype=torch.float32) for k, v in stat.items()}

    step = 0
    progress_bar = tqdm(total=cfg.train_steps)

    # infinite iterator over dataloader (cycles automatically)
    dataloader_iter = cycle(dataloader)

    while step < cfg.train_steps:
        batch = next(dataloader_iter)
        batch = move_to_device(batch, cfg.device, non_blocking=True)

        normalized_batch = normalize_batch(
            batch,
            policy_config.input_features,
            policy_config.output_features,
            policy_config.normalization_mapping,
            stats_tensors,
        )

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=cfg.device, enabled=use_amp):
            loss, _ = policy(normalized_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale for gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % cfg.save_freq == 0:
            save_path = output_dir / f"checkpoint_{step}"
            save_policy(policy, save_path, ds_metadata.stats)
            logging.info(f"Saved checkpoint to {save_path}")
            if use_wandb:
                wandb.log({"checkpoint_step": step}, step=step)

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

        if step % cfg.log_freq == 0:
            logging.info(f"Step {step}: Loss = {loss.item():.6f}")
            if use_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "step": step,
                    }
                )

    final_dir = output_dir / "final_model"
    save_policy(policy, final_dir, ds_metadata.stats)
    logging.info("Training finished.")
    if use_wandb:
        wandb.finish()


@draccus.wrap()
def main(cfg: TrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
