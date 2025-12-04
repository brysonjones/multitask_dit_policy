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

"""Multi-Task Diffusion Transformer (DiT) Policy

Transformer-based diffusion policy for multi-task robot learning with text and vision conditioning.
Supports both diffusion and flow matching objectives for action generation.
"""

from collections import deque
from pathlib import Path

import draccus
import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor

from multitask_dit_policy.model.objectives import DiffusionObjective, FlowMatchingObjective
from multitask_dit_policy.model.observation_encoder import ObservationEncoder
from multitask_dit_policy.model.transformer import DiffusionTransformer
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig
from multitask_dit_policy.utils.utils import populate_queues


class MultiTaskDiTPolicy(nn.Module):
    config_class = MultiTaskDiTConfig
    name = "multi_task_dit"

    def __init__(self, config: MultiTaskDiTConfig, dataset_metadata: LeRobotDatasetMetadata | None = None):
        super().__init__()

        # Extract features from dataset metadata if provided and features aren't already set
        if dataset_metadata is not None and (not config.input_features or not config.output_features):
            input_features = {}
            output_features = {}
            for key, ft in dataset_metadata.features.items():
                if key == "index":
                    continue

                shape = tuple(ft["shape"])
                if key.startswith(OBS_IMAGE):
                    h, w, c = shape
                    input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
                elif key == OBS_STATE:
                    input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
                elif key == ACTION:
                    output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)
                elif key == OBS_ENV_STATE:
                    raise ValueError(
                        f"Environment state features such as {key} are not supported for this policy. "
                        f"Please remove this feature from your dataset before training."
                    )

            config.input_features = input_features
            config.output_features = output_features

        config.validate_features()
        self.config = config

        self._queues = None

        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim
        self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

        action_dim = config.action_feature.shape[0]
        horizon = config.horizon

        self.model_objective = config.model_objective
        if config.is_diffusion:
            self.objective = DiffusionObjective(
                config.get_objective_config(),
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_flow_matching:
            self.objective = FlowMatchingObjective(
                config.get_objective_config(),
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        else:
            raise ValueError(f"Unsupported model_objective: {self.model_objective}")

        self.reset()

    def get_optim_params(self) -> list:
        """Returns parameter groups with different learning rates for vision vs non-vision parameters."""
        non_vision_params = []
        vision_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "observation_encoder.vision_encoder" in name:
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.observation_encoder.vision.lr_multiplier,
            },
        ]

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        actions = self.objective.conditional_sample(self.noise_predictor, batch_size, conditioning_vec)

        start_idx = n_obs_steps - 1
        end_idx = start_idx + self.config.n_action_steps
        return actions[:, start_idx:end_idx]

    def reset(self):
        """Clear observation and action queues."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)

        if self.config.observation_encoder.text:
            self._queues["task"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        loss = self.objective.compute_loss(self.noise_predictor, batch, conditioning_vec)

        return loss, None

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given observations."""
        self.eval()

        original_batch_keys = set(batch.keys())
        new_batch = {}
        for k in self._queues:
            if k in original_batch_keys:
                if self._queues[k] and isinstance(self._queues[k][-1][0], str):
                    # for task description which is a list of strings
                    new_batch[k] = self._queues[k][-1]
                else:
                    queue_values = list(self._queues[k])
                    new_batch[k] = torch.stack(queue_values, dim=1)
        batch = new_batch

        actions = self._generate_actions(batch)
        return actions

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method manages caching of observations and actions by generating an action chunk
        and returning actions from the cache until it's depleted.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def save(self, save_directory: str | Path):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config as JSON using draccus
        with open(save_directory / "config.json", "w") as f, draccus.config_type("json"):
            draccus.dump(self.config, f, indent=4)

        # Save model weights as safetensors (matching lerobot convention)
        save_model_as_safetensor(self, str(save_directory / "model.safetensors"))

    @classmethod
    def load(cls, pretrained_model_name_or_path: str | Path):
        path = Path(pretrained_model_name_or_path)

        # Load config from JSON using draccus
        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {path}")

        with draccus.config_type("json"):
            with open(config_file, "r") as f:
                config = draccus.load(cls.config_class, f)

        model = cls(config)

        # Load model weights from safetensors (matching lerobot convention)
        model_file = path / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"model.safetensors not found in {path}")

        load_model_as_safetensor(model, str(model_file))
        return model
