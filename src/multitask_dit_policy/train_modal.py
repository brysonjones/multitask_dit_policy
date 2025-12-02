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

import os
import warnings
from dataclasses import dataclass

import modal

from multitask_dit_policy.train import TrainConfig, train

# Suppress Pydantic warnings from draccus ChoiceRegistry union types
# This is an interaction with draccus that we can't control
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


@dataclass
class ModalTrainConfig(TrainConfig):
    """Training config for Modal cloud execution.

    Inherits all parameters from TrainConfig and adds Modal-specific options.
    """

    # Modal-specific options
    detach: bool = False  # If True, detach from CLI and run in background on Modal

    # Modal resource configuration
    gpu_type: str = "A10"
    timeout_hours: float = 10.0  # Timeout in hours (converted to seconds internally)
    num_cpus: float = 16.0
    memory_amount: int = 65536  # in MB
    modal_volume_name: str = "multitask_dit_data"

    def __post_init__(self):
        """Validate GPU type is one of the allowed values."""
        valid_gpu_types = {
            "T4",
            "L4",
            "A10",
            "A100",
            "A100-40GB",
            "A100-80GB",
            "L40S",
            "H100",
            "H100!",
            "H200",
            "B200",
        }
        if self.gpu_type not in valid_gpu_types:
            raise ValueError(
                f"Invalid gpu_type: {self.gpu_type}. " f"Must be one of: {', '.join(sorted(valid_gpu_types))}"
            )


def create_modal_app_and_function(cfg: ModalTrainConfig):
    """
    Create Modal app and function with config-based compute parameters
    """
    # set up environment variables
    if "HUGGINGFACE_TOKEN" not in os.environ:
        raise ValueError(
            "HUGGINGFACE_TOKEN is not set. This is needed to use LerobotDatasets. \
        Please set it in your environment variables and try again."
        )

    env_vars = {
        "HF_LEROBOT_HOME": "/data_volume/lerobot",
        "HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"],
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    dockerfile_image = modal.Image.from_dockerfile("Dockerfile", force_build=False).env(env_vars)

    app = modal.App("multitask-dit-training")
    modal_volume = modal.Volume.from_name(cfg.modal_volume_name, create_if_missing=True)

    timeout_seconds = int(cfg.timeout_hours * 3600)  # modal needs seconds

    @app.function(
        cpu=cfg.num_cpus,
        memory=cfg.memory_amount,
        gpu=cfg.gpu_type,
        image=dockerfile_image,
        volumes={"/data_volume": modal_volume},
        timeout=timeout_seconds,
        serialized=True,
    )
    def train_policy(train_cfg: TrainConfig):
        """Train the multitask DiT policy using Modal.

        This function is called remotely by the local entrypoint.
        """
        # Suppress Pydantic warnings from draccus ChoiceRegistry union types
        # This must be done FIRST before any imports that trigger draccus/pydantic
        # This is an interaction with draccus that we can't control
        import warnings

        warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
        warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
        warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

        import logging
        from pathlib import Path

        import huggingface_hub

        # Configure logging to output to stderr with proper format
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,  # Override any existing configuration
        )

        huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])

        volume_name = cfg.modal_volume_name

        # Prepend /data_volume to dataset_path directory
        data_volume_dataset_path = Path("/data_volume") / train_cfg.dataset_path.lstrip("/")
        if not data_volume_dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_volume_dataset_path}\n"
                f"Please ensure the dataset is uploaded to the Modal volume at this path.\n"
                f"To upload data to the Modal volume, use:\n"
                f"  modal volume put {volume_name} <local_path> {train_cfg.dataset_path}\n"
                f"Or mount the volume and copy files directly."
            )
        train_cfg.dataset_path = str(data_volume_dataset_path)

        # Prepend /data_volume to output directory
        data_volume_output = Path("/data_volume") / train_cfg.output_dir.lstrip("/")
        if data_volume_output.exists():
            raise FileExistsError(
                f"Output directory already exists: {data_volume_output}\n"
                f"This would overwrite existing checkpoints. Please either:\n"
                f"  1. Use a different output directory (specify --output_dir=<new_path>)\n"
                f"  2. Delete the existing directory from the Modal volume:\n"
                f"     modal volume rm {volume_name} {train_cfg.output_dir}\n"
                f"  3. Or mount the volume and delete it manually."
            )
        train_cfg.output_dir = str(data_volume_output)

        logging.info("Starting training with config:")
        logging.info(f"  Dataset path: {train_cfg.dataset_path}")
        logging.info(f"  Batch size: {train_cfg.batch_size}")
        logging.info(f"  Train steps: {train_cfg.train_steps}")
        logging.info(f"  Output dir: {train_cfg.output_dir}")
        logging.info(f"  Device: {train_cfg.device}")
        if train_cfg.checkpoint_path:
            logging.info(f"  Resuming from checkpoint: {train_cfg.checkpoint_path}")

        # reuse the same train function used for local training
        train(train_cfg)

        logging.info("Training completed successfully")

    return app, train_policy


@draccus.wrap()
def main(cfg: ModalTrainConfig):
    app, train_policy = create_modal_app_and_function(cfg)

    with modal.enable_output():
        with app.run(detach=cfg.detach):
            train_policy.remote(cfg)


if __name__ == "__main__":
    main()
