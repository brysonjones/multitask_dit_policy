# multitask_dit_policy
An open-source implementation of Multitask Diffusion-Transformer (DiT) Policy for robot manipulation

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

## Overview
The goal of this project is to provide the community with a high quality implementation of Multitask DiT Policy that
can be used as a baseline for model research and building on top of.

![Multitask DiT Policy Architecture](assets/mtdp_architecture.png)
*Architecture diagram of the Multitask Diffusion-Transformer (DiT) Policy model*

I've made an effort to make the implementations as readable as possible, at the sacrifice of making the most
optimized implementations, specifically with regards to the training loop.

For a deep dive on technical details of the model, see the blog-post [here](https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy)

## System Requirements

### GPU Requirements

- **Inference**: At least an RTX 5070 Ti (or equivalent GPU) is recommended for running inference with reasonable speed performance.
- **Training**: A GPU with at least 24GB of VRAM is recommended for training, as I would target batch sizes over 128 if possible for training stability.


## Environment Setup

### Installation

This project uses uv for python environment management. Install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the pinned python version and install the package:

```bash
uv python install
uv sync
```

### Environment Variables

Set the following environment variables before training:

- `WANDB_API_KEY` - Optional, for Weights & Biases logging. Training will proceed without it if not set.
- `HUGGINGFACE_TOKEN` - Required for using LeRobotDataset

Add to your `~/.bashrc`:

```bash
echo 'export WANDB_API_KEY={{your_wandb_key}}' >> ~/.bashrc  # Optional
echo 'export HUGGINGFACE_TOKEN={{your_hf_token}}' >> ~/.bashrc
source ~/.bashrc
```

## Dataset

I have built this implementation around the LeRobotDataset format from the [LeRobot](https://github.com/huggingface/lerobot/tree/main/src/lerobot) project.

To see details on this format see [LeRobotDataset](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

To train, you will need a dataset in this format available locally.

If you don't have a LeRobotDataset yet, you can use a toy dataset provided by HuggingFace:
```
hf download lerobot/pusht --repo-type dataset --local-dir /your/local/dir/pusht
```

NOTE: I intentionally did ***not*** add the ability to pull LeRobotDatasets from the HF hub and instead require all datasets to be locally available (unless you use Modal training, where the datasets should be stored on a Modal volume as described below). If this capability is of interest to you, please create an issue.


## Local Training

The project uses draccus for config management. Pass arguments using `--key=value` syntax:

```bash
uv run -m multitask_dit_policy.train \
    --dataset_path=/path/to/dataset \
    --batch_size=16 \
    --train_steps=2000 \
    --device=cuda \
    --output_dir=outputs/train_multi_task_dit \
```

To see the full list of configuration options, run:
```bash
uv run -m multitask_dit_policy.train --help
```

### Resuming Training

To resume training from a checkpoint, use the `--checkpoint_path` parameter to specify the path to a previously saved checkpoint directory:

```bash
uv run -m multitask_dit_policy.train \
    --dataset_path=/path/to/dataset \
    --checkpoint_path=outputs/train_multi_task_dit/checkpoint_1000 \
    --batch_size=16 \
    --train_steps=2000 \
    --device=cuda \
    --output_dir=outputs/train_multi_task_dit_resumed
```

**Notes:**
- The checkpoint directory should contain `model.safetensors` and `config.json` files (saved automatically during training)
- You can use the same `--output_dir` or specify a new one to avoid overwriting previous checkpoints
- The model weights will be loaded from the checkpoint, but training will start from step 0 (the step counter resets)
- Ensure you use the same dataset and compatible configuration settings as the original training run

NOTE: If you are using the toy `pusht` dataset, the images will be below the default crop shape of (224, 224) for CLIP, and you will need to resize the images using:
```
--policy.observation_encoder.vision.type=clip \
--policy.observation_encoder.vision.resize_shape='[224,224]' 
```

## Cloud Training Using Modal

Modal has a great developer experience, especially when you're just doing small training experiments up to 8 GPUs. I've added a simple script that will deploy training jobs onto modal with specified GPU resources.

> **⚠️ NOTE:** Compared to some GPU providers, Modal's prices can be noticeably higher (sometimes >1.5x the commodity price or more). Please budget accordingly and check costs before launching long jobs!

Below is an overview of how you can use scripts to train a policy on modal

### Setting up Modal

Sign up for an account [here](https://modal.com/)

Install Modal CLI and authenticate:

```bash
uv sync --extra modal
modal token new
```

### Creating a Volume

Create a new Modal volume:

```bash
modal volume create multitask_dit_data  # Note you can replace `multitask_dit_data` with the volume name of your choice
```

### Uploading a Dataset

For Modal training, you'll need to upload your dataset to a Modal volume first:

```bash
# Upload dataset to Modal volume (replace 'multitask_dit_data' with your volume name if different)
modal volume put multitask_dit_data /path/to/local/dataset /path/on/volume
```

### Modal Training

The modal training configuration parameters extend the local training config, allowing you to specify compute parameters such as GPU type, # of CPUs, and System RAM. For a complete list, please see the [configuration definition](./src/multitask_dit_policy/train_modal.py)

```bash
uv run -m multitask_dit_policy.train_modal \
    --dataset_path=modal/path/to/dataset \
    --batch_size=320 \
    --train_steps=2000 \
    --output_dir=training_runs/train_multi_task_dit \
    --num_workers=10 \
    --device=cuda \
    --gpu_type=B200 \
    --use_amp=true \
    --timeout_hours=10
```

**Note:** When specifying the dataset `root` with Modal, specify the path relative to `/data_volume` (e.g., `datasets/my_dataset`). The training script will automatically prepend `/data_volume` to your root path, so it becomes `/data_volume/datasets/my_dataset`.

To run in detached mode which will keep the training job running if the terminal session closes/ends, use:
```
--detach=true
```

## Baseline Configuration and Tuning

This section provides suggested default configuration parameters and common tuning points for both training and inference.

### Suggested Default Hyperparameters (Assuming 30Hz control frequency)
```
- Batch Size: 256-320 -> You will need a larger GPU likely on the cloud for this, but it will result in the best performance
- Horizon: 32
- # Number of action steps: 24
- Objective: Diffusion
- # of traing steps: 30k steps
```

### Training Tuning Points

- **Flow Matching with Beta Sampling Distribution**: Consider switching to flow matching with beta sampling distribution for potentially improved performance. This hasn't been shown to be a silver bullet in any experiments I've seen, but it occasionally results in smoother and more consistent actions.

- **Number of Transformer Layers**: The model's capacity should match your dataset size:
  - **Small datasets** (< 100 examples): Try reducing the number of layers to 4
  - **Large datasets** (> 5k examples): Try increasing to 8 layers

- **Horizon**: The model can be sensitive to the horizon you choose. Start with around a 1 second horizon based on your control frequency (`horizon=30` for a 30 Hz frequency), then experiment with increasing from there. The horizon determines how far into the future the model predicts actions.

### Inference Tuning Points

- **Diffusion Sampling**: For faster inference, use DDIM with 10 sampling steps instead of the default settings.

- **n_action_steps**: The model can be very sensitive to `n_action_steps`. Start with it being around 0.8 seconds based on your control frequency and tune from there. There's a trade-off between reactiveness and long-horizon task execution and stability:
  - **Lower values**: More reactive but potentially less stable for long-horizon tasks
  - **Higher values**: Better for long-horizon execution but may be less responsive

## Running Inference

The project includes an inference script that loads a trained model checkpoint and runs inference on a single random sample from your dataset, displaying the predicted actions and observations.

This is to simply demonstrate how you would set up an inference loop if you wanted to integrate this policy into your own project.

### Basic Usage

```bash
uv run -m multitask_dit_policy.examples.inference \
    --checkpoint_dir=outputs/train_multi_task_dit_test_1/final_model \
    --dataset_path=/path/to/dataset \
    --device=cuda
```

### Configuration Options

- `--checkpoint_dir` - **Required**. Path to the checkpoint directory containing `model.safetensors`, `config.json`, and `dataset_stats.json`
- `--dataset_path` - **Required**. Path to the LeRobotDataset directory

### Example

```bash
uv run -m multitask_dit_policy.examples.inference \
    --checkpoint_dir=outputs/train_multi_task_dit_test_1/final_model \
    --dataset_path=/your/local/dir/pusht \
    --device=cuda \
```

The script will:
1. Load the model from the specified checkpoint directory
2. Load dataset statistics for proper normalization
3. Select a random sample from the dataset
4. Run inference to generate predicted actions
5. Display a visualization showing the input images (if available) and the predicted action trajectory

**Note:** The checkpoint directory must contain `dataset_stats.json` for proper action normalization. This file is automatically saved during training.

## Common Failure Modes and Debugging

Training these models can be finicky (as is all AI research...)

Here are some common failure modes I've seen when training this particular model, and approaches to debugging

### Idling / No Motion

In some cases, you may train the model and during inference see its outputs "collapse", resulting in static or no motion. This collapse can occur at the starting point mid-way through a task.

My intuition is this happens when the tasks or training data is especially multi-modal, and based on the observations the policy oscillates in its actions around some average output.

This appears to happen in two specific cases:
- When you don't have enough training data for your task. If you only have 20-50 examples, try to roughly double your dataset size and try again for the same task. Once you have above 300 examples or so for a single task, if you are still seeing this, the task may be too complex, or have some part of the task that's unobservable that is causing the issue.
- When your dataset contains multiple similar tasks. An example would be picking up and moving 2 different objects. While the object is different, the model is heavily relying on the language conditioning which might not be rich enough to give the model a strong differentiation in the actions it should take.

**Debugging tips:**
- Increase dataset size (double until you get to over 300 examples)
- Train for longer, and up to 100k steps, even when the loss flatlines
- Check if the model is receiving proper language instructions or increase diversity of instruction

### Executing the Wrong Task

Sometimes the robot will completely ignore your instruction and perform some other task. This generally will only happen if you have trained on multiple tasks

**Potential causes:**
- Language instruction ambiguity
- Insufficient task-specific training data
- Model confusion between similar tasks in the multitask dataset

**Debugging tips:**
- Verify language instruction clarity and specificity
- Check task distribution in your training dataset and add weighting to the failing/ignored task
- Consider task-specific fine-tuning 

## Contributing

Contributions, improvements, and bug fixes are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup instructions and guidelines.

## Acknowledgements and References

Many utility functions were adapted from LeRobot to build this project. Additionally the base structure of the policy was inspired by the LeRobot Vanilla Diffusion Policy implementation, with most interfaces remaining identical to simplify downstream integration into the LeRobot project.

The integration into LeRobot can be found [here](https://github.com/huggingface/lerobot/pull/2545)

> **⚠️ NOTE:** The LeRobot integration is currently in an active merge request that is being worked on to be merged into main.

Additionally, the following resources were referenced during this implementation:

```bibtex
@misc{bostondynamics2025lbm,
  author = {Eric Cousineau and Scott Kuindersma and Lucas Manuelli and Pat Marion and Boston Dynamics and Toyota Research Institute},
  title = {Large Behavior Models and Atlas Find New Footing},
  year = {2025},
  url = {https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/},
  note = {Blog post}
}

@misc{trilbmteam2025carefulexaminationlargebehavior,
      title={A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation}, 
      author={TRI LBM Team},
      year={2025},
      eprint={2507.05331},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.05331}, 
}

@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## Cite This Work
If you use this work in your research, please cite:

```bibtex
@misc{jones2025multitaskditpolicy,
  author = {Bryson Jones},
  title = {Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy},
  year = {2025},
  url = {https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy},
  note = {Blog post}
}

@software{jones2025multitaskditpolicyrepo,
  author = {Bryson Jones},
  title = {multitask_dit_policy: An Open-Source Implementation of Multitask Diffusion-Transformer Policy for Robot Manipulation},
  year = {2025},
  url = {https://github.com/brysonjones/multitask_dit_policy},
  note = {Software}
}
```
