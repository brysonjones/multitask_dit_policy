# multitask_dit_policy
An open-source implementation of Multitask Diffusion-Transformer (DiT) Policy for robot manipulation

## Overview
The goal of this project is to provide the community with a high quality implementation of Multitask DiT Policy that
can be used as a baseline for model research and building on top of.

I have made an effort to make the implementations as readable as possible, at the sacrifice of making the most
optimized implementations, specifically with regards to the training loop.

For a deep dive on technical details of the model, see [here](TODO)

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

## Environment Variables

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

To train this model, you will need a dataset in this format available locally.

If you don't have a LeRobotDataset yet, you can use a toy dataset provided by HuggingFace:
```
hf download lerobot/pusht --repo-type dataset --local-dir /your/local/dir/pusht
```

NOTE: I intentionally didn't add the ability to pull datasets from the HF hub with the interface I've implemented, as I think it adds unneccessary complexity and distracts from the simplicity I aim for with this project. If this capability is of interest to you, please create an issue.


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

NOTE: If you are using the toy `pusht` dataset, the images will be below the default crop shape of (224, 224) for CLIP, and you will need to resize the images using:
```
--policy.observation_encoder.vision.type=clip \
--policy.observation_encoder.vision.resize_shape='[224,224]' 
```

## Cloud Training Using Modal

Modal has a great developer experience, especially when you're just doing small training experiments up to 8 GPUs. I've added a simple script that will deploy training jobs onto modal with specified GPU resources.

NOTE: compared to some GPU providers, Modal's prices can be a bit steeper (>1.5x the commodity price or more), so be aware of that.

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

The modal training configuration parameters extend the local training config, allowing you to specfiy compute parameters such as GPU type, # of CPUs, and System RAM. For a complete list, please see the [configuration definition](./src/multitask_dit_policy/train_modal.py)

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


## Contributing

Contributions, improvements, and bug fixes are welcome! Please feel free to submit bug reports, feature requests, and pull requests. This project is open to everyone in accordance with the license provided in the repo.


## Acknowledgements and References

Many utility functions were adapted from LeRobot to build this project. Additionially the base structure of the policy was inspired by the LeRobot Vanilla Diffusion Policy implementation, with most interfaces remaining identical to simplify downstream integration into the lerobot project.

The integration into LeRobot can be found [here](TODO)

Additionally, the following resources were referenced during this implementation:

```bibtex
@misc{bostondynamics2025lbm,
  author = {Eric Cousineau and Scott Kuindersma and Lucas Manuelli and Pat Marion and Boston Dynamics and Toyota Research Institute},
  title = {Large Behavior Models and Atlas Find New Footing},
  year = {2025},
  url = {https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/},
  note = {Blog post}
}
```

```bibtex
@misc{trilbmteam2025carefulexaminationlargebehavior,
      title={A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation}, 
      author={TRI LBM Team},
      year={2025},
      eprint={2507.05331},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.05331}, 
}
```

## Cite This Work

```bibtex
@misc{jones2025multitaskditpolicy,
  author = {Bryson Jones},
  title = {Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy},
  year = {2025},
  url = {https://brysonkjones.substack.com/p/dissecting-multitask-diffusion-transformer-policy},
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
