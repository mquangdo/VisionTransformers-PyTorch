# VisionTransformers-PyTorch

PyTorch implementation of a Vision Transformer (ViT) for classification on a synthetic MNIST + texture dataset.

## Overview

This repository contains:

- A ViT model built from core components (patch embedding, multi-head attention, transformer blocks).
- A custom dataset that overlays colored MNIST digits on texture backgrounds.
- Training and inference utilities.
- Attention and positional embedding visualization helpers.

## Repository Structure

```text
VisionTransformers-PyTorch/
├── config/default.yaml                 # Dataset, model, and training configuration
├── dataset/mnist_color_texture_dataset.py
├── model/
│   ├── patch_embedding.py
│   ├── attention.py
│   ├── transformers_layer.py
│   └── transformers.py
└── tools/
    ├── train.py
    └── inference.py
```

## Requirements

Install Python dependencies:

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm pyyaml einops
```

> Note: this repository does not currently include a pinned `requirements.txt`.

## Data

The dataset loader expects a root directory (configured by `dataset_params.root_dir`) with:

- `imdb.json`
- digit image files referenced by `digit_image`
- texture image files referenced by `texture_image`

`imdb.json` is expected to include at least:

- `train_data`
- `test_data`
- `texture_classes_index`

Each data entry should include keys such as:
`digit_name`, `digit_image`, `texture_name`, `texture_image`, `color_r`, `color_g`, `color_b`.

## Configuration

Default config is in:

`/tmp/workspace/mquangdo/VisionTransformers-PyTorch/config/default.yaml`

It defines:

- `dataset_params` (data root)
- `model_params` (patch size, embedding size, number of layers/heads, classes)
- `train_params` (batch size, epochs, learning rate, checkpoint path, seed)

## Training

The training utility is implemented in:

`/tmp/workspace/mquangdo/VisionTransformers-PyTorch/tools/train.py`

Primary entrypoint function:

- `train(args)` where `args.config_path` points to a YAML config.

## Inference

The inference utility is implemented in:

`/tmp/workspace/mquangdo/VisionTransformers-PyTorch/tools/inference.py`

Primary entrypoint function:

- `inference(args)` where `args.config_path` points to a YAML config.

Inference includes:

- Classification accuracy computation
- Positional embedding cosine-similarity visualization (`output/position_plot.png`)
- Attention rollout overlays (`output/input_*.png`, `output/overlay_*.png`)

## Notes

- Checkpoints are saved under `train_params.task_name` using `train_params.ckpt_name`.
- Output visualizations are written to the `output/` directory.
