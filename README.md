# TakeHome Assignment: Multiclass Lung Segmentation

A complete segmentation pipeline for the Shenzhen chest X-ray dataset, distinguishing background, left lung, and right lung using PyTorch and UNet architecture.

## Task
You are tasked with building a complete segmentation pipeline for the Shenzhen chest Xray dataset, distinguishing background, left lung, and right lung. Your solution should live in a single Jupyter notebook that can be run from top to bottom without modification.

Begin by loading the raw images and their binary lung masks, then transform those masks into a threeclass format. Use a clear, reproducible data pipeline—include any resizing, normalization, or augmentation steps you deem helpful for model generalization. Implement a custom PyTorch Dataset and DataLoader to feed paired images and multiclass masks into your network.

Define a UNet (or equivalent encoderdecoder) in PyTorch, with configurable depth and featuremap sizes via parameters at the top of your notebook (you can use already implemented models for fine tunning). While training, log training and validation metrics each epoch and plot curves to demonstrate convergence.

After training, evaluate on a heldout set and visualize a handful of cases: show the input, groundtruth mask, and your model's prediction side by side with color coding for each class. Save your best model checkpoint and include a short inference cell that loads the checkpoint, runs prediction on a new image, and overlays the segmentation mask on the Xray.

Your notebook should open with a concise configuration section—paths, hyperparameters, random seeds—and end with clear instructions on how to rerun all cells. Document any design choices, experimental variations, or extra features (for example, additional augmentations or a combined loss function) in markdown cells. Strive for clean, modular code organized into functions or class definitions, and ensure that anyone can clone your notebook, install dependencies, and reproduce your results in one go.


## Features

- **Multiclass Segmentation**: Background, left lung, and right lung classification
- **UNet Architecture**: Configurable encoder-decoder with ResNet backbone
- **Data Pipeline**: Custom PyTorch Dataset with transforms and augmentation
- **Training Pipeline**: Complete training with validation, metrics, and checkpointing
- **Visualization**: Side-by-side comparison of input, ground truth, and predictions
- **Configuration**: Hydra-based configuration management
- **Jupyter Support**: Interactive notebook for experimentation

## Quick Start

### Docker (Recommended - Main Way)

The easiest way to run this project is using Docker with GPU support:

```bash
# Build and run Docker container
make docker

# Or step by step:
make docker-build
make docker-run
```

This will:
- Build a Docker image with CUDA 12.4.1 support
- Mount your current directory to the container
- Provide GPU access for training
- Install all dependencies automatically

### Local Development

#### Prerequisites
- Python 3.12+
- UV package manager

#### Installation
```bash
# Install dependencies
uv sync
```

### Usage

**Command Line:**
```bash
# Train model
uv run run.py mode=train

# Test model
uv run run.py mode=test
```

**Jupyter Notebook:**
```bash
# Start Jupyter Lab
uv run jupyter-lab --port 8000
# Open jupyter_lab/run.ipynb
```

## Monitoring

**TensorBoard:**
```bash
tensorboard --logdir logs
# Visit http://localhost:6006/
```
