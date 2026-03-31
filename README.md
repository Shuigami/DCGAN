# Autoformation-5: DCGAN MNIST Implementations

This project contains educational implementations of **Deep Convolutional Generative Adversarial Networks (DCGAN)** applied to the MNIST dataset of handwritten digits. It provides side-by-side implementations in two popular deep learning frameworks to demonstrate their respective APIs and performance.

## Overview

DCGAN is a powerful architecture for generative modeling that uses convolutional layers in both the Generator and the Discriminator. This project follows the architecture proposed by Radford et al. in "[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)".

### Features

- **PyTorch Implementation**: Located in `DCGAN_pytorch.py`.
- **Keras/TensorFlow Implementation**: Located in `DCGAN.py`.
- **Data Handling**: Automatically downloads and processes the MNIST dataset.
- **Visualization**: Includes scripts to visualize generated samples during and after training.

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

Note: The PyTorch dependencies are included in the `pyproject.toml`. If you wish to run the Keras version, you may need to install TensorFlow manually:

```bash
uv add tensorflow
```

## Usage

### Running the PyTorch version

```bash
uv run python DCGAN_pytorch.py
```

### Running the Keras version

```bash
uv run python DCGAN.py
```

## Architecture

- **Generator**: Uses Transposed Convolution layers to upscale a latent noise vector into a 28x28 grayscale image.
- **Discriminator**: A standard Convolutional Neural Network (CNN) that learns to classify images as "real" (from MNIST) or "fake" (from the Generator).

## References

1. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
