# Active Deletion Probing (ADP)

This repository contains the experimental source code for the ADP project. It covers MNIST, Fashion-MNIST, CIFAR-10, SVHN, and Tiny-ImageNet, including baselines, membership/deletion inference attacks, and result-analysis scripts.

Only source code is included in this repository. Papers, experimental results, datasets, and trained models are not included.

## Environment Setup

Python 3.9 or later is recommended. Run the following commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU support, first install a PyTorch build compatible with your local CUDA version by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Datasets

Use the following environment variables to specify dataset locations:

```bash
export MNIST_DATA_ROOT=/path/to/mnist
export FASHION_MNIST_DATA_ROOT=/path/to/fashion-mnist
export CIFAR10_DATA_ROOT=/path/to/cifar10
export SVHN_DATA_ROOT=/path/to/svhn
export TINYIMAGENET_DATA_ROOT=/path/to/tiny-imagenet-200
```

The primary entry points automatically download MNIST, Fashion-MNIST, CIFAR-10, and SVHN through `torchvision` when the datasets are missing. Tiny-ImageNet must be downloaded and extracted manually. Its root directory should contain the `train/` and `val/` directories.

## Running the Code

Run all commands from the project root. Examples:

```bash
# MNIST
python On_MNIST/main_baseline.py

# Fashion-MNIST
python On_FMNIST/main_baseline.py

# CIFAR-10
python On_CIFAR10/main_baseline.py

# SVHN
python On_SVHN/main_baseline_for_different_unl_samples.py

# Tiny-ImageNet
python On_TinyImageNet/main_baseline.py
```

Additional experiment entry points are located in the corresponding dataset directories:

- `main_baseline*.py`: baselines and experiments with different deletion sizes;
- `main_ADP*.py`: ADP experiments;
- `main_lira.py` and `main_rmia.py`: membership inference attacks;
- `main_unsupervised_baseline.py`: unsupervised attack baselines;
- `VIB_unl_*.py`: machine-unlearning methods and ablation experiments;
- `Results_analysis/`: scripts for analyzing and plotting locally generated results.

Experiments generate results, caches, and model files inside the corresponding dataset directories. These artifacts are excluded by `.gitignore` and will not be committed.

## Project Structure

```text
On_MNIST/          MNIST experiments
On_FMNIST/         Fashion-MNIST experiments
On_CIFAR10/        CIFAR-10 experiments
On_SVHN/           SVHN experiments
On_TinyImageNet/   Tiny-ImageNet experiments
Results_analysis/  Result analysis and plotting code
```
