# ADP

本仓库包含 ADP 项目的实验源代码，覆盖 MNIST、Fashion-MNIST、CIFAR-10、SVHN 和 Tiny-ImageNet，并包含基线、成员推断/删除推断及结果分析脚本。

仓库只保存源代码，不包含论文、实验结果、数据集或训练好的模型。

## 环境配置

建议使用 Python 3.9 或更高版本，并在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需使用 GPU，请先按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与本机 CUDA 版本匹配的 PyTorch。

## 数据集

通过环境变量指定数据集保存位置：

```bash
export MNIST_DATA_ROOT=/path/to/mnist
export FASHION_MNIST_DATA_ROOT=/path/to/fashion-mnist
export CIFAR10_DATA_ROOT=/path/to/cifar10
export SVHN_DATA_ROOT=/path/to/svhn
export TINYIMAGENET_DATA_ROOT=/path/to/tiny-imagenet-200
```

MNIST、Fashion-MNIST、CIFAR-10 和 SVHN 的主要入口会在数据缺失时通过 `torchvision` 自动下载。Tiny-ImageNet 需要自行下载并解压，其目录中应包含 `train/` 和 `val/`。

## 运行示例

所有命令都应从项目根目录执行。例如：

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

其他实验入口位于对应数据集目录下：

- `main_baseline*.py`：基线与不同删除规模实验；
- `main_ADP*.py`：ADP 相关实验；
- `main_lira.py`、`main_rmia.py`：成员推断攻击；
- `main_unsupervised_baseline.py`：无监督攻击基线；
- `VIB_unl_*.py`：不同机器遗忘方法及消融实验；
- `Results_analysis/`：本地实验结果的分析和绘图脚本。

实验会在相应数据集目录中生成结果、缓存和模型文件。这些产物已由 `.gitignore` 排除，不会提交到仓库。

## 项目结构

```text
On_MNIST/          MNIST 实验
On_FMNIST/         Fashion-MNIST 实验
On_CIFAR10/        CIFAR-10 实验
On_SVHN/           SVHN 实验
On_TinyImageNet/   Tiny-ImageNet 实验
Results_analysis/  结果分析与绘图代码
```
