# Pneumonia Detection Using Anchor-Free Object Detection

This repository contains the implementation of an anchor-free pneumonia
detection system for chest X-ray images.

The project was developed as part of the Numerical Analysis for Machine
Learning course at Politecnico di Milano. The main objective is to study
pneumonia localization using an FCOS-style object detector with ResNet
backbones.

## Project Overview

The project combines two main lines of work.

The application context is inspired by the work of Wu et al., which
investigates pneumonia detection and localization on the RSNA Pneumonia
Detection Challenge dataset using an anchor-free detection framework.

The detector implemented in this repository instead follows the FCOS
formulation more directly. The resulting system combines the RSNA
pneumonia detection problem with the FCOS anchor-free detection paradigm.

The objective is not to reproduce the reference implementation as a black
box, but to implement and study an FCOS-style detector specifically for
pneumonia localization.

The experimental study compares two backbone depths and two initialization
strategies:

| Model | Backbone | Initialization |
|---|---|---|
| `R50-IN` | ResNet-50 | ImageNet |
| `R50-CX` | ResNet-50 | Chest-Xray |
| `R101-IN` | ResNet-101 | ImageNet |
| `R101-CX` | ResNet-101 | Chest-Xray |

All four detector configurations are trained and evaluated on the RSNA
Pneumonia Detection Challenge dataset.

The Chest X-Ray Images (Pneumonia) dataset is used separately for
domain-specific pretraining of the ResNet backbone.

## Main Features

### FCOS-Style Anchor-Free Detection

The detector does not use predefined anchor boxes.

Predictions are generated directly from spatial locations of the feature
maps. For each location, the detector predicts:

- a pneumonia classification score;
- four bounding-box distances `(l, t, r, b)`;
- a centerness score.

### Multi-Scale Feature Pyramid

The ResNet backbone provides hierarchical feature representations at
different spatial resolutions.

The detector uses `C3`, `C4` and `C5` to construct an FCOS-style feature
pyramid with levels `P3` through `P7`.

### Target Assignment

Training targets follow an FCOS-style location-based formulation.

For each feature-map location, the target generation procedure determines
whether the location is associated with a ground-truth bounding box and,
when positive, computes:

- the positive-location mask;
- `(l, t, r, b)` regression targets;
- the centerness target.

Different regression ranges are assigned to the different FPN levels.

### Detection Loss

The training objective combines three components:

- Sigmoid Focal Loss for classification;
- Generalized IoU Loss for bounding-box regression;
- Sigmoid Focal Loss for centerness.

The total objective is:

    L = L_cls + L_reg + L_ctr

with equal weights for the three components.

The use of Focal Loss for centerness is a project-specific modification of
the original FCOS formulation.

### Transfer Learning

Two backbone initialization strategies are supported:

    ImageNet pretrained weights
    Chest-Xray pretrained weights

The Chest-Xray initialization is obtained through a separate classification
pretraining stage before detector training on the RSNA dataset.

### Training Stabilization

The training pipeline supports:

- backbone freezing;
- differential learning rates;
- learning-rate warm-up;
- cosine annealing;
- gradient clipping;
- Exponential Moving Average (EMA);
- validation-based checkpoint selection.

Both a standard trainer and an extended `TrainerV2` implementation are
provided.

### Evaluation and Analysis

The project provides utilities for:

- Average Precision (AP);
- Average Recall at 10 detections (AR@10);
- image-level precision;
- image-level recall;
- specificity;
- F1-score;
- Youden's J;
- mean matched IoU;
- box-level precision, recall and F1.

For image-level evaluation, the operating threshold can be calibrated on
the validation set using Youden's J statistic.

### Visualization

The repository provides tools for:

- confusion matrices;
- prediction visualization;
- per-image detection results;
- feature-flow visualization;
- qualitative comparison of predictions;
- analysis of prediction disagreements.

Two visualization modes are available.

The standard mode uses the model-specific calibrated threshold.

A dedicated diagnostic mode uses a fixed threshold of `0.10` and disables
NMS, redundancy suppression and the detection cap. It is intended only for
qualitative inspection.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python environment
and dependency management.

### Install uv

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

### Clone the Repository

```bash
git clone https://github.com/Francesco2035/fcos-pneumonia-detection-naml-25-26.git
cd fcos-pneumonia-detection-naml-25-26
```

### Install Dependencies

The repository contains both `pyproject.toml` and `uv.lock`.

Create the environment and install the locked dependencies with:

```bash
uv sync
```

Run project commands through the managed environment with:

```bash
uv run python main.py --help
```

## Dataset Setup

The medical datasets are not included in the repository.

The project uses two datasets for different purposes.

### RSNA Pneumonia Detection Challenge

The RSNA Pneumonia Detection Challenge is the main dataset for detector
training, validation and evaluation.

Official source:

https://www.rsna.org/challenge-datasets/2018

The RSNA data should be stored locally under the project's `data/`
directory.

### Chest X-Ray Images (Pneumonia)

The Chest X-Ray Images (Pneumonia) dataset is used only for domain-specific
backbone pretraining.

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The expected location is:

```text
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

The exact RSNA paths are controlled by `src/config.py`.

## Downloading the Chest X-Ray Dataset from Kaggle

Install the Kaggle CLI as an isolated tool using uv:

```bash
uv tool install kaggle
```

Verify the installation:

```bash
kaggle --version
```

Authenticate with Kaggle:

```bash
kaggle auth login
```

Create the data directory:

```bash
mkdir -p data
```

Download the dataset:

```bash
kaggle datasets download \
    -d paultimothymooney/chest-xray-pneumonia \
    -p data/
```

Extract the archive:

```bash
unzip data/chest-xray-pneumonia.zip -d data/
```

After extraction, ensure that the pretraining dataset is available as:

```text
data/chest_xray/
```

The Kaggle CLI also supports authentication through an API token.

## Running the Project

The main entry point is:

```text
main.py
```

The command-line interface provides four execution modes:

```text
pretrain
train
analyze
visualize
```

Run:

```bash
uv run python main.py --help
```

to display all available options.

## Pretraining

Chest-Xray pretraining can be performed before detector training.

Example:

```bash
uv run python main.py \
    --mode pretrain \
    --pretrain-architecture 50 \
    --pretrain-data-dir data/chest_xray \
    --pretrain-output-dir checkpoints/pretrain
```

The pretraining interface supports configurable architecture, image size,
epochs, batch size, learning rate, weight decay, workers, random seed and
optional automatic mixed precision.

## Detector Training

### ImageNet Initialization

```bash
uv run python main.py \
    --mode train \
    --experiment resnet50_imagenet \
    --backbone imagenet \
    --resnet-depth 50
```

### Chest-Xray Initialization

```bash
uv run python main.py \
    --mode train \
    --experiment resnet50_chestxray \
    --backbone chest_xray \
    --resnet-depth 50
```

The training interface also supports:

```text
--epochs
--lr
--batch-size
--freeze-resnet
--trainer
--warmup-epochs
--backbone-lr-factor
--ema-decay
--weight-decay
--resume
--load-weights
--load-backbone-weights
```

Run:

```bash
uv run python main.py --help
```

for the complete list of parameters.

## Analysis

After training, a detector checkpoint can be evaluated using:

```bash
uv run python main.py \
    --mode analyze \
    --checkpoint path/to/checkpoint.pt \
    --backbone imagenet \
    --resnet-depth 50
```

The analysis pipeline performs:

1. threshold calibration;
2. validation inference;
3. metric computation;
4. per-image result storage;
5. confusion-matrix generation;
6. prediction visualization;
7. feature-flow visualization.

## Visualization

### Standard Visualization

The standard mode uses the calibrated model-specific threshold:

```bash
uv run python main.py \
    --mode visualize \
    --checkpoint path/to/checkpoint.pt \
    --backbone imagenet \
    --resnet-depth 50 \
    --visualization-mode youden
```

### Diagnostic Visualization

For qualitative inspection of candidate detections:

```bash
uv run python main.py \
    --mode visualize \
    --checkpoint path/to/checkpoint.pt \
    --backbone imagenet \
    --resnet-depth 50 \
    --visualization-mode no_th
```

The `no_th` mode uses a fixed threshold of `0.10` and disables NMS,
redundancy suppression and the detection cap.

This mode must not be used for quantitative evaluation.

## Repository Structure

```text
.
├── main.py
├── pretrain.py
├── submit.py
├── plot.py
├── dataset_analysis.py
├── compare_prediction_differences.py
│
├── pyproject.toml
├── uv.lock
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── calibration.py
│   ├── config.py
│   ├── detection_loss.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── metrics.py
│   ├── train.py
│   ├── train_v2.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── chest_xray.py
│   │   ├── DICOMDataset.py
│   │   ├── RSNAPneumoniaDataset.py
│   │   ├── split.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   ├── detection_head.py
│   │   ├── detector.py
│   │   ├── fpn.py
│   │   ├── resnet.py
│   │   └── target_generator.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── comparison.py
│   │   ├── geometry.py
│   │   ├── io.py
│   │   └── visualizer.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py
│       └── visualizer.py
│
├── data/
│   ├── rsna/
│   └── chest_xray/
│
├── checkpoints/
├── experiments/
├── visualization/
├── plots/
├── results/
├── csv/
├── dataset_analysis/
│
└── docs/
    ├── notes/
    ├── papers/
    ├── presentation/
    └── report/
```

## Main Components

### `main.py`

Central command-line entry point for:

- pretraining;
- detector training;
- analysis;
- visualization.

It also provides command-line control over backbone selection, checkpoint
loading, training parameters, thresholding and visualization.

### `pretrain.py`

Contains the Chest-Xray classification pretraining procedure used to obtain
domain-specific ResNet initialization weights.

### `src/models/`

Contains the main detector components:

- `resnet.py`: ResNet implementation;
- `backbone.py`: backbone and feature extraction utilities;
- `fpn.py`: Feature Pyramid Network;
- `detection_head.py`: FCOS-style detection heads;
- `target_generator.py`: target assignment;
- `detector.py`: complete detector framework.

### `src/datasets/`

Contains dataset loaders, splitting utilities and image transformations
for the RSNA and Chest-Xray datasets.

### `src/train.py` and `src/train_v2.py`

Contain the training implementations.

`TrainerV2` extends the training pipeline with warm-up, differential
learning rates and EMA.

### `src/analysis/`

Contains post-training analysis utilities, including metric computation,
geometric analysis, result storage, qualitative comparisons and
visualization.

### `src/visualization/`

Contains plotting and visualization utilities.

## Checkpoints

Large model checkpoints are not included in the repository.

The trained checkpoints are several hundred megabytes in size and should
not be committed to the normal Git history.

The recommended approach is to distribute large checkpoints separately and
keep the Git repository focused on source code, configuration and
documentation.

Once a checkpoint has been downloaded locally, it can be passed to the
analysis or visualization commands with:

```text
--checkpoint path/to/checkpoint.pt
```

Git LFS can also be used when direct distribution through GitHub is
required.

## References

- Tian, Z., Shen, C., Chen, H., He, T.
  *FCOS: Fully Convolutional One-Stage Object Detection.*
  ICCV, 2019.

- He, K., Zhang, X., Ren, S., Sun, J.
  *Deep Residual Learning for Image Recognition.*
  CVPR, 2016.

- Lin, T.-Y. et al.
  *Feature Pyramid Networks for Object Detection.*
  CVPR, 2017.

- Lin, T.-Y. et al.
  *Focal Loss for Dense Object Detection.*
  ICCV, 2017.

- Rezatofighi, H. et al.
  *Generalized Intersection over Union: A Metric and A Loss for Bounding Box
  Regression.*
  CVPR, 2019.

- Wu, L. et al.
  *Pneumonia detection based on RSNA dataset and anchor-free deep learning
  detector.*
  Scientific Reports, 2024.

## Dataset Sources

- RSNA Pneumonia Detection Challenge:
  https://www.rsna.org/challenge-datasets/2018

- Chest X-Ray Images (Pneumonia):
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
