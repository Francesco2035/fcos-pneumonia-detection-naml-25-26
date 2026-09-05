# Pneumonia Detection Using Anchor-Free Object Detection

This repository contains my implementation of an anchor-free pneumonia detection system for chest X-ray images.

I developed the project for the Numerical Analysis for Machine Learning course at Politecnico di Milano. The main goal is to study pneumonia localization using an FCOS-style object detector with ResNet backbones.

## Project Overview

The project combines the RSNA Pneumonia Detection Challenge with an anchor-free object detection approach based on FCOS.

The application context is inspired by the work of Wu et al. on pneumonia detection and localization using the RSNA dataset. Instead of reproducing the reference implementation, I implemented an FCOS-style detector and used it to study pneumonia localization in chest X-ray images.

I compare two ResNet depths and two initialization strategies:

| Model | Backbone | Initialization |
|---|---|---|
| `R50-IN` | ResNet-50 | ImageNet |
| `R50-CX` | ResNet-50 | Chest-Xray |
| `R101-IN` | ResNet-101 | ImageNet |
| `R101-CX` | ResNet-101 | Chest-Xray |

All four detector configurations are trained and evaluated on the RSNA Pneumonia Detection Challenge dataset.

I use the Chest X-Ray Images (Pneumonia) dataset separately for domain-specific pretraining of the ResNet backbone.

## Main Features

### FCOS-Style Anchor-Free Detection

The detector does not use predefined anchor boxes.

Predictions are generated directly from spatial locations of the feature maps. For each location, the model predicts:

- a pneumonia classification score;
- four bounding-box distances `(l, t, r, b)`;
- a centerness score.

### Multi-Scale Feature Pyramid

The ResNet backbone provides feature representations at different spatial resolutions.

I use `C3`, `C4` and `C5` to build an FCOS-style feature pyramid with levels `P3` through `P7`.

### Target Assignment

Training targets follow an FCOS-style location-based formulation.

For each feature-map location, the target generation step determines whether the location is associated with a ground-truth bounding box and, when positive, computes:

- the positive-location mask;
- `(l, t, r, b)` regression targets;
- the centerness target.

Different regression ranges are assigned to the different FPN levels.

### Detection Loss

The training objective combines three components:

- Sigmoid Focal Loss for classification;
- Generalized IoU Loss for bounding-box regression;
- Sigmoid Focal Loss for centerness.

The total loss is:

```text
L = L_cls + L_reg + L_ctr
```

The three components are equally weighted.

Using Focal Loss for centerness is one of the project-specific modifications with respect to the original FCOS formulation.

### Transfer Learning

I support two backbone initialization strategies:

```text
ImageNet pretrained weights
Chest-Xray pretrained weights
```

The Chest-Xray initialization comes from a separate classification pretraining stage before detector training on the RSNA dataset.

### Training

The training pipeline supports:

- backbone freezing;
- differential learning rates;
- learning-rate warm-up;
- cosine annealing;
- gradient clipping;
- Exponential Moving Average (EMA);
- validation-based checkpoint selection.

Two training implementations are available:

- `standard`: the original training pipeline;
- `v2`: an extended version with additional training features such as warm-up, differential learning rates and EMA.

`TrainerV2` extends the standard trainer, but the additional operations can make training slower.

### Evaluation and Analysis

I use several evaluation metrics, including:

- Average Precision (AP);
- Average Recall at 10 detections (AR@10);
- image-level precision;
- image-level recall;
- specificity;
- F1-score;
- Youden's J;
- mean matched IoU;
- box-level precision, recall and F1.

For image-level evaluation, I calibrate the operating threshold on the validation set using Youden's J statistic.

### Visualization

The repository includes tools for:

- confusion matrices;
- prediction visualization;
- per-image detection results;
- feature-flow visualization;
- qualitative comparison of predictions;
- analysis of prediction disagreements.

Two visualization modes are available.

The standard mode uses the model-specific calibrated threshold.

The diagnostic mode uses a fixed threshold of `0.10` and disables NMS, redundancy suppression and the detection cap. I use it only for qualitative inspection, not for quantitative evaluation.

## Installation

I use [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.

### Install uv

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check the installation with:

```bash
uv --version
```

### Clone the Repository

```bash
git clone https://github.com/Francesco2035/fcos-pneumonia-detection-naml-25-26.git
cd fcos-pneumonia-detection-naml-25-26
```

### Install Dependencies

I use `pyproject.toml` to define the project and its dependencies, while `uv.lock` pins the exact versions used in the environment.

Install the dependencies with:

```bash
uv sync
```

Project commands can then be run through the managed environment:

```bash
uv run python main.py --help
```

## Dataset Setup

The datasets are not included in the repository.

I use two datasets for different purposes.

### RSNA Pneumonia Detection Challenge

The RSNA Pneumonia Detection Challenge is the main dataset I use for detector training, validation and evaluation.

Official source:

https://www.rsna.org/challenge-datasets/2018

The RSNA data should be stored locally under the project's `data/` directory.

The exact RSNA paths are defined in `src/config.py`.

### Chest X-Ray Images (Pneumonia)

I use the Chest X-Ray Images (Pneumonia) dataset for domain-specific backbone pretraining.

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The expected structure is:

```text
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

### Downloading the Chest X-Ray Dataset

The Kaggle CLI can be installed with:

```bash
uv tool install kaggle
```

Check the installation:

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

After extraction, make sure the dataset is available as:

```text
data/chest_xray/
```

## Running the Project

The main entry point is:

```text
main.py
```

All the main project operations are exposed through this file:

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

to see all available options.

## Pretraining

Chest-Xray pretraining can be run directly through `main.py`.

For example:

```bash
uv run python main.py \
    --mode pretrain \
    --pretrain-architecture 50 \
    --pretrain-data-dir data/chest_xray \
    --pretrain-output-dir checkpoints/pretrain
```

The pretraining interface supports configurable architecture, image size, epochs, batch size, learning rate, weight decay, workers, random seed and optional automatic mixed precision.

A standalone version of the same procedure is also available under `scripts/pretrain.py`. This is useful when I want to run pretraining separately from the main CLI.

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

### Training Checkpoints

There are two different ways to start from an existing checkpoint.

`--resume` continues the training of an existing experiment from its last checkpoint.

`--load-weights` starts a new run using the selected weights. I use this when I want to reuse weights in a new experiment, for example for fine-tuning.

A backbone-only version is also available through:

```text
--load-backbone-weights
```

## Analysis

After training, I can evaluate a detector checkpoint using:

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

The `no_th` mode uses a fixed threshold of `0.10` and disables NMS, redundancy suppression and the detection cap.

I use this mode only for qualitative inspection.

## Standalone Scripts

The `scripts/` directory contains a few standalone entry points for tasks that are separate from the main training pipeline.

### `scripts/pretrain.py`

This is the standalone version of the ResNet pretraining procedure on the Chest-Xray classification dataset.

It supports ResNet-50 and ResNet-101, optional loading of existing weights, configurable training parameters, backbone freezing and optional CUDA automatic mixed precision.

Example:

```bash
uv run python scripts/pretrain.py \
    --architecture 50 \
    --data-dir data/chest_xray \
    --output-dir checkpoints/pretrain
```

### `scripts/plot.py`

I use this script to create plots from TensorBoard-exported CSV files.

Several curves can be compared in the same figure by repeating the `--curve` argument.

Example:

```bash
uv run python scripts/plot.py \
    --metric "Validation AP" \
    --curve "ResNet-50 ImageNet:checkpoints/resnet50_imagenet/ap.csv" \
    --curve "ResNet-101 ImageNet:checkpoints/resnet101_imagenet/ap.csv"
```

The script exports both PNG and SVG versions of the plot.

The default output directory is:

```text
plots/
```

### `scripts/dataset_analysis.py`

I use this script for a basic analysis of the two datasets used in the project.

For the Chest-Xray dataset, it reports the number of images in the train, validation and test splits and their class distribution.

For the RSNA dataset, it analyzes image-level labels, annotations, DICOM metadata and bounding-box statistics.

Example:

```bash
uv run python scripts/dataset_analysis.py
```

The default dataset locations are:

```text
data/chest_xray/
data/rsna-pneumonia-detection-challenge/
```

The default output directory is:

```text
dataset_analysis/
```

### `scripts/compare_prediction_differences.py`

I use this script for a more targeted qualitative comparison of the four final detector models.

It first reads the existing per-image CSV results and looks for interesting disagreement cases, including differences in predicted categories, number of detections and localization quality.

It then runs inference only on the selected validation images and compares the actual predictions across models.

The script saves composite figures for verified category and localization differences.

The stored Youden thresholds are reused, so the script does not perform a new threshold calibration or recompute AP/AR.

Example:

```bash
uv run python scripts/compare_prediction_differences.py
```

The default output directory is:

```text
visualization/differences_csv_first/
```

## Repository Structure

```text
.
├── main.py
│
├── scripts/
│   ├── pretrain.py
│   ├── plot.py
│   ├── dataset_analysis.py
│   └── compare_prediction_differences.py
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

This is the main entry point of the project.

It handles:

- pretraining;
- detector training;
- analysis;
- visualization.

It also provides command-line control over backbone selection, checkpoint loading, training parameters, thresholding and visualization.

### `src/models/`

Contains the main detector components:

- `resnet.py`: ResNet implementation;
- `backbone.py`: backbone and feature extraction;
- `fpn.py`: Feature Pyramid Network;
- `detection_head.py`: FCOS-style detection heads;
- `target_generator.py`: target assignment;
- `detector.py`: complete detector framework.

### `src/datasets/`

Contains dataset loaders, dataset splitting and image transformations for the RSNA and Chest-Xray datasets.

### `src/train.py` and `src/train_v2.py`

Contain the two training implementations.

`TrainerV2` extends the standard trainer with additional training features such as warm-up, differential learning rates and EMA.

### `src/analysis/`

Contains the post-training analysis code, including metric computation, geometric analysis, result storage, qualitative comparisons and visualization.

### `src/visualization/`

Contains plotting and visualization utilities used by the main pipeline.

## Checkpoints

The four final detector checkpoints are available through the Hugging Face repository:

https://huggingface.co/Francesco2035/fcos-pneumonia-detection-model-checkpoints

The available models are:

- **R50-IN** — ResNet-50 with ImageNet initialization
- **R50-CX** — ResNet-50 with Chest-Xray initialization
- **R101-IN** — ResNet-101 with ImageNet initialization
- **R101-CX** — ResNet-101 with Chest-Xray initialization

Only the final detector checkpoints are provided. The intermediate Chest-Xray pretraining checkpoints are not included, since they are not required to use the final models.

The checkpoints are several hundred megabytes in size, so they are distributed separately from the Git repository.

Once a checkpoint has been downloaded, I can pass it to the analysis or visualization commands using:

```text
--checkpoint path/to/checkpoint.pt
```

For example:

```bash
uv run python main.py \
    --mode analyze \
    --checkpoint path/to/resnet101_imagenet_long_ft.pt \
    --backbone imagenet \
    --resnet-depth 101
```

The `--backbone` and `--resnet-depth` arguments should match the checkpoint being used.

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
  *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression.*  
  CVPR, 2019.

- Wu, L. et al.  
  *Pneumonia detection based on RSNA dataset and anchor-free deep learning detector.*  
  Scientific Reports, 2024.

## Dataset Sources

- RSNA Pneumonia Detection Challenge:  
  https://www.rsna.org/challenge-datasets/2018

- Chest X-Ray Images (Pneumonia):  
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## License

This project is released under the MIT License.

See the `LICENSE` file for the full license text.

## AI Assistance

Some utility scripts and supporting code were developed with the assistance of generative AI tools.

The detector and the main training pipeline were implemented from scratch, using papers, tutorials and other publicly available resources as references.
