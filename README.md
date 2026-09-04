
# Pneumonia Detection Using Anchor-Free Object Detection -  NAML_25-26

This repository contains the implementation of an anchor-free pneumonia
detection system for chest X-ray images, developed as part of the
Numerical Analysis for Machine Learning course at Politecnico di Milano.

The project investigates the use of an FCOS-style object detector with
ResNet backbones for pneumonia localization on the RSNA Pneumonia
Detection Challenge dataset.

## Project Overview

The core of the project combines ideas from two main references.

The application and problem setting are inspired by the work of Wu et al.,
which addresses pneumonia detection and localization on the RSNA dataset
using an anchor-free detection framework.

The detector implemented in this repository instead follows the
FCOS (Fully Convolutional One-Stage Object Detection) formulation more
directly. In particular, the model uses:

- ResNet-50 or ResNet-101 backbones
- a Feature Pyramid Network (FPN)
- FCOS-style anchor-free prediction
- classification, bounding-box regression and centerness branches
- dense predictions directly from spatial feature-map locations

The purpose of the project is not to reproduce the reference implementation
as a black box, but to study the FCOS formulation itself in the context of
pneumonia localization.

The experimental study focuses on two factors:

1. backbone depth: ResNet-50 vs. ResNet-101
2. backbone initialization: ImageNet vs. Chest-Xray pretraining

This results in four final configurations:

| Configuration | Backbone | Initialization |
|---|---|---|
| `R50-IN` | ResNet-50 | ImageNet |
| `R50-CX` | ResNet-50 | Chest-Xray |
| `R101-IN` | ResNet-101 | ImageNet |
| `R101-CX` | ResNet-101 | Chest-Xray |

All final detection experiments are performed on the RSNA dataset.
The Chest X-Ray Images (Pneumonia) dataset is used only to provide
domain-specific backbone pretraining before training on RSNA.

## Main Features

### Anchor-Free FCOS-Style Detection

The detector does not rely on predefined anchor boxes.

For each spatial location of the FPN feature maps, the model predicts:

- a pneumonia classification score
- four box distances `(l, t, r, b)`
- a centerness score

The final detector therefore performs dense prediction directly from
feature-map locations.

### Multi-Scale Feature Pyramid

The backbone provides hierarchical feature maps which are processed by
an FPN.

The implementation uses:

- `C3`, `C4`, `C5` from the ResNet backbone
- top-down feature fusion for `P3`, `P4`, `P5`
- additional `P6` and `P7` levels

The resulting pyramid provides predictions at multiple spatial scales.

### Target Assignment

Training targets are generated using an FCOS-style assignment strategy.

A feature-map location is mapped to image coordinates and can become a
positive sample when it lies inside a ground-truth bounding box and its
regression target is compatible with the corresponding FPN level.

The target generation produces:

- positive-location masks
- `(l, t, r, b)` regression targets
- centerness targets

### Detection Loss

The training objective combines three components:

- Sigmoid Focal Loss for classification
- GIoU Loss for bounding-box regression
- Sigmoid Focal Loss for centerness

The total objective is:

    L = L_cls + L_reg + L_ctr

with equal weights for the three components.

The centerness objective is a project-specific modification of the original
FCOS formulation, which uses binary cross-entropy for this branch.

### Transfer Learning and Pretraining

Two initialization strategies are supported:

- ImageNet pretrained ResNet weights
- Chest-Xray pretrained ResNet weights

Chest-Xray pretraining is performed as a separate classification stage and
the resulting backbone weights are subsequently used to initialize the
RSNA detection model.

### Training Stabilization

The training pipeline supports several stabilization mechanisms:

- backbone freezing during the initial epochs
- differential learning rates for backbone and detector components
- learning-rate warm-up
- cosine annealing
- gradient clipping
- Exponential Moving Average (EMA)
- validation-based checkpoint selection

The implementation allows these parameters to be adapted to the convergence
behaviour of each experiment.

### Evaluation and Analysis

The repository includes evaluation and analysis utilities for:

- Average Precision (AP)
- Average Recall at 10 detections (AR@10)
- image-level precision
- image-level recall
- image-level F1-score
- specificity
- Youden's J
- matched mean IoU
- box-level precision, recall and F1

For image-level evaluation, the operating threshold can be calibrated on
the validation set using Youden's J statistic.

### Visualization

Two visualization modes are available.

The standard mode uses the model-specific calibrated threshold and the
normal detection filtering pipeline.

A dedicated low-threshold visualization mode can instead be used to inspect
candidate detections with a fixed threshold of `0.10`, while disabling NMS,
redundancy suppression and the detection cap. This mode is intended for
qualitative inspection only and is not used for quantitative evaluation.

### Experiment Analysis

The analysis pipeline can generate:

- per-image detection results
- aggregate metrics
- confusion matrices
- prediction visualizations
- feature-flow examples

These outputs are stored in experiment-specific directories.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python environment
and dependency management.

### Install uv

If `uv` is not already installed, follow the official installation
instructions:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh