<div align="center">

![alzheimersCNN](./public/logo.svg)

_Predicting levels of Alzheimers using a CNN with MRI images_

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?style=for-the-badge&logo=matplotlib&logoColor=fff)](#)
</div>

A convolutional neural network (CNN) built in PyTorch to classify the severity of Alzheimer's disease from MRI brain scans. Trained on the [MRI Scans Alzheimer Detection Dataset](https://huggingface.co/datasets/yogitamakkar178/mri_scans_alzeimer_detection) via Hugging Face, the model achieves a best test accuracy of **~99%** across four severity classes.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Reflection](#reflection)

## Overview

This project trains a custom CNN to classify MRI brain scans into four Alzheimer's severity categories. Key design choices include:

- Custom stratified train/test splitting (80/20) to address class imbalance in the original dataset splits
- Pixel value normalization applied to both train and test sets
- Dropout regularization to reduce overfitting
- Training and test accuracy tracked across all epochs and visualized with a line plot

## Dataset

**[MRI Scans Alzheimer Detection](https://huggingface.co/datasets/yogitamakkar178/mri_scans_alzeimer_detection)** — ~10,800 grayscale MRI brain scan images, each labeled with one of four Alzheimer's severity levels.

The original train/test splits from Hugging Face were found to be class-imbalanced, leading to overfitting. The splits are discarded and regenerated with an 80/20 stratified split to ensure balanced class representation across both sets.

| Split | Size |
|---|---|
| Train | 80% |
| Test | 20% |

**Preprocessing:**

Both splits are normalized with `mean=0.5, std=0.5`. No data augmentation is applied to the final model — it was tested but produced negligible or slightly negative results at this dataset scale.

## Architecture

The model is a four-block CNN followed by a two-layer fully connected classifier.

| Layer | Operation | Output Shape |
|---|---|---|
| Input | Grayscale MRI scan | `1 × 128 × 128` |
| Block 1 | Conv2d(1→32, 3×3) + LeakyReLU + MaxPool(2×2) | `32 × 64 × 64` |
| Block 2 | Conv2d(32→64, 3×3) + LeakyReLU + MaxPool(2×2) | `64 × 32 × 32` |
| Block 3 | Conv2d(64→128, 3×3) + LeakyReLU + MaxPool(2×2) | `128 × 16 × 16` |
| Block 4 | Conv2d(128→256, 3×3) + LeakyReLU + MaxPool(2×2) | `256 × 8 × 8` |
| FC1 | Linear(16384 → 2048) + Dropout(0.5) | `2048` |
| FC2 | Linear(2048 → 4) | `4` |

Each convolutional block uses same-padding (`padding=1`) to preserve spatial dimensions before pooling.

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.0001` |
| Loss function | Cross-entropy |
| Batch size | `32` |
| Epochs | `30` |

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended; CPU fallback available)

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
datasets
transformers
```

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/alzheimers-cnn.git
cd alzheimers-cnn
```

2. **Create and activate a virtual environment (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install torch torchvision datasets transformers numpy pandas matplotlib seaborn
```

4. **Authenticate with Hugging Face** (required to download the dataset):

```bash
huggingface-cli login
```

## Usage

Open and run `AlzheimersCNN.ipynb` in Jupyter:

```bash
jupyter notebook AlzheimersCNN.ipynb
```

Run all cells in order to:
1. Download and preprocess the dataset
2. Define and initialize the model
3. Train for 30 epochs, printing loss and accuracy per epoch
4. Evaluate on the test set after each epoch
5. Plot train vs. test accuracy across all epochs

## Results

The model achieves a best test accuracy of approximately **99%**, with train and test accuracies remaining closely aligned throughout training — test accuracy trailing train accuracy by roughly 2%.

Training and test accuracy are plotted across all 30 epochs using `seaborn`, giving a clear view of convergence and any signs of overfitting.

## Reflection

### What worked

Regenerating the train/test splits was the single most impactful change in this project. The original Hugging Face splits were class-imbalanced, which caused the model to overfit and generalize poorly. Switching to a custom stratified 80/20 split improved test accuracy by approximately **10%** and brought train and test accuracy much closer together across epochs.

### What didn't

Data augmentation (color jitter) was tested but produced no meaningful improvement — and was slightly detrimental. This is likely due to the relatively small dataset size (~10.8k images); augmentation tends to be more impactful at larger scales.

### Potential improvements

- **Normalization:** The pipeline uses generic `mean=0.5, std=0.5` values. Computing the true per-channel mean and standard deviation of the dataset could improve optimizer performance.
- **Larger dataset:** With only ~10.8k images, the model may not generalize as well to real-world MRI scans. A larger, more diverse dataset would likely yield more robust results.
- **Hyperparameter tuning:** Further experimentation is warranted with batch size, learning rate, number of layers, filter counts, and activation functions to find optimal values.