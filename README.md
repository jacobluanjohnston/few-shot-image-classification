# CSE 144 Final Project — Transfer Learning Challenge
UC Santa Cruz, Spring 2026

## Overview
100-class image classification using transfer learning (EfficientNet-B3 pretrained on ImageNet), fine-tuned on a few-shot dataset of 1,000 training images across 100 classes.

## Kaggle Leaderboard
<!-- TODO -->
![Leaderboard](leaderboard.png)

## Model Weights
<!-- TODO -->
Trained model weights are available on Google Drive:
[Download weights](YOUR_GOOGLE_DRIVE_LINK_HERE)

## Setup
```bash
pip install torch torchvision matplotlib tqdm scikit-learn pillow pandas
```

## Data
Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project) and place it in a `data/` folder:
```
data/
  train/
    0/ ... 99/
  test/
    0.jpg ... 999.jpg
  sample_submission.csv
```

## Training
Open `final.ipynb` in Jupyter and run all cells in order.

- Best checkpoint is saved to `checkpoints/best_effb3.pt`

## Inference
Run the final cell in `final.ipynb` to generate `submission.csv`.

## Reproducibility
All runs use `set_seed(42)`. Results should be reproducible across machines with the same PyTorch version.
