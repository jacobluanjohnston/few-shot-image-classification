# 3 minute presentation (no code)

## Group Members and Task Distribution
- Jacob — model selection, training pipeline, data loading, fine-tuning strategy

## The Network Backbone
- Initially, we thought small dataset needs smaller model. We thought a huge model (ViT, large ResNet) has too many parameters to fine-tune reliably on 1000 images. It might overfit fast. We picked a middle ground, EfficientNet-B3 at 12M params, as a trade off between powerful enough to learn and small enough not to overfit
  - EfficientNet-B3 pretrained on ImageNet (12.4M parameters) - https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html
  - Chosen for strong accuracy-to-size tradeoff vs larger models (ResNet50, ViT). Source: https://pytorch.org/vision/stable/models/efficientnet.html
  - Final classifier head replaced: Dropout(0.4) → Linear(1536, 100 classes)
    - Original last layer outputs 1000 numbers, one score per ImageNet class, we need 100 outputs for 100 classes
    - 0.4 is common default for fine-tuning. Source: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html and original Srivastava 2014 Dropout paper suggests 0.5
    - Try to fine tune between 0.3 - 0.5.
      - Too low (0.1) and almost all neurons active very pass, prone to memorization/overfitting.
      - Too high (0.7) and too much information is thrown away/underfitting
- Now we're trying OpenAI's CLIP (Contrastive Language-Image Pretraining)
  - EfficientNet only knows ImageNet's 1000 categories. Our dataset is mixed (food, cars, planes, etc.) sampled from multiple sources probably.
  - CLIP was trained on 400M diverse internet images so it has broader knowledge
  - Tradeoff is 151M params but CLIP's pretraining is so strong that possibly with even a few images it will generalize better than the previous model.
- We basically started conservative, see the ceiling, and upgraded. Then we will see if we should try a weaker model, stronger model, or begin to tweak the hyperparameters more.

## The Training Pipeline (e.g., from scratch/fine-tuning)
- We first tried EfficientNet
  - Transfer learning: ImageNet pretrained weights, fine-tuned on our dataset
  - ImageNet has 1.2M images across 1000 categories. EfficientNet-B3 spent weeks learning to recognize edges, textures, shapes, and objects from all of that.
  - Phase 1 (5 epochs): Backbone frozen, head-only training
    - Lets new classifier stabilize (become reasonable, no longer random garbage anymore) before touching pretrained weights, THEN backpropagate
  - Phase 2 (20 epochs): Full fine-tuning with differential learning rates - backbone at 1e-5, head at 1e-4
    - Backbone is tiny because we already have great ImageNet weights. Let's nudge it slightly toward our dataset, don't destroy past patterns/knowledge
    - Raise 10x on head, learn faster to catch up.
    - Teaching a chef/assistant metaphor
  - Best checkpoint saved based on validation accuracy
  - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- But EfficientNet plateaued at 30.56% so we switched to CLIP, OpenAI, for aforementioned resons
  - Same 2-phase fine-tuning strategy as EfficientNet
    - Phase 1: Freeze CLIP backbone, train new 100-class head only
    - Phase 2: Unfreeze all, backbone LR 1e-6 (even smaller than EfficientNet's 1e-5 - CLIP's weights are more valuable)
    - Head is smaller: Dropout(0.3) -> Linear(512, 100 classes)
      - 512 because ViT-B-32 outputs 512-dim feature vectors vs EfficientNet's 1536
    - Architecture: ViT-B-32 (Vision Transformer, Base size, 32x32 patches)
      - Image split into grid of 32x32 patches, each patch treated like word in a sentence
  - TA suggested only fine-tuning first 2 layers of head as conservative approach to prevent overfitting
  - Our experimental evidence disagreed — val accuracy improved with each stage of progressive unfreezing:
    - Head only: 57.41% val
    - After stage 1 (blocks 11,10,9): 62.04% val
    - After stage 2 (blocks 8,7,6): 71.30% val
    - After stage 3 (blocks 5,4,3): 73.61% val
    - After stage 4 (blocks 2,1,0): 79.63% val
    - After stage 5 (extended): 82.41% val
  - Val kept improving with each unfreeze — no sign of overfitting from additional blocks
  - Conclusion: progressive unfreezing worked better than conservative head-only approach for our mixed dataset

## Hyper-parameters
- CLIP-specific changes:
  - Image size: handled internally by CLIP's preprocess, no manual resize necessary
  - Phase 2 backbone LR: 1e-6 vs 1e-5 for EfficientNet - CLIP weights more valuable
  - Head LR: 1e-4 (same as EfficientNet's)
  - Dropout: 0.3 vs 0.4. CLIP features should generalize better out of the box.
  - No manual augmentation - CLIP's preprocess handles normalization, augmentation to be added in next experiment
  - Model: ViT-B-32, 151M parameters
  - Train/val split: same as EfficientNet — stratified 80/20 (863 train, 216 val)

- Image size: 224×224 (per assignment recommendation)
  - ImageNet suggested 300x300
- Batch size: 32 -> 64
- Optimizer: AdamW (weight decay 1e-4)
  - Commonly used optimizer in deep learning, standard regularization value.
  - Penalizes large weights to prevent overfitting
- Scheduler: CosineAnnealingLR per phase
  - Standard fine-tuning choice.
  - Smoothly decays learning rate from high to near zero following cosine curve instead of dropping it suddenly
  - Gives better final accuracy than fixed learning rate (LR). 
  - https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- Loss: CrossEntropyLoss with label smoothing 0.1
  - CrossEntropy is standard loss for classification.
  - 0.1 regularization to prevent overconfidence especially with small datasets.
  - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Phase 1 LR: 1e-3 | Phase 2 LR: backbone 1e-5, head 1e-4
  - 1e-3 is Adam's classic default lr.
  - Backbone gets 10x smaller than head
    - Howard & Ruder paper (ULMFiT) suggests each layer's LR 2.6x smaller than the layer above it. We simplified that to just two groups (backbone vs head) with a 10x difference.
- Train/val split: stratified 80/20 (863 train, 216 val) https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation(15)
  - With only 8 images per class, we use augmentation to artificially expand the dataset.
    - RandomResizedCrop - simulates different distances and positions
    - RandomHorizontalFlip - doubles effective dataset size for symmetric objects
    - ColorJitter - handles lighting variation
    - RandomRotation(15) - handles slight camera tilt
    - https://pytorch.org/vision/stable/transforms.html

## Current results, Training/testing curve
Over 95% goal.

**Experiment 1 — M1 Mac (baseline)**
- Phase 1 (5 epochs): val 7.87%
- Phase 2 (20 epochs): best val 26.85%
- Still climbing at epoch 20 → needed more epochs

**Experiment 2 — RTX 3060, 40 epochs**
- Phase 1 (8 epochs): val 7.87%
- Phase 2 (40 epochs): best val 30.56% (plateaued ~epoch 25)
- Train acc 45% vs val 30% → overfitting on small dataset
- Kaggle public score: 11.82% (public leaderboard = 11% of test data, very noisy)

**Experiment 3 — CLIP ViT-B-32 (3060, 40 epochs)**
- Phase 1 (8 epochs): val 57.41%
- Phase 2 (40 epochs): best val 78.70%
- Kaggle public score: 70.91%
- Train 99% vs val 78% → still some overfitting but massive improvement over EfficientNet

**Experiment 4 — CLIP ViT-B-32 Progressive Unfreezing (3060, 8+40 epochs)**
- Phase 1 (8 epochs): val 57.41%
- Phase 2 — 4 stages × 10 epochs, unfreezing 3 transformer blocks per stage (top → bottom)
  - Stage 1 (blocks 11,10,9): val 62.04%
  - Stage 2 (blocks 8,7,6): val 71.30%
  - Stage 3 (blocks 5,4,3): val 73.61%
  - Stage 4 (blocks 2,1,0): best val 79.63%
- Kaggle public score: 72.73%
- Slight improvement over full unfreeze (79.63% vs 78.70% val, 72.73% vs 70.91% Kaggle)
- Val still climbing at end of stage 4 — more epochs could help

**Experiment 5 — CLIP + Stage 5 extended (best submission)**
- Progressive unfreezing + 20 extra epochs (stage 5)
- Best val: 82.41% at stage 5 epoch 5
- Kaggle public score: 73.64% ← best submission so far
- Train ~99% vs val 82% → some overfitting but best generalization

**Experiment 6 — Training on all 1000 images (no val split)**
- Removed 80/20 split, trained on all 1079 images
- Train reached 99.07% (stage 4) → 100% (stage 5)
- No val accuracy to measure — flying blind
- Kaggle public score: 71.82% → worse than experiment 5
- Lesson: val set was doing useful work — tells us when to stop, acts as regularizer

**Experiment 7 — CLIP + Augmentation (RandomResizedCrop, HFlip, ColorJitter, Rotation)**
- Added augmentation before CLIP's preprocess
- Best val: 78.70%, train only reached 93.5% (healthier gap)
- Kaggle public score: 67.27% → worse
- Lesson: augmentation pipeline conflicted with CLIP's preprocess — double cropping distorted inputs
  - Train/val gap closed (93% vs 78% = 15% gap vs 99% vs 82% = 17% gap) showing less overfitting
  - But absolute val accuracy dropped — augmentation hurt more than it helped here

**Observations:**
- EfficientNet plateaued at 30% — not enough pretraining diversity for mixed dataset
- CLIP jumped to 78% val / 70% Kaggle — 400M diverse pretraining makes the difference
- Progressive unfreezing gave slight additional boost (70.91% → 72.73% Kaggle)
- Stage 5 extended training pushed further to 73.64% Kaggle — best result
- Training on all data and augmentation both hurt — val set is critical for knowing when to stop
- Training curves saved to training_curves.png
- Best Kaggle score: 73.64% = 83.64 Kaggle points (passing threshold is 60%)

## What's next?
- Resubmit experiment 5 checkpoint (73.64%) as baseline
- TTA (test-time augmentation) — run each test image multiple times, average predictions, no retraining needed
- CLIP ViT-L/14 — larger CLIP variant, 4x more parameters, likely what top scorers use
- Ensemble — combine ViT-B/32 + ViT-L/14 predictions
- Pseudo-labeling/self supervised learning?
- ~~MixUp / CutMix augmentation~~ conflicted with CLIP preprocess pipeline
  - Blends two training images together, interpolates their labels
  - Cut and paste, slightly more aggressive
  - Both regularization techniques designed for small datasets
  - MixUp — https://arxiv.org/abs/1710.09412
  - CutMix — https://arxiv.org/abs/1905.04899

## Sources
https://cs231n.github.io/transfer-learning/

https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html

https://pytorch.org/vision/stable/models/efficientnet.html

https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf - Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

https://pytorch.org/docs/stable/optim.html#per-parameter-options

https://arxiv.org/abs/1711.00489 - "Don't Decay the Learning Rate, Increase the Batch Size" (Smith et al. 2018)

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) - https://arxiv.org/abs/2103.00020

https://arxiv.org/abs/2010.11929

https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

MixUp — https://arxiv.org/abs/1710.09412

CutMix — https://arxiv.org/abs/1905.04899