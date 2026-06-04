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
- Kaggle submission pending

**Observations:**
- More epochs helped (26% → 30%) but model plateaued
- Gap between train/val indicates overfitting — need stronger regularization
- Next: MixUp augmentation to combat overfitting

## What's next?
- Stronger backbone (EfficientNet-B4/B5 or ViT)
- Pseudo-labeling/self supervised learning?
- ~~MixUp / CutMix augmentation~~ Try stronger backbone first
  - Blends two training images together, interpolates their labels
  - Cut and paste, slightly more aggressive
  - Both regularization techniques designed for small datasets
  - MixUp — https://arxiv.org/abs/1710.09412
  - CutMix — https://arxiv.org/abs/1905.04899
- ~~Test-time augmentation (TTA)~~ Try stronger backbone first
  - Instead of an image at a time, run through multiple times with different augmentations (flipped, rotated, cropped) and average all predictions. Easy accuracy boost with no extra training
- ~~Ensemble multiple seeds/architectures~~ Try stronger backbone first
  - Train multiple models (different seeds and architectures), average their predictions
    - Kaggle method

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