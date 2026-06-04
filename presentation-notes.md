# 3 minute presentation (no code)

## Group Members and Task Distribution
- Jacob — model selection, training pipeline, data loading, fine-tuning strategy

## The Network Backbone
- EfficientNet-B3 pretrained on ImageNet (12.4M parameters) - https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html
- Chosen for strong accuracy-to-size tradeoff vs larger models (ResNet50, ViT). Source: https://pytorch.org/vision/stable/models/efficientnet.html
- Final classifier head replaced: Dropout(0.4) → Linear(1536, 100 classes)
  - Original last layer outputs 1000 numbers, one score per ImageNet class, we need 100 outputs for 100 classes
  - 0.4 is common default for fine-tuning. Source: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html and original Srivastava 2014 Dropout paper suggests 0.5
  - Try to fine tune between 0.3 - 0.5.
    - Too low (0.1) and almost all neurons active very pass, prone to memorization/overfitting.
    - Too high (0.7) and too much information is thrown away/underfitting

## The Training Pipeline (e.g., from scratch/fine-tuning)
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

## Hyper-parameters
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
- Results TBD — training in progress

## What's next?
- Pseudo-labeling/self supervised learning?
- Stronger backbone (EfficientNet-B4/B5 or ViT)
- MixUp / CutMix augmentation
- Test-time augmentation (TTA)
- Ensemble multiple seeds/architectures

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