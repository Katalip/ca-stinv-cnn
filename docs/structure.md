
## Structure of the repository
Descriptions of each file can be found below

```
|   .gitignore
|   environment.yml
|   README.md
|   requirements.txt
|   LICENSE
|   pyproject.toml
|   .flake8
|   .pre-commit-config.yaml
|
\---src
    |   train.py -> Training script
    |   val.py -> Validation and testing script
    |   run_components.py -> Functions for training/evaluating/testing
    |
    +---cfg
    |       resnet.yml -> Config file for ResNet
    |       convnext.yml -> Config file for ConvNeXt
    |
    +---modules
    |   |   augs.py -> Augmentations used in the study
    |   |   dataset.py -> Data classess. Standard dataloaders are used
    |   |   macenko_torch.py -> Edited version of Macenko normalization in pytorch.
    |   |                       We added a small function for getting optimal stain vectors
    |   |   metrics.py -> Main metrics: dice, precision, recall
    |   |   utils.py -> Some helper functions
    |   |
    |   +---models
    |   |   |   convnext.py -> Implementation of ConvNeXt from their official repository
    |   |   |   convnext_smp_unet_he.py -> Unet with ConvNeXt as backbone, stain-invariant training branch, and channel attention
    |   |   |   resnet_smp_unet_he.py -> Unet with ResNet as backbone, stain-invariant training branch, and channel attention
    |   |   |   stinv_training.py -> Domain-predictor and gradient reversal
    |   |   |   isw.py -> Reimplementation of instance-selective whitening
    |   |   |   cov_attention.py -> Proposed channel attention mechanism
    |   |
    |   +---stainspec -> This folder contains official implementation of one of the compared methods
```
