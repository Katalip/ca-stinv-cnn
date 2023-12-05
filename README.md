## 🔬 Improving Stain Invariance of CNNs

> Full Title: Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training <br>
Kudaibergen Abutalip, Numan Saeed, Mustaqeem Khan, Abdulmotaleb El Saddik <br>
Accepted at Medical Imaging with Deep Learning 2023, Nashville, USA <br>
> [Paper Link](https://openreview.net/pdf?id=uZ1SVZgEJ02) <br>

Thank you for expressing interest in our work

## 📖 Navigation
<details open>
  <summary>Environment Setup</summary>
  
  ## Setup
  1. Create an environment:
  ```
  conda create -n <env_name> python=3.9
  ```
  2. This version of pytorch requires separate installation (creating env from env.yml file or requirements.txt does not find it)
  ```
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```
  3. Install other necessary dependencies
  ```
  pip install -r requirements.txt
  ```

</details>
<details>
  <summary>Repository Structure</summary>

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
  
</details>
<details>
  <summary>Config File Explanation</summary>
</details>
<details>
  <summary>Datasets</summary>
</details>
<details>
  <summary>Train</summary>
</details>
<details>
  <summary>Evaluate</summary>
</details>

## Citation
```
@article{Abutalip2023ImprovingSI,
  title={Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training},
  author={Kudaibergen Abutalip and Numan Saeed and Mustaqeem Khan and Abdulmotaleb El Saddik},
  journal={ArXiv},
  year={2023},
  volume={abs/2304.11445},
  url={https://api.semanticscholar.org/CorpusID:258298481}
}```
