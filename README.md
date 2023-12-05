## 🔬 Improving Stain Invariance of CNNs

> Full Title: Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training <br>
Kudaibergen Abutalip, Numan Saeed, Mustaqeem Khan, Abdulmotaleb El Saddik <br>
Accepted at Medical Imaging with Deep Learning 2023, Nashville, USA <br>
> [Paper Link](https://openreview.net/pdf?id=uZ1SVZgEJ02) <br>

Thank you for expressing interest in our work

## 📖 Navigation
<details open>
  <summary>Environment Setup</summary>
  
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
<br>
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

<br>

```
Train:
  experiment_name: Name for the experiment (training run) 
  device: GPU or CPU, default: 'cuda'
  epochs: N of epochs for training
  val_epoch: Frequency of validation during training
  checkpoint_epoch: When to save model state
  start_epoch: Define if resuming from previous run
Data:
  train_imgs: Path to the training imgs of HUBMAP_HPA_22
  masks: Path to the training masks of HUBMAP_HPA_22
  labels: Path to the csv file with metadata of HUBMAP_HPA_22
  nfolds: Number of folds
  fold: Which fold to use
  seed: Random seed. Default: 309  
Loader:
  batch_size: Batch size for dataloaders
  num_workers: N of workers for dataloaders
Architecture:
  encoder: Backbone name. Either 'resnet50' or 'convnext_tiny'  
  weights: Initialized from 'segmentation models pytorch' pretrained weights for resnet, and loaded from .ckpt file for convnext
Logging:
  wandb_project: Wandb project name
Eval:
  checkpoint_epoch: Load model state from this epoch
  neptune:
    root: Directory that contains NEPTUNE img subfolders (each folder contains imgs prepared with different stain)
    he: Folder name for imgs stained with HE
    pas: Folder name for imgs stained with PAS
    sil: Folder name for imgs stained with SIL
    tri: Folder name for imgs stained with TRI
  aidpath:
    imgs: Path to the training imgs of AIDPATH
    masks: Path to the training masks of AIDPATH
  hubmap21_kidney:
    imgs: Path to the training imgs of HUBMAP 21 Kidney
    masks: Path to the training masks of HUBMAP 21 Kidney
```
  
</details>
<details>
  <summary>Datasets</summary>
  <br>
  We provide the links below and give a short description of their origin.

**HPA + HuBMAP 2022**. Human Protein Atlas (HPA) is a Swedish-based program (make a link), and The Human BioMolecular Atlas Program (HuBMAP) details its data contributors (US) [here.](https://hubmapconsortium.org/hubmap-data/#:~:text=HuBMAP%20data%20was%20generated%20using,assay%20types%20used%20in%20each) [Description of the dataset.](https://www.biorxiv.org/content/10.1101/2023.01.05.522764v1) [Download link](https://zenodo.org/record/7545745#.Y-M5SXZBwal) It is important to mention that the test set was not available during this study and this download page has been created recently <br>

**The Nephrotic Syndrome Study Network (NEPTUNE)** is a North American multi-center consortium. We use a subset of this dataset that contains only glomeruli with annotations of Bowman’s space to match the training data. Samples were collected across 29 enrollment centers (US and Canada).
[Description.](https://www.sciencedirect.com/science/article/pii/S0085253820309625)
The [download link](https://github.com/ccipd/DL-kidneyhistologicprimitives) is available at the bottom as online supplemental material (we use files named with 'glom_capsule').

**Academia and Industry Collaboration for Digital Pathology (AIDPATH)** is a Europen project. The data is collected in Spain and hosted by Mendeley. [Description.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7058889/#fn2) [Download](https://data.mendeley.com/datasets/k7nvtgn2x6/3)

WSIs in **HuBMAP21 Kidney** should come from the data contributors that can be viewed by the link provided above. [Description.](https://www.biorxiv.org/content/10.1101/2021.11.09.467810v1) [Download](https://github.com/cns-iu/ccf-research-kaggle-2021) (Data section)

We do not perform any specific preprocessing. Training images are resized to 768x768, while test samples are resized to sizes that match stats (pixel size, magnification) of the train data.
NEPTUNE images to 480x480
AIDPATH samples to 256x256
HuBMAP21 Kidney WSIs to 224x224
  
</details>
<details>
  <summary>Train</summary>
  <br>
  
  To train the model:
```
cd src
python train.py <encoder>.yml
```

E.g.
```
cd src
python train.py resnet.yml
```

Checkpoints and logs are stored in the Experiments folder in the parent directory and also logged with wandb
  
</details>
<details>
  <summary>Evaluate</summary>
  <br>
  
  To evaluate the model:
```
cd src
python val.py <encoder>.yml
```

E.g.
```
cd src
python val.py resnet.yml
```

Logs are stored in the Experiments folder in the parent directory
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
