## Improving Stain Invariance of CNNs (Implementation)

> Full title: Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training <br>
> Paper: To do (under review) <br> 
> Slide: To do <br>

> **Abstract**:
> *Variability in staining protocols, which can result in a diverse set of whole slide images (WSI) with different slide preparation techniques, chemicals, and scanner configurations, presents a significant challenge for adapting convolutional neural networks (CNNs) for computational pathology applications. This distribution shift can negatively impact the performance of deep learning models on unseen samples, especially in the task of semantic segmentation. In this study, we propose a method for improving the generalizability of CNNs to stain changes in a single-source setting. Our approach uses a channel attention mechanism that detects stain-specific features and a modified stain-invariant training scheme based on recent findings. We evaluate our method on multi-center, multi-stain datasets and demonstrate its effectiveness through interpretability analysis. Our approach achieves substantial improvements over baselines and competitive performance compared to other methods, as measured by various evaluation metrics. We also show that combining our method with stain augmentation leads to mutually beneficial results and outperforms other techniques. Overall, our study makes several important contributions to the field of computational pathology, including the proposal of a novel method for improving the generalizability of CNNs to stain changes in semantic segmentation and the modification of the previously proposed stain-invariant training scheme.* <br>

Thank you for expressing interest in our work.

## Setup
  
1. Create an environment:
```
conda create -n <env_name> python=3.7
```
2. This version of pytorch requires separate installation (creating env from env.yml file or requirements.txt does not find it)
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other necessary dependencies
```
pip install -r requirements.txt
```

## Structure of the repository
Descriptions of each file can be found below

```
|   .gitignore
|   environment.yml
|   README.md
|   requirements.txt
|   
\---src
    |   train.py -> Training script 
    |   val.py -> Validation and testing script
    |   
    +---cfg
    |       train_cfg.yml -> Main config file for training 
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
    |   |           
    |   +---stainspec -> This folder contains official implementation of one of the compared methods
```         

## Datasets
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

## Config file explanation
```
BS: Batch size
NFOLDS: K in StratifiedKfold split from sklearn library
FOLD: The fold used for train/val split
SEED: Random seed
TRAIN: Path to HPA train images folder 
MASKS: Path to HPA train masks folder
LABELS: Path to HPA csv file containing metadata
EXPERIMENT_NAME: Name for the experiment
NUM_WORKERS: Num workers
EPOCHS: N of epochs
DEVICE: To use GPU or not
PRETRAINED_WEIGHTS_CONVNEXT: Path to pretrained ConvNeXt weights 
VAL_EPOCH: How frequently to compute validation stats
CHECKPOINT_EPOCH: How frequently to save model state
START_EPOCH: Starting epoch in case previous training was interrupted 
```

#### To do:
- [ ] Add other configs
- [ ] Add other datasets in testing script
- [ ] Add training logs
- [x] Add dataset links
- [x] Add config explanation
- [ ] Check environment setup on other OS 
- [ ] Add instruction on how to adapt the method for your own model
- [ ] Check macenko_torch.py with newer pytorch versions
- [ ] Check whether wandb login info gets stored or not
- [ ] Automate model selection
- [ ] List potential small erros due to hyperparameter changes
- [ ] Add test time stain normalization in val.py

