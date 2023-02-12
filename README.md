## Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training

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
    |   train.py
    |   val.py
    |   
    +---cfg
    |       train_cfg.yml
    |       
    +---modules
    |   |   augs.py
    |   |   dataset.py
    |   |   macenko_torch.py
    |   |   metrics.py
    |   |   utils.py
    |   |   
    |   +---models
    |   |   |   convnext.py
    |   |   |   convnext_smp_unet_he.py
    |   |   |   resnet_smp_unet_he.py
    |   |           
    |   +---stainspec
```         

