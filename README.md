## Improving Stain Invariance of CNNs for Segmentation by Fusing Channel Attention and Domain-Adversarial Training

> Paper: To do <br> 
> Slide: To do <br>


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

