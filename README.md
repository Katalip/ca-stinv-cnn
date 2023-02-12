# ca-stinv
Thank you for expressing expressing interest in our work.

Structure of the repository and the descriptions of each file can be found below
  
To create an environment:

conda create -n ftu_seg python=3.7
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install:
    pandas==1.3.5
    seaborn==0.11.2
    opencv-python==4.6.0.66
    albumentations==1.2.1
    wandb
    torchstain==1.2.0
    segmentation-models-pytorch==0.3.0
    kmeans1d==0.3.1



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

