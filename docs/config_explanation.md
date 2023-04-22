## Config file explanation
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
