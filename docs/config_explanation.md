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
