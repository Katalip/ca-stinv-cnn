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