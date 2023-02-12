import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.dataset import *
from modules.metrics import *
from modules.utils import *

from albumentations import *

import yaml
import os

dirname = os.path.dirname(__file__)

cfg_name = 'train_cfg.yml'
with open(os.path.join(dirname, f'cfg/{cfg_name}')) as f:
        # use safe_load instead load
        cfg = yaml.safe_load(f)


bs = cfg['BS']
nfolds = cfg['NFOLDS']
fold = cfg['FOLD']
SEED = cfg['SEED']
TRAIN = os.path.join(dirname, cfg['TRAIN'])
MASKS = os.path.join(dirname, cfg['MASKS'])
LABELS = os.path.join(dirname, cfg['LABELS'])
EXPERIMENT_NAME = cfg['EXPERIMENT_NAME']
NUM_WORKERS = cfg['NUM_WORKERS']
EPOCHS = cfg['EPOCHS']
DEVICE = cfg['DEVICE']
save_root = os.path.join(dirname, f'../experiments/{EXPERIMENT_NAME}')

datasets_for_val = ['hpa_hubmap_2022']

def evaluate_model(model, val_loader, metric, save_preds=False):
    model.eval()
    model.output_type = ['loss']
    total_dice, total_precision, total_recall = 0, 0, 0
    
    with torch.no_grad():
        
        valid_loss = 0

        for i, data in enumerate(val_loader):
            
            img, mask = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            
            outputs = model({'image':img, 'mask':mask})
            
            if save_preds:
                save_predictions_as_imgs(outputs['raw'], mask, folder=f"{save_root}", idx=i)
            
            val_dice, val_precision, val_recall = metric(outputs['raw'], mask)
            valid_loss += outputs['bce_loss'].mean()

            total_dice += val_dice
            total_precision += val_precision
            total_recall += val_recall

            
        valid_loss /= len(val_loader)
        mean_dice = total_dice / len(val_loader)
        precision = total_precision / len(val_loader)
        recall = total_recall / len(val_loader)

        return valid_loss, mean_dice, precision, recall


from modules.models.resnet_smp_unet_he import *
from modules.models.convnext_smp_unet_he import *

if __name__ == '__main__':

    # Loss, Metric
    loss_func = nn.BCEWithLogitsLoss()
    metric = main_metrics()

    # Model
    model = ResUNet(stinv_training=True, stinv_enc_dim=1, pool_out_size=6, filter_sensitive=True, n_domains=6, domain_pool='max').cuda()

    checkpoint = 1 # 0 
    state_dict = f"{save_root}/epoch_{checkpoint}.pth"  
    model.load_state_dict(torch.load(state_dict))


    if 'hpa_hubmap_2022' in datasets_for_val:

        df = pd.read_csv(LABELS)
        kf = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True) 
        train, val = list(kf.split(df, df['organ']))[fold]

        sub_df = df.iloc[val]

        organs = ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']
        # organs = ['lung']
        print(f'Model: {state_dict}')
        
        log = open(f'{save_root}/val_res.txt', 'a')
        log.write('Dataset: hpa_hubmap_2022' + '\n')
        log.write(f'Fold: {fold}' + '\n')
        log.write(f'Model: {state_dict}' + '\n')

        notes = ''
        log.write(notes + '\n')

        val_data = hpa_hubmap_data(fold=fold, train=False)
        val_loader = DataLoader(val_data, shuffle = True, batch_size = bs)   
        loss, dice, precision, recall = evaluate_model(model, val_loader, metric)
        res = f'Organ: all | Loss: {loss.item():.3f} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
        log.write(res + '\n')
        print(notes)
        print(res)

        for organ in organs: 
            mask = (sub_df['organ'] == organ)
            val_ids = sub_df[mask].id.astype(str).values
            val_data = hpa_hubmap_data_val(fold=fold, ids=val_ids, train=False)
            val_loader = DataLoader(val_data, shuffle = True, batch_size = bs)   
            loss, dice, precision, recall = evaluate_model(model, val_loader, metric)
            res = f'Organ: {organ} | Loss: {loss.item():.3f} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
            log.write(res + '\n')
            print(res)

        log.write('\n')
        log.close()
    
