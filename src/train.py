import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.dataset import *
from modules.metrics import *
from modules.utils import *
from modules.augs import *
from modules.models.resnet_smp_unet_he import *
from modules.models.convnext_smp_unet_he import *

import wandb
import time
import os
import gc
import yaml
from argparse import ArgumentParser

from albumentations import *

argparser = ArgumentParser()
argparser.add_argument('config_name', help='Name of the yml config file')
args = argparser.parse_args()

# Config File
dirname = os.path.dirname(__file__)
cfg_name = args.config_name
with open(os.path.join(dirname, f'cfg/{cfg_name}')) as f:
        cfg = yaml.safe_load(f)

EXPERIMENT_NAME = cfg['Global'].get('experiment_name') 
DEVICE = cfg['Global'].get('device')
EPOCHS = cfg['Global'].get('epochs')
VAL_EPOCH = cfg['Global'].get('val_epoch')
CHECKPOINT_EPOCH = cfg['Global'].get('checkpoint_epoch')
START_EPOCH = cfg['Global'].get('start_epoch')

BATCH_SIZE = cfg['Loader'].get('batch_size')
NUM_WORKERS = cfg['Loader'].get('num_workers')

ENCODER_NAME = cfg['Architecture'].get('encoder')
assert ENCODER_NAME in ('resnet50', 'convnext_tiny')

PRETRAINED_WEIGHTS = cfg['Architecture'].get('weights')
WANDB_PROJECT = cfg['Logging'].get('wandb_project') 


SAVE_PATH = os.path.join(dirname, f'../experiments/{EXPERIMENT_NAME}')

try:
        os.makedirs(os.path.join(dirname, f'../experiments/'), exist_ok=True)
except:
        raise Exception('Folder creation error')

try:
        os.makedirs(SAVE_PATH, exist_ok=True)
except:
        raise Exception('Folder creation error')


def train(model, train_loader, val_loader, optimizer, scaler):
    
    iterations_per_epoch = len(train_loader)

    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=WANDB_PROJECT, resume=True)

    metric = mean_Dice()

    log = open(f'{SAVE_PATH}/train_logs.txt', 'a')
    log.write(EXPERIMENT_NAME + '\n')
    
    global CHECKPOINT_EPOCH
    
    for epoch in range(START_EPOCH, EPOCHS):

        start_time = time.time()
        
        #Train 
        model.train()
        model.output_type = ['loss', 'stain_info']
        train_loss, stain_loss, total_loss, score  = 0, 0, 0, 0
        iter = 0

        for data in train_loader:

            optimizer.zero_grad()
            img, mask, stain_matrices, img_tf = data
            # img, mask, stain_matrices = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            stain_matrices = stain_matrices.to(DEVICE)
            img_tf = img_tf.to(DEVICE)

            # Domain adaptation parameter λ
            p = float(iter + epoch * iterations_per_epoch) / (EPOCHS * iterations_per_epoch)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            #iter += 1
            
            with torch.cuda.amp.autocast():
                outputs = model({'image':img, 'mask':mask, 'alpha':alpha, 'image_tf':img_tf})
                # BCE
                loss = outputs['bce_loss'].mean()

                # For forcing stain invariance / pred
                loss_stinv = rmse_loss(outputs['stain_info'], stain_matrices) 
                score += metric(outputs['raw'], mask).item()

            scaler.scale(loss + 0.5*loss_stinv).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            total_loss += (loss + 0.5*loss_stinv).item() 
            train_loss += loss.item() 
            stain_loss += loss_stinv.item()

        end_time = time.time()
        elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)) 

        total_loss /= len(train_loader)
        train_loss /= len(train_loader)
        stain_loss /= len(train_loader)

        score /= len(train_loader)
        temp = f"Epoch: {epoch + 1} | Total_loss: {total_loss:.3f} | Train_loss: {train_loss:.3f} | Stain_loss: {stain_loss:.3f} | Train Dice: {score:.3f} | Epoch Time: {elapsed} | " 
        print(temp, end='')
        log.write(temp)
        
        if epoch % VAL_EPOCH == 0 or (epoch % CHECKPOINT_EPOCH == 0 and epoch > 0):
            save_preds_flag = (epoch % CHECKPOINT_EPOCH == 0)
            val_loss, val_score = evaluate_model(model, val_loader, metric, save_preds = save_preds_flag) 
            
            temp = f"Val_loss: {val_loss:.3f} | Val Dice: {val_score:.3f}" 
            print(temp, end='')
            log.write(temp) 

            wandb.log({'train_loss':train_loss, 
                    'train_dice': score, 'val_loss': val_loss, 'total_loss':total_loss,
                    'stain_loss': stain_loss, 
                    # 'isw_loss': stain_loss,
                    'val_dice': val_score, 
                    'lr': optimizer.param_groups[0]['lr'],  
                    'wd': optimizer.param_groups[0]['weight_decay'],
                    'experiment_name':EXPERIMENT_NAME})
        
        print('')
        log.write('\n')

        if (epoch % CHECKPOINT_EPOCH == 0 or epoch == EPOCHS - 1 or epoch > EPOCHS - 20) and (epoch > 0): 
            save_model(model, f"{SAVE_PATH}/epoch_{epoch}.pth")
        
        if CHECKPOINT_EPOCH == 50 and epoch > EPOCHS - CHECKPOINT_EPOCH:
            CHECKPOINT_EPOCH = 20

    wandb.finish()


def evaluate_model(model, val_loader, metric, save_preds=False):
    
    model.eval()
    model.output_type = ['loss']

    with torch.no_grad():
        valid_loss = 0
        val_score = 0
    
        for i, data in enumerate(val_loader):
            
            img, mask = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)           
            outputs = model({'image':img, 'mask':mask})
            
            if save_preds:
                save_predictions_as_imgs(outputs['raw'], mask, folder=f"{SAVE_PATH}", idx=i)

            valid_loss += outputs['bce_loss'].mean() 
            val_score += metric(outputs['raw'], mask).item()

        valid_loss /= len(val_loader)
        val_score /= len(val_loader)

        return valid_loss, val_score


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    gc.collect()

    # Data
    train_data = hpa_hubmap_data_he(cfg=cfg['Data'], train=True, tfms=get_aug_a1(), selection_tfms=get_aug_selection())
    val_data = hpa_hubmap_data_he(cfg=cfg['Data'], train=False)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Loss
    best_loss, best_score = float('inf'), 0
    bce_loss = nn.BCEWithLogitsLoss() 
    mse_loss = nn.MSELoss()
    def rmse_loss(out, target):
        return torch.sqrt(mse_loss(out, target))

    # Model
    if ENCODER_NAME == 'resnet50':
        model = ResUNet(stinv_training=True, stinv_enc_dim=1, pool_out_size=6, filter_sensitive=True, n_domains=6, domain_pool='max').cuda()
    elif ENCODER_NAME == 'convnext_tiny':
        model = ConvNextUNet(stinv_training=True, stinv_enc_dim=1, pool_out_size=6, filter_sensitive=True, n_domains=6, domain_pool='max', encoder_pretrain=PRETRAINED_WEIGHTS).cuda()
        model.load_pretrain()

    if START_EPOCH > 0:
        model.load_state_dict(torch.load(f"{SAVE_PATH}/epoch_{START_EPOCH}.pth"))
        print(f'Starting from epoch: {START_EPOCH}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    train(model, train_loader, val_loader, optimizer, scaler)