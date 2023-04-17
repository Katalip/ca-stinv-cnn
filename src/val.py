import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from albumentations import *

from modules.dataset import hpa_hubmap_data_he, neptune_data, aidpath_data, hubmap21_kidney_data
from modules.metrics import DicePrecisionRecall
from modules.utils import save_predictions_as_imgs
from modules.models.resnet_smp_unet_he import ResUNet
from modules.models.convnext_smp_unet_he import ConvNextUNet

import yaml
import os
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('config_name', help='Name of the yml config file')
args = argparser.parse_args()

# Config File
dirname = os.path.dirname(__file__)
cfg_name = args.config_name
with open(os.path.join(dirname, f'cfg/{cfg_name}')) as f:
        cfg = yaml.safe_load(f)

EXPERIMENT_NAME = cfg['Train'].get('experiment_name') 
DEVICE = cfg['Train'].get('device')

BATCH_SIZE = cfg['Loader'].get('batch_size')
NUM_WORKERS = cfg['Loader'].get('num_workers')

ENCODER_NAME = cfg['Architecture'].get('encoder')
assert ENCODER_NAME in ('resnet50', 'convnext_tiny')
CHECKPOINT_EPOCH = cfg['Eval'].get('checkpoint_epoch')
 
SAVE_PATH = os.path.join(dirname, f'../experiments/{EXPERIMENT_NAME}')

try:
        os.makedirs(os.path.join(dirname, f'../experiments/'), exist_ok=True)
except:
        raise Exception('Folder creation error')

try:
        os.makedirs(SAVE_PATH, exist_ok=True)
except:
        raise Exception('Folder creation error')

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
                save_predictions_as_imgs(outputs['raw'], mask, folder=f"{SAVE_PATH}", idx=i)
            
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


def eval_hpa_hubmap22(cfg, model, metric, log, batch_size=4):

    print(f'Dataset: HPA_HuBMAP_2022. Model: {state_dict}')
    
    log.write("Dataset: HPA_HuBMAP_2022" + '\n')
    log.write(f"Fold: {cfg['fold']}" + '\n')
    log.write(f'Model: {state_dict}' + '\n')

    val_data = hpa_hubmap_data_he(cfg=cfg, train=False)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = evaluate_model(model, val_loader, metric)
    res = f'Organ: all | Loss: {loss.item():.3f} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    log.write(res + '\n')
    print(res)

    log.write('\n')  
    

def eval_neptune(cfg, model, metric, log, batch_size=4, img_size = 480):

    print(f'Dataset: NEPTUNE. Model: {state_dict}')
    
    log.write(f"Dataset: NEPTUNE. Model: {state_dict}" + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')
    
    for path in [cfg['he'], cfg['pas'], cfg['tri'], cfg['sil']]:
        neptune_path = os.path.join(cfg['root'], path)

        log.write(path + '\n')

        val_data = neptune_data(neptune_path, full_val=True, img_size=img_size)
        val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
        loss, dice, precision, recall = evaluate_model(model, val_loader, metric)

        res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
        log.write(res + '\n')
        print(res)

    log.write('\n')    

def eval_aidpath(cfg, model, metric, log, img_size = 256, batch_size=4):

    print(f'Dataset: AIDPATH. Model: {state_dict}')
    log.write(f'Dataset: AIDPATH. Model: {state_dict}' + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')

    val_data = aidpath_data(cfg, full_val=True, img_size=img_size)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = evaluate_model(model, val_loader, metric)
    res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
    log.write(res + '\n')
    print(res)

    log.write('\n')


def eval_hubmap_kidney(cfg, model, metric, log, img_size=224, batch_size=4):

    print(f'Dataset: HuBMAP21 Kidney. Model: {state_dict}')
    log.write(f'Dataset: HuBMAP21 Kidney. Model: {state_dict}' + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')

    val_data = hubmap21_kidney_data(cfg, full_val=True, img_size=img_size)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = evaluate_model(model, val_loader, metric)
    res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
    log.write(res + '\n')
    print(res)

    log.write('\n')



    
if __name__ == '__main__':

    if ENCODER_NAME == 'resnet50':
        model = ResUNet(stinv_training=True, stinv_enc_dim=1, pool_out_size=6, filter_sensitive=True, n_domains=6, domain_pool='max').cuda()
    elif ENCODER_NAME == 'convnext_tiny':
        model = ConvNextUNet(stinv_training=True, stinv_enc_dim=0, pool_out_size=6, filter_sensitive=True, n_domains=6, domain_pool='max', encoder_pretrain=None).cuda()
    
    state_dict = f"{SAVE_PATH}/epoch_{CHECKPOINT_EPOCH}.pth"  
    model.load_state_dict(torch.load(state_dict), strict=False)

    metric = DicePrecisionRecall()

    log = open(f"{SAVE_PATH}/val_res.txt", 'a')

    # eval_hpa_hubmap22(cfg['Data'], model, metric, log)
    # eval_neptune(cfg['Eval']['neptune'], model, metric, log)
    eval_aidpath(cfg['Eval']['aidpath'], model, metric, log)
    # eval_hubmap_kidney(cfg['Eval']['hubmap21_kidney'], model, metric, log)

    log.write('-'*20 + '\n')
    log.close()


    
    



    
    

        
    
