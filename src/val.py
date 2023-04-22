import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
from albumentations import *

from modules.models.resnet_smp_unet_he import ResUNet
from modules.models.convnext_smp_unet_he import ConvNextUNet
from run_components import eval_hpa_hubmap22, eval_aidpath, eval_hubmap_kidney, eval_neptune

import yaml
import os
from argparse import ArgumentParser


def parse_config():
    argparser = ArgumentParser()
    argparser.add_argument('config_name', help='Name of the yml config file')
    args = argparser.parse_args()

    # Config File
    dirname = os.path.dirname(__file__)
    cfg_name = args.config_name
    with open(os.path.join(dirname, f'cfg/{cfg_name}')) as f:
            cfg = yaml.safe_load(f)

    cfg['Train']['save_path'] = os.path.join(dirname, f"../experiments/{cfg['Train'].get('experiment_name')}")

    return cfg


def check_save_folders(cfg):

    try:
            os.makedirs(os.path.join(f'../experiments/'), exist_ok=True)
    except:
            raise Exception('Folder creation error')

    try:
            os.makedirs(cfg['Train']['save_path'], exist_ok=True)
    except:
            raise Exception('Folder creation error')


def main(cfg):
    if cfg['Architecture'].get('encoder') == 'resnet50':
        model = ResUNet(stinv_training=True, stinv_enc_dim=1, pool_out_size=6, filter_sensitive=True, \
                        n_domains=6, domain_pool='max').cuda()
        
    elif cfg['Architecture'].get('encoder') == 'convnext_tiny':
        model = ConvNextUNet(stinv_training=True, stinv_enc_dim=0, pool_out_size=6, filter_sensitive=True, \
                        n_domains=6, domain_pool='max', encoder_pretrain=None).cuda()
    
    state_dict = f"{cfg['Train']['save_path']}/epoch_{cfg['Eval'].get('checkpoint_epoch')}.pth"  
    model.load_state_dict(torch.load(state_dict), strict=False)

    log = open(f"{cfg['Train']['save_path']}/val_res.txt", 'a')

    eval_hpa_hubmap22(cfg['Data'], model, state_dict, log, \
                      device=cfg['Train'].get('device'), save_path=cfg['Train'].get('save_path'))
 
    eval_neptune(cfg['Eval']['neptune'], model, state_dict, log, \
                 device=cfg['Train'].get('device'), save_path=cfg['Train'].get('save_path'))
    
    eval_aidpath(cfg['Eval']['aidpath'], model, state_dict, log, \
                 device=cfg['Train'].get('device'), save_path=cfg['Train'].get('save_path'))
    
    eval_hubmap_kidney(cfg['Eval']['hubmap21_kidney'], model, state_dict, log, \
                 device=cfg['Train'].get('device'), save_path=cfg['Train'].get('save_path'))

    log.write('-'*20 + '\n')
    log.close()

    
if __name__ == '__main__':
    cfg = parse_config()
    check_save_folders(cfg)
    main(cfg)
    


    
    



    
    

        
    
