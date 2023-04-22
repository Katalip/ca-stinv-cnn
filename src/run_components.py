import os
import wandb
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from modules.metrics import mean_Dice
from modules.utils import save_model, save_predictions_as_imgs
from modules.metrics import DicePrecisionRecall
from modules.dataset import hpa_hubmap_data_he, neptune_data, aidpath_data, hubmap21_kidney_data
from modules.utils import save_predictions_as_imgs

mse_loss = nn.MSELoss()
def rmse_loss(out, target):
    return torch.sqrt(mse_loss(out, target))

def train_model(model, train_loader, val_loader, optimizer, scaler, cfg):
 
    torch.cuda.empty_cache()
    gc.collect()
    
    iterations_per_epoch = len(train_loader)

    wandb.init(project=cfg['Logging'].get('wandb_project'), name=cfg['Train'].get('experiment_name'), resume=True)

    metric = mean_Dice()

    log = open(f"{cfg['Train'].get('save_path')}/train_logs.txt", 'a')
    log.write(cfg['Train'].get('experiment_name') + '\n')
    
    for epoch in range(cfg['Train'].get('start_epoch'), cfg['Train'].get('epochs')):

        start_time = time.time()
        
        #Train 
        model.train()
        model.output_type = ['loss', 'stain_info']
        train_loss, stain_loss, total_loss, score  = 0, 0, 0, 0
        iter = 0

        for data in train_loader:

            optimizer.zero_grad()
            img, mask, stain_matrices, img_tf = data
            img = img.to(cfg['Train'].get('device'))
            mask = mask.to(cfg['Train'].get('device'))
            stain_matrices = stain_matrices.to(cfg['Train'].get('device'))
            img_tf = img_tf.to(cfg['Train'].get('device'))

            # Domain adaptation parameter λ
            p = float(iter + epoch * iterations_per_epoch) / (cfg['Train'].get('epochs') * iterations_per_epoch)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
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
        
        if epoch % cfg['Train'].get('val_epoch') == 0 or (epoch % cfg['Train'].get('checkpoint_epoch') == 0 and epoch > 0):
            save_preds_flag = (epoch % cfg['Train'].get('checkpoint_epoch') == 0)
            val_loss, val_score = evaluate_model(model, val_loader, metric, cfg, save_preds = save_preds_flag) 
            
            temp = f"Val_loss: {val_loss:.3f} | Val Dice: {val_score:.3f}" 
            print(temp, end='')
            log.write(temp) 

            wandb.log({'train_loss':train_loss, 
                    'train_dice': score, 'val_loss': val_loss, 'total_loss':total_loss,
                    'stain_loss': stain_loss, 
                    'val_dice': val_score, 
                    'lr': optimizer.param_groups[0]['lr'],  
                    'wd': optimizer.param_groups[0]['weight_decay'],
                    'experiment_name': cfg['Train'].get('experiment_name')})
        
        print('')
        log.write('\n')

        if (epoch % cfg['Train'].get('checkpoint_epoch') == 0 or epoch == cfg['Train'].get('epochs') - 1 or epoch > cfg['Train'].get('epochs') - 20) and (epoch > 0): 
            save_model(model, f"{cfg['Train'].get('save_path')}/epoch_{epoch}.pth")

    wandb.finish()


def evaluate_model(model, val_loader, metric, cfg, save_preds=False):
    
    model.eval()
    model.output_type = ['loss']

    with torch.no_grad():
        valid_loss = 0
        val_score = 0
    
        for i, data in enumerate(val_loader):
            
            img, mask = data
            img = img.to(cfg['Train'].get('device'))
            mask = mask.to(cfg['Train'].get('device'))           
            outputs = model({'image':img, 'mask':mask})
            
            if save_preds:
                save_predictions_as_imgs(outputs['raw'], mask, folder=f"{cfg['Train'].get('save_path')}", idx=i)

            valid_loss += outputs['bce_loss'].mean() 
            val_score += metric(outputs['raw'], mask).item()

        valid_loss /= len(val_loader)
        val_score /= len(val_loader)

        return valid_loss, val_score


def test_external(model, val_loader, metric, device='GPU', save_path=None, save_preds=False):
    model.eval()
    model.output_type = ['loss']
    total_dice, total_precision, total_recall = 0, 0, 0
    
    with torch.no_grad():
        
        valid_loss = 0

        for i, data in enumerate(val_loader):
            
            img, mask = data
            img = img.to(device)
            mask = mask.to(device)
            
            outputs = model({'image':img, 'mask':mask})
            
            if save_preds:
                save_predictions_as_imgs(outputs['raw'], mask, folder=f"{save_path}", idx=i)
            
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



def eval_hpa_hubmap22(cfg, model, state_dict, log, device, save_path, batch_size=4):

    metric = DicePrecisionRecall()

    print(f'Dataset: HPA_HuBMAP_2022. Model: {state_dict}')
    
    log.write("Dataset: HPA_HuBMAP_2022" + '\n')
    log.write(f"Fold: {cfg['fold']}" + '\n')
    log.write(f'Model: {state_dict}' + '\n')

    val_data = hpa_hubmap_data_he(cfg=cfg, train=False)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = test_external(model, val_loader, metric, device, save_path)
    res = f'Organ: all | Loss: {loss.item():.3f} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    log.write(res + '\n')
    print(res)

    log.write('\n')  
    

def eval_neptune(cfg, model, state_dict, log, device, save_path, batch_size=4, img_size = 480):

    metric = DicePrecisionRecall()

    print(f'Dataset: NEPTUNE. Model: {state_dict}')
    
    log.write(f"Dataset: NEPTUNE. Model: {state_dict}" + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')
    
    for path in [cfg['he'], cfg['pas'], cfg['tri'], cfg['sil']]:
        neptune_path = os.path.join(cfg['root'], path)

        log.write(path + '\n')

        val_data = neptune_data(neptune_path, full_val=True, img_size=img_size)
        val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
        loss, dice, precision, recall = test_external(model, val_loader, metric, device, save_path)

        res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
        log.write(res + '\n')
        print(res)

    log.write('\n')    


def eval_aidpath(cfg, model, state_dict, log, device, save_path, img_size = 256, batch_size=4):

    metric = DicePrecisionRecall()

    print(f'Dataset: AIDPATH. Model: {state_dict}')
    log.write(f'Dataset: AIDPATH. Model: {state_dict}' + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')

    val_data = aidpath_data(cfg, full_val=True, img_size=img_size)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = test_external(model, val_loader, metric, device, save_path)
    res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
    log.write(res + '\n')
    print(res)

    log.write('\n')


def eval_hubmap_kidney(cfg, model, state_dict, log, device, save_path, img_size=224, batch_size=4):

    metric = DicePrecisionRecall()

    print(f'Dataset: HuBMAP21 Kidney. Model: {state_dict}')
    log.write(f'Dataset: HuBMAP21 Kidney. Model: {state_dict}' + '\n')

    notes = f'Img size: {img_size}'
    log.write(notes + '\n')

    val_data = hubmap21_kidney_data(cfg, full_val=True, img_size=img_size)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size)   
    loss, dice, precision, recall = test_external(model, val_loader, metric, device, save_path)
    res = f'Loss: {loss.item()} | Dice: {dice:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f})'
    log.write(res + '\n')
    print(res)

    log.write('\n')


