import wandb
import time
import gc

import torch
import torch.nn as nn
import numpy as np

from modules.metrics import mean_Dice
from modules.utils import save_model, save_predictions_as_imgs

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