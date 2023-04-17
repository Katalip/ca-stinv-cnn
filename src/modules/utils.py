import torch
from torchvision.utils import save_image
import pandas as pd
import numpy as np
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_stats(stats, save_root, fold):
    losses, val_losses, train_scores, val_scores = stats
    column_names = ['Train_loss','Val_loss','Train_Dice','Val_Dice']
    df = pd.DataFrame(np.stack([losses,val_losses,train_scores,val_scores],axis=1),columns=column_names)
    df.to_csv(f"{save_root}/logs_fold{fold}.csv")
    return df

def save_predictions_as_imgs(preds, y, folder=None, idx = 0):
    
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    try:
        os.mkdir(f'{folder}/viz_mask')
    except:
        pass

    save_image(preds, f"{folder}/viz_mask/pred_{idx}.png")
    save_image(y, f"{folder}/viz_mask/gt_{idx}.png")



