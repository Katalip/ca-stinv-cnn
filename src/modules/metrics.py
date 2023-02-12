import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.):
        
        # If you use this for a batch, you're not computing dice for each image separately
        # Gives higher dice because impact of less accurately segmented parts is dumpened

        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        intersection = (y_true * y_pred).sum()
        dice = (2.0*intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
        
        return dice

class mean_Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, preds, mask):
        
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        N = len(preds)
        p = preds.reshape(N,-1)
        t = mask.reshape(N,-1)
        t = t > 0.5
        
        uion = p.sum(-1) + t.sum(-1)
        overlap = (p*t).sum(-1)
        dice = 2*overlap/(uion+0.0001)
        
        return dice.mean()        


class main_metrics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, mask):
        
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        N = len(preds)
        p = preds.reshape(N,-1)
        t = mask.reshape(N,-1)
        t = t>0.5
        
        uion = p.sum(-1) + t.sum(-1)
        overlap = (p*t).sum(-1)
        dice = 2*overlap/(uion+0.0001)

        tp = (p*t).sum(-1)
        tp_sum_fn = t.sum(-1)
        recall = tp/(tp_sum_fn + 0.0001)

        tp_sum_fp = p.sum(-1)
        precision = tp/(tp_sum_fp + 0.0001)
    
        return dice.mean().item(), precision.mean().item(), recall.mean().item()
