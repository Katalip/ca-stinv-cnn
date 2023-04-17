
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedKFold

import yaml
import os

import torch
import torchvision.transforms as torchtf
from torch.utils.data import Dataset

from albumentations import *
from modules.macenko_torch import TorchMacenkoNormalizer


dirname = os.path.dirname(__file__)

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

# HPA_Hubmap_22
organ_meta = dict(
    kidney = dict(
        label = 1,
        um    = 0.5000,
        ftu   ='glomeruli',
    ),
    prostate = dict(
        label = 2,
        um    = 6.2630,
        ftu   ='glandular acinus',
    ),
    largeintestine = dict(
        label = 3,
        um    = 0.2290,
        ftu   ='crypt',
    ),
    spleen = dict(
        label = 4,
        um    = 0.4945,
        ftu   ='white pulp',
    ),
    lung = dict(
        label = 5,
        um    = 0.7562,
        ftu   ='alveolus',
    ),
)

organ_to_label = {k: organ_meta[k]['label'] for k in organ_meta.keys()}
label_to_organ = {v:k for k,v in organ_to_label.items()}

class hpa_hubmap_data_he(Dataset):
    def __init__(self, cfg, train=True, tfms=None, selection_tfms=None):

        self.TRAIN_IMGS = os.path.join(dirname, '../' + cfg['train_imgs'])
        self.MASKS = os.path.join(dirname, '../' + cfg['masks'])
        LABELS = os.path.join(dirname, '../' + cfg['labels'])

        df = pd.read_csv(LABELS)
        ids = df.id.astype(str).values
        organs = df.organ.astype(str).values

        kf = StratifiedKFold(n_splits=cfg['nfolds'], random_state=cfg['seed'], shuffle=True)
        ids = set(ids[list(kf.split(ids, organs))[cfg['fold']][0 if train else 1]])
            
        self.fnames = [fname for fname in os.listdir(self.TRAIN_IMGS) if fname.split('.')[0] in ids]
        self.train = train
        self.tfms = tfms
        self.selection_tfms = selection_tfms
        self.df = df
    
        self.normalizer = TorchMacenkoNormalizer()
        self.T = torchtf.Compose([
                 torchtf.ToTensor(),
                 torchtf.Lambda(lambda x: x*255)
                ])

    def __len__(self):
        return len(self.fnames)
    
    def get_stain_info(self, img):

        try:
            stain_matrix, _, _ = self.normalizer.get_stain_matrix(self.T(img))
        except:
            print("Lapack error -> returning default stain matrix")
            stain_matrix = torch.tensor([0.58096592, 0.27774671, 0.66405821, 0.53398947, 0.47064348,
       0.79856872]).float()

        stain_matrix = stain_matrix.view(-1)

        return stain_matrix

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.TRAIN_IMGS,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.MASKS, fname), cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img, mask = augmented['image'], augmented['mask']

        if self.selection_tfms is not None:
            augmented = self.selection_tfms(image=img,mask=mask)
            img_tf = augmented['image']
        
        mask = img2tensor(mask)
        res = [img2tensor(img/255.0), mask]

        if self.train:
            stain_matrix = self.get_stain_info(img)                
            res.append(stain_matrix)

        if self.selection_tfms:
            res.append(img2tensor(img_tf/255.0))

        return res


class hpa_hubmap_data_val(Dataset):
    def __init__(self, ids=None, fold=None, train=True, tfms=None):
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('.')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS, fname),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return img2tensor(img/255.0), img2tensor(mask) 
    

# NEPTUNE
class neptune_data(Dataset):
    def __init__(self, root, train=True, full_val=False, tfms=None, split_size=0.8, img_size=None):
        
        root = os.path.join(dirname, '../', root)
        fnames = sorted(os.listdir(root))
        fnames = [i for i in fnames if 'mask' not in i]
        
        if not full_val:
            if train:
                fnames = fnames[:int(len(fnames)*split_size)]
            else:
                fnames = fnames[int(len(fnames)*split_size):]

        self.fnames = fnames

        self.root = root
        self.tfms = tfms
        self.img_size = img_size
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, fname)), cv2.COLOR_BGR2RGB)
        
        mask_name = fname.split('.')[0]
        mask_name += '_mask_capsule.png' if 'HE_glom' not in self.root else '_mask.png'

        mask = cv2.imread(os.path.join(self.root, mask_name),cv2.IMREAD_GRAYSCALE)

        if self.img_size:
            img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
            
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        return img2tensor(img/255.0), img2tensor(mask)
        


# AIDPATH
class aidpath_data(Dataset):
    def __init__(self, cfg, train=True, full_val = False, tfms=None, split_size=0.8, img_size=None):
        
        path_imgs = os.path.join(dirname, '../', cfg['imgs'])
        path_masks = os.path.join(dirname, '../', cfg['masks'])

        fnames = sorted(os.listdir(path_imgs))

        if not full_val:
            if train:
                fnames = fnames[:int(len(fnames)*split_size)]
            else:
                fnames = fnames[int(len(fnames)*split_size):]

        self.fnames = fnames

        self.path_imgs = path_imgs
        self.path_masks = path_masks
        self.tfms = tfms
        self.img_size = img_size
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.path_imgs,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.path_masks, fname),cv2.IMREAD_GRAYSCALE)

        if self.img_size:
            img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        return img2tensor(img/255.0), img2tensor(mask)
        


# Hubmpap 21 Kidney
class hubmap21_kidney_data(Dataset):
    def __init__(self, cfg, train=True, full_val = False, tfms=None, split_size=0.8, img_size=None):
        
        path_imgs = os.path.join(dirname, '../', cfg['imgs'])
        path_masks = os.path.join(dirname, '../', cfg['masks'])

        fnames = sorted(os.listdir(path_imgs))

        if not full_val:
            if train:
                fnames = fnames[:int(len(fnames)*split_size)]
            else:
                fnames = fnames[int(len(fnames)*split_size):]

        self.fnames = fnames

        self.path_imgs = path_imgs
        self.path_masks = path_masks
        self.tfms = tfms
        self.img_size = img_size
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.path_imgs,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.path_masks, fname),cv2.IMREAD_GRAYSCALE)

        if self.img_size:
            img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)

        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']

        return img2tensor(img/255.0), img2tensor(mask)

