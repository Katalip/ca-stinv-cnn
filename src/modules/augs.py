from torchvision import transforms as torchtf
from albumentations import *
import numpy as np
from .stainspec.randstainNA.randstainna import RandStainNA
from .macenko_torch import TorchMacenkoNormalizer
import os

dirname = os.path.dirname(__file__)

# Modified version, local import
normalizer = TorchMacenkoNormalizer()

T = torchtf.Compose([
    torchtf.ToTensor(),
    torchtf.Lambda(lambda x: x*255)
])

class stain_aug:
    """This class represents stain augmentation by perturbing optimal stain vectors from Macenko normalization 
    
    Attributes:
        normalizer (obj): Instance of TorchMacenkoNormalizer
        T (obj): Composition of transforms from torchvision for basic preprocessing
        p (float): probability of applying stain augmentation
        sigma1 (float): scaling value is chosen uniformly from (1 - sigma1, 1 + sigma1)
        sigma2 (float): shifting value is chosen uniformly from (-sigma2, sigma2)
        type (str): 'light' for linear scale and shift, 'strong' for element-wise perturbations 

    """

    def __init__(self, normalizer, T, p, sigma1, sigma2, type='light'):
        """
        Args:
            normalizer (obj): Instance of TorchMacenkoNormalizer
            T (obj): Composition of transforms from torchvision for basic preprocessing
            p (float): probability of applying stain augmentation
            sigma1 (float): scaling value is chosen uniformly from (1 - sigma1, 1 + sigma1)
            sigma2 (float): shifting value is chosen uniformly from (-sigma2, sigma2)
            type (str, optional): 'light' for linear scale and shift, 'strong' for element-wise perturbations. Defaults to 'light'
        """

        self.normalizer = normalizer
        self.T = T
        self.p = p
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.type = type

    def augment(self, image):
        """Applies stain augmentation to the input image

        Args:
            image (np.ndarray): Input image 

        Returns:
            np.ndarray: Augmented image
        """


        if self.normalizer:
            
            src = T(image)

            try:
                stain_vectors_src, _, maxC_src = normalizer.get_stain_matrix(src)
            except:
                print("Lapack error -> returning img itself")
                return image

            if self.type == 'light':
                alpha_sig = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
                beta_sig = np.random.uniform(-self.sigma2, self.sigma2)
                img, _, _ = normalizer.normalize(src, maxC_src, stain_vectors_src * alpha_sig + beta_sig, stains=False)
            
            elif self.type == 'strong':
                for index in [0, 1, 2]:
                    alpha_sig = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
                    beta_sig = np.random.uniform(-self.sigma2, self.sigma2)
                    stain_vectors_src[index] = stain_vectors_src[index] * alpha_sig + beta_sig

                img, _, _ = normalizer.normalize(src, maxC_src, stain_vectors_src, stains=False)

            img = img.numpy().astype(np.uint8)
        else:
            img = image

        return img

    def __call__( self, force_apply, image, mask) : 
        """Wrapper for integrating with Albumentations compose 

        Args:
            force_apply (bool): Albumentations requires this argument for proper functioning
            image (np.ndarray): Input image
            mask (np.ndarray): Segmentation mask

        Returns:
            np.ndarray: Augmented image
        """

        if np.random.rand(1) < self.p:
            return {'image':self.augment(image), 'mask':mask}
        else:
            return {'image':image, 'mask':mask}


def get_aug_a1(p=1.0):

    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=35, p=0.6),
        RandomBrightnessContrast((-0.2, 0.2), 0.2, p=0.6),  
    ], p=p)


def get_aug_a1_hsv(p=1.0):
    
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=35, p=0.6),
        RandomBrightnessContrast((-0.2, 0.2), 0.2, p=0.6),
        HueSaturationValue(15, 15, 5, p=0.5),
    ], p=p)


def get_aug_a1_randstna(p=1.0):
    
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=35, p=0.6),
        RandomBrightnessContrast((-0.2, 0.2), 0.2, p=0.6),
        RandStainNA(yaml_file='modules/stainspec/randstainNA/HPA.yaml', std_hyper=-0.3, probability=0.5,
                            distribution='normal', is_train=True),
    ], p=p)


def get_aug_a1_staug_light(p=1.0):
    
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=35, p=0.6),
        RandomBrightnessContrast((-0.2, 0.2), 0.2, p=0.6),
        stain_aug(normalizer, T, 0.5, 0.3, 0.3, type='light'),
    ], p=p)


def get_aug_a1_staug_strong(p=1.0):
    
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=35, p=0.6),
        RandomBrightnessContrast((-0.2, 0.2), 0.2, p=0.6),
        stain_aug(normalizer, T, 0.5, 0.2, 0.2, type='strong'),
    ], p=p)


def get_aug_selection(p=1.0):
    return Compose([
        # For ISW
        # ColorJitter(p=1),
        # GaussianBlur(p=1),

        # For our method
        stain_aug(normalizer, T, 1.0, 0.35, 0.35, type='strong')
    ], p=p)