from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch import Unet

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import kmeans1d


# Reverse autograd. Adapted from:  https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        alpha = ctx.alpha
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


class domain_predictor(torch.nn.Module):
    """This class is used for predicting domain/stain information 

     Attributes:
        fc_input_features (int): N from B x N after flattening feature maps B x C x H x W
        n_centers (int): The number of domains or in this case n of entries in the stain matrix 

    """
    def __init__(self, fc_input_features, n_centers):
        """
        Args:
            fc_input_features (int): N from B x N after flattening feature maps B x C x H x W
            n_centers (int): The number of domains or in this case n of entries in the stain matrix
        """
        super(domain_predictor, self).__init__()
        # domain predictor
        self.fc_feat_in = fc_input_features
        self.n_centers = n_centers
        self.E = fc_input_features // 2		
        self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.linear = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.E)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.domain_embedding(x)
        
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(out)

        domain_prob = self.linear(out)

        return domain_prob


# from https://github.com/ilmaro8/HE_adversarial_CNN/blob/main/train/Training_script_domain_adversarial_regressor.py
class domain_predictor_original(torch.nn.Module):
    def __init__(self, fc_input_features, n_centers):
        super(domain_predictor_original, self).__init__()
        # domain predictor
        self.fc_feat_in = fc_input_features
        self.n_centers = n_centers
        self.E = fc_input_features // 2	
        self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.linear = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.E)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.domain_embedding(x)
        out = self.dropout(out)
        domain_prob = self.linear(out)

        return domain_prob



class CovarianceAttention(nn.Module):
    """Used for computing channel attention based on feature covariances 

    Attributes:
        in_channels (int): The number of input channels (C) from B x C x H x W
    """
    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): The number of input channels (C) from B x C x H x W
        """
        super(CovarianceAttention, self).__init__()
        self.fc = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_tf):
        """Copmuter channel attention scores from feature covariances
        Steps:
        1) Compute individiaul covariance matrices (COV, COV') of feature maps of the input image (F) and its augmented version (F')
        2) Compute variance matrix (V) that represents difference between two covariance matrices (COV, COV')
        3) Apply linear transformation (FC) to get channel weights
        4) Map these values to the range (0, 1) with sigmoid  

        Args:
            x (torch.tensor): Feature maps of the input image after n-th encoder stage, n is optional
            x_tf (torch.tensor): Feature maps of the transformed/augmendted input image after n-th encoder stage, n is optional

        Returns:
            torch.tensor: Channel weights
        """

        cov_base, _ = get_covariance_matrix(x)
        cov_tf, _ = get_covariance_matrix(x_tf)
    
        var_matrix = get_var_matrix(cov_base, cov_tf)

        channel_attention = self.fc(var_matrix).unsqueeze(3) 
        channel_attention = self.sigmoid(channel_attention)

        return channel_attention


# https://github.com/shachoi/RobustNet/blob/main/network/instance_whitening.py
def get_covariance_matrix(f_map, eye=None):
    """Computes covariance matrix of the given feature maps

    Args:
        f_map (torch.tensor): Input feature maps
        eye (torch.tensor, optional): Identity matrix for denominator stabilization. Defaults to None.

    Returns:
        torch.tensor: Covariance matrix
        int: Batch size
    """
    eps = 1e-5
    f_map = f_map.to(torch.float32)
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B


def get_var_matrix(cov, cov_tf):
    """Computes variance matrix from two covariance matrices

    Args:
        cov (torch.tensor): Covariance matrix of feature maps of the input image
        cov_tf (torch.tensor): Covariance matrix of feature maps of the transformed/augmented input image

    Returns:
        torch.tensor: Variance matrix
    """
    mu = 0.5*(cov + cov_tf)
    sigma_sq = 0.5*((cov - mu)**2 + (cov_tf - mu)**2)

    return sigma_sq


# reimplemented from https://github.com/shachoi/RobustNet/blob/main/network/cov_settings.py
class ISW(nn.Module):
    def __init__(self, clusters=50):
        super(ISW, self).__init__()
        self.clusters = clusters


    def get_mask_matrix(self, var_matrix):

        dim = var_matrix.shape[-1]
        var_flatten = torch.flatten(var_matrix)
        
        clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters) # 50 clusters
        num_sensitive = var_flatten.size()[0] - clusters.count(0)  # 1: Insensitive Cov, 2~50: Sensitive Cov
        _, indices = torch.topk(var_flatten, k=int(num_sensitive))

        mask_matrix = torch.flatten(torch.zeros(dim, dim).cuda())
        mask_matrix[indices] = 1

        mask_matrix = mask_matrix.view(dim, dim)

        return mask_matrix


    def forward(self, x, x_tf):

        x = F.instance_norm(x)
        x_tf = F.instance_norm(x_tf)

        cov_base, _ = get_covariance_matrix(x)
        cov_tf, _ = get_covariance_matrix(x_tf)
    
        var_matrix = get_var_matrix(cov_base, cov_tf)
        var_matrix = torch.mean(var_matrix, dim=0, keepdim=True)

        mask = self.get_mask_matrix(torch.triu(var_matrix, diagonal=1))
        
        cov_base = cov_base * mask
        num_remove_cov = mask.sum() + 0.0001
        B = cov_base.shape[0]

        off_diag_sum = torch.sum(torch.abs(cov_base), dim=(1,2), keepdim=True) # - margin # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss



class RGB(nn.Module):
    """Used for standardizing input images to ImageNet stats
    """
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
    
    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class ResUNet(nn.Module):
    """Implements U-Net architecture with ResNet backbone, stain-invariant training branch with channel attention
    from feature covariances

    Attributes:
        encoder_pretrain (str, optional): Choice of pretrained weights. Additional options are available in 
                         segmentation models pytorch package documentation. Defaults to 'imagenet'.
        n_classes (int, optional): N of classes being predicted. Defaults to 1.
        stinv_training (bool, optional): To use stain-invariant training or not. Defaults to False.
        stinv_enc_dim (int, optional): Which encoder stage outputs to use for stain-invariant training. Defaults to 1.
        n_domains (int, optional): N of entries in the stain matrix. Defaults to 6.
        pool_out_size (int, optional): Output size after downsampling feature maps of n-th encoder stage. Defaults to 6.
        domain_pool (str, optional): Type of layer to use for downsampling. Defaults to 'max'.
        filter_sensitive (bool, optional): To use channel attention from feature covariance or not. Defaults to False.
    """
    
    def load_pretrain(self):
        pretain = self.encoder_pretrain
        print('load %s' % pretain)
        state_dict = torch.load(pretain, map_location=lambda storage, loc: storage)  # True
        print(self.encoder.load_state_dict(state_dict, strict=False))  # True
    
    def __init__(self,
                 encoder_pretrain='imagenet',
                 n_classes = 1,
                 stinv_training = False,
                 stinv_enc_dim=1,
                 n_domains = 6,
                 pool_out_size = 6,
                 domain_pool = 'max',
                 filter_sensitive=False,
                 ):
        """
        Args:
            encoder_pretrain (str, optional): Choice of pretrained weights. Additional options are available in 
                         segmentation models pytorch package documentation. Defaults to 'imagenet'.
            n_classes (int, optional): N of classes being predicted. Defaults to 1.
            stinv_training (bool, optional): To use stain-invariant training or not. Defaults to False.
            stinv_enc_dim (int, optional): Which encoder stage outputs to use for stain-invariant training. Defaults to 1.
            n_domains (int, optional): N of entries in the stain matrix. Defaults to 6.
            pool_out_size (int, optional): Output size after downsampling feature maps of n-th encoder stage. Defaults to 6.
            domain_pool (str, optional): Type of layer to use for downsampling, 'max' for max pooling, 'avg' for avgpooling,
                        'seq_conv' for conv layer. Defaults to 'max'.
            filter_sensitive (bool, optional): To use channel attention from feature covariance or not. Defaults to False.
        """
        super(ResUNet, self).__init__()
        decoder_dim = [256, 128, 64, 32, 16]
        self.n_classes = n_classes
        self.stinv_training = stinv_training
        self.stinv_enc_dim = stinv_enc_dim
        self.filter_sensitive=filter_sensitive
        self.mask_matrix = None
        self.iter_to_filtered_n = {}
        self.domain_pool = domain_pool
        
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()
        self.encoder_pretrain = encoder_pretrain

        model = Unet(encoder_name="resnet50", encoder_weights=encoder_pretrain)
        self.encoder = model.encoder
        self.decoder = model.decoder

        encoder_dim = self.encoder.out_channels
        
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim[-1], n_classes, kernel_size=1),
        )

        if self.stinv_training:
            
            if self.domain_pool == 'max':
                self.downsample = nn.AdaptiveMaxPool2d((pool_out_size, pool_out_size))

            elif self.domain_pool == 'avg':
                self.downsample = nn.AvgPool2d(64, stride=64)
            
            elif self.domain_pool == 'seq_conv':
                self.downsample = nn.Sequential(
                nn.Conv2d(encoder_dim[stinv_enc_dim], encoder_dim[stinv_enc_dim], kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(encoder_dim[stinv_enc_dim]),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dim[stinv_enc_dim], encoder_dim[stinv_enc_dim], kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(encoder_dim[stinv_enc_dim]),
                nn.ReLU(inplace=True),
                nn.Conv2d(encoder_dim[stinv_enc_dim], encoder_dim[stinv_enc_dim], kernel_size=4, stride=4, bias=False)
            )

            self.cov_attention = CovarianceAttention(encoder_dim[stinv_enc_dim])
            self.stain_predictor = domain_predictor(encoder_dim[stinv_enc_dim]*pool_out_size*pool_out_size, n_domains)
            self.isw = ISW()


    def forward(self, batch):
        
        x = batch['image']
        x = self.rgb(x)
        
        encoder = self.encoder(x)

        last = self.decoder(*encoder)
        
        logit = self.logit(last)
        
        output = {}
        output['raw'] = logit

        if 'loss' in self.output_type:
            if self.n_classes == 1:
                output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
        
        
        if 'inference' in self.output_type:
            probability_from_logit = torch.sigmoid(logit)
            output['probability'] = probability_from_logit

        # For stain predictions
        if 'stain_info' in self.output_type:
            
            fmaps = encoder[self.stinv_enc_dim]

            if self.filter_sensitive:

                self.encoder.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled = True):

                        fmaps_transformed = self.encoder.conv1(batch['image_tf'])
                        fmaps_transformed = self.encoder.bn1(fmaps_transformed)
                        fmaps_transformed = self.encoder.relu(fmaps_transformed)
                        
                        if self.stinv_enc_dim > 1:
                            for stage_idx in range(2, self.stinv_enc_dim + 1):
                                fmaps_transformed = getattr(self.encoder, f'layer{stage_idx - 1}')(fmaps_transformed)

                self.encoder.train()

                # with torch.cuda.amp.autocast(enabled=False):
                #     isw_loss = self.isw(fmaps, fmaps_transformed)  
                # output['isw_loss'] = isw_loss 

                with torch.cuda.amp.autocast(enabled=False):
                    channel_attention = self.cov_attention(fmaps, fmaps_transformed)

                fmaps = fmaps * channel_attention
            
            encoder_out = self.downsample(fmaps)

            encoder_out = encoder_out.view(encoder_out.shape[0], -1)
            reverse_feature = GradientReversal.apply(encoder_out, batch['alpha'])
            
            output_domain = self.stain_predictor(reverse_feature)
            output['stain_info'] = output_domain

        return output


def test_network(pretrained=False):
    batch_size = 2
    image_size = 768 #800
    
    # ---
    batch = {
        'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
        'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float()
    }
    batch = {k: v.cuda() for k, v in batch.items()}
    
    net = ResUNet().cuda()
  
    if pretrained:
        net.load_pretrain()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)
    
    print('batch')
    for k, v in batch.items():
        print('%32s :' % k, v.shape)
    
    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print('%32s :' % k, v.shape)
    for k, v in output.items():
        if 'loss' in k:
            print('%32s :' % k, v.item())


if __name__ == '__main__':
    test_network()