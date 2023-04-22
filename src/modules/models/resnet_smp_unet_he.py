import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet

from .cov_attention import CovarianceAttention
from .isw import ISW
from .stinv_training import GradientReversal, domain_predictor


class RGB(nn.Module):
    """Used for standardizing input images to ImageNet stats"""

    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

    def __init__(
        self,
    ):
        super(RGB, self).__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(
            self.mean.shape
        )
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(
            self.std.shape
        )

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
        print("load %s" % pretain)
        state_dict = torch.load(
            pretain, map_location=lambda storage, loc: storage
        )  # True
        print(self.encoder.load_state_dict(state_dict, strict=False))  # True

    def __init__(
        self,
        encoder_pretrain="imagenet",
        n_classes=1,
        stinv_training=False,
        stinv_enc_dim=1,
        n_domains=6,
        pool_out_size=6,
        domain_pool="max",
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
        self.filter_sensitive = filter_sensitive
        self.mask_matrix = None
        self.iter_to_filtered_n = {}
        self.domain_pool = domain_pool

        self.output_type = ["inference", "loss"]
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
            if self.domain_pool == "max":
                self.downsample = nn.AdaptiveMaxPool2d(
                    (pool_out_size, pool_out_size)
                )

            elif self.domain_pool == "avg":
                self.downsample = nn.AvgPool2d(64, stride=64)

            elif self.domain_pool == "seq_conv":
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        encoder_dim[stinv_enc_dim],
                        encoder_dim[stinv_enc_dim],
                        kernel_size=4,
                        stride=4,
                        bias=False,
                    ),
                    nn.BatchNorm2d(encoder_dim[stinv_enc_dim]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        encoder_dim[stinv_enc_dim],
                        encoder_dim[stinv_enc_dim],
                        kernel_size=4,
                        stride=4,
                        bias=False,
                    ),
                    nn.BatchNorm2d(encoder_dim[stinv_enc_dim]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        encoder_dim[stinv_enc_dim],
                        encoder_dim[stinv_enc_dim],
                        kernel_size=4,
                        stride=4,
                        bias=False,
                    ),
                )

            self.cov_attention = CovarianceAttention(
                encoder_dim[stinv_enc_dim]
            )
            self.stain_predictor = domain_predictor(
                encoder_dim[stinv_enc_dim] * pool_out_size * pool_out_size,
                n_domains,
            )
            self.isw = ISW()

    def forward(self, batch):
        x = batch["image"]
        x = self.rgb(x)

        encoder = self.encoder(x)

        last = self.decoder(*encoder)

        logit = self.logit(last)

        output = {}
        output["raw"] = logit

        if "loss" in self.output_type:
            if self.n_classes == 1:
                output["bce_loss"] = F.binary_cross_entropy_with_logits(
                    logit, batch["mask"]
                )

        if "inference" in self.output_type:
            probability_from_logit = torch.sigmoid(logit)
            output["probability"] = probability_from_logit

        # For stain predictions
        if "stain_info" in self.output_type:
            fmaps = encoder[self.stinv_enc_dim]

            if self.filter_sensitive:
                self.encoder.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        fmaps_transformed = self.encoder.conv1(
                            batch["image_tf"]
                        )
                        fmaps_transformed = self.encoder.bn1(fmaps_transformed)
                        fmaps_transformed = self.encoder.relu(
                            fmaps_transformed
                        )

                        if self.stinv_enc_dim > 1:
                            for stage_idx in range(2, self.stinv_enc_dim + 1):
                                fmaps_transformed = getattr(
                                    self.encoder, f"layer{stage_idx - 1}"
                                )(fmaps_transformed)

                self.encoder.train()

                # with torch.cuda.amp.autocast(enabled=False):
                #     isw_loss = self.isw(fmaps, fmaps_transformed)
                # output['isw_loss'] = isw_loss

                with torch.cuda.amp.autocast(enabled=False):
                    channel_attention = self.cov_attention(
                        fmaps, fmaps_transformed
                    )

                fmaps = fmaps * channel_attention

            encoder_out = self.downsample(fmaps)

            encoder_out = encoder_out.view(encoder_out.shape[0], -1)
            reverse_feature = GradientReversal.apply(
                encoder_out, batch["alpha"]
            )

            output_domain = self.stain_predictor(reverse_feature)
            output["stain_info"] = output_domain

        return output


def test_network(pretrained=False):
    batch_size = 2
    image_size = 768

    # ---
    batch = {
        "image": torch.from_numpy(
            np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))
        ).float(),
        "mask": torch.from_numpy(
            np.random.choice(2, (batch_size, 1, image_size, image_size))
        ).float(),
    }
    batch = {k: v.cuda() for k, v in batch.items()}

    net = ResUNet().cuda()

    if pretrained:
        net.load_pretrain()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)

    print("batch")
    for k, v in batch.items():
        print("%32s :" % k, v.shape)

    print("output")
    for k, v in output.items():
        if "loss" not in k:
            print("%32s :" % k, v.shape)
    for k, v in output.items():
        if "loss" in k:
            print("%32s :" % k, v.item())


if __name__ == "__main__":
    test_network()
