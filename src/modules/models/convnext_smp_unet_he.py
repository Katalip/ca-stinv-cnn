import numpy as np
import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from .convnext import *
from .cov_attention import CovarianceAttention
from .isw import ISW
from .stinv_training import GradientReversal, domain_predictor


class RGB(nn.Module):
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


class ConvNextUNet(nn.Module):
    """Implements U-Net architecture with ConvNext backbone, stain-invariant training branch with channel attention
    from feature covariances

    Attributes:
        encoder (obj): Backbone from original ConvNext implementation
        encoder_cfg (dict): Config for the backbone
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
        pretrain = self.encoder_pretrain
        # print('load %s' % pretain)
        state_dict = torch.load(
            pretrain, map_location=lambda storage, loc: storage
        )[
            "model"
        ]  # True
        self.encoder.load_state_dict(state_dict, strict=False)  # True

    def __init__(
        self,
        encoder=ConvNeXt,
        encoder_cfg=dict(),
        encoder_pretrain=None,
        n_classes=1,
        stinv_training=False,
        stinv_enc_dim=0,
        n_domains=6,
        pool_out_size=6,
        domain_pool="max",
        filter_sensitive=False,
    ):
        """
        Args:
            encoder (obj): Backbone from original ConvNext implementation
            encoder_cfg (dict): Config for the backbone
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
        super(ConvNextUNet, self).__init__()
        decoder_dim = [256, 128, 64, 32, 16]
        self.n_classes = n_classes
        self.stinv_training = stinv_training
        self.stinv_enc_dim = stinv_enc_dim
        self.filter_sensitive = filter_sensitive
        self.mask_matrix = None
        self.iter_to_filtered_n = {}
        self.domain_pool = domain_pool

        # ----
        self.output_type = ["inference", "loss"]
        self.rgb = RGB()
        self.encoder_pretrain = encoder_pretrain

        conv_dim = 32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )

        self.encoder = encoder(**encoder_cfg)
        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = UnetDecoder(
            encoder_channels=[0, conv_dim] + encoder_dim,
            decoder_channels=decoder_dim,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim[-1], n_classes, kernel_size=1),
        )
        self.aux = nn.ModuleList(
            [
                nn.Conv2d(encoder_dim[i], n_classes, kernel_size=1, padding=0)
                for i in range(len(encoder_dim))
            ]
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
        fmaps_stem = encoder[0]
        encoder = encoder[1:]

        conv = self.conv(x)

        if 1:
            feature = encoder[
                ::-1
            ]  # reverse channels to start from head of encoder
            head = feature[0]
            skip = feature[1:] + (conv, None)
            d = self.decoder.center(head)

            decoder = []
            for i, decoder_block in enumerate(self.decoder.blocks):
                s = skip[i]
                d = decoder_block(d, s)
                decoder.append(d)
            last = d

        logit = self.logit(last)

        output = {}
        output["raw"] = logit

        if "loss" in self.output_type:
            output["bce_loss"] = F.binary_cross_entropy_with_logits(
                logit, batch["mask"]
            )

        if "inference" in self.output_type:
            probability_from_logit = torch.sigmoid(logit)
            output["probability"] = probability_from_logit

        # For stain predictions
        if "stain_info" in self.output_type:
            fmaps = fmaps_stem

            if self.filter_sensitive:
                self.encoder.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        fmaps_transformed = self.encoder.downsample_layers[0](
                            batch["image_tf"]
                        )

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


def test_network():
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
        "organ": torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
    }
    batch = {k: v.cuda() for k, v in batch.items()}

    net = ConvNextUNet().cuda()
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
