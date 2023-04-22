import warnings

warnings.filterwarnings("ignore")

import os
from argparse import ArgumentParser

import torch
import yaml
from torch.utils.data import DataLoader

from modules.augs import get_aug_a1, get_aug_selection
from modules.dataset import hpa_hubmap_data_he
from modules.models.convnext_smp_unet_he import ConvNextUNet
from modules.models.resnet_smp_unet_he import ResUNet
from run_components import train_model


def parse_config():
    argparser = ArgumentParser()
    argparser.add_argument("config_name", help="Name of the yml config file")
    args = argparser.parse_args()

    # Config File
    dirname = os.path.dirname(__file__)
    cfg_name = args.config_name
    with open(os.path.join(dirname, f"cfg/{cfg_name}")) as f:
        cfg = yaml.safe_load(f)

    cfg["Train"]["save_path"] = os.path.join(
        dirname, f"../experiments/{cfg['Train'].get('experiment_name')}"
    )

    return cfg


def create_save_folders(cfg):
    try:
        os.makedirs("../experiments/", exist_ok=True)
    except Exception as e:
        print(f"Folder creation error: {e}")

    try:
        os.makedirs(cfg["Train"].get("save_path"), exist_ok=True)
    except Exception as e:
        print(f"Folder creation error: {e}")


def main(cfg):
    train_data = hpa_hubmap_data_he(
        cfg=cfg["Data"],
        train=True,
        tfms=get_aug_a1(),
        selection_tfms=get_aug_selection(),
    )
    val_data = hpa_hubmap_data_he(cfg=cfg["Data"], train=False)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=cfg["Loader"].get("batch_size"),
        num_workers=cfg["Loader"].get("num_workers"),
    )
    val_loader = DataLoader(
        val_data,
        shuffle=True,
        batch_size=cfg["Loader"].get("batch_size"),
        num_workers=cfg["Loader"].get("num_workers"),
    )

    assert cfg["Architecture"].get("encoder") in ("resnet50", "convnext_tiny")

    if cfg["Architecture"].get("encoder") == "resnet50":
        model = ResUNet(
            stinv_training=True,
            stinv_enc_dim=1,
            pool_out_size=6,
            filter_sensitive=True,
            n_domains=6,
            domain_pool="max",
        ).cuda()
    elif cfg["Architecture"].get("encoder") == "convnext_tiny":
        model = ConvNextUNet(
            stinv_training=True,
            stinv_enc_dim=0,
            pool_out_size=6,
            filter_sensitive=True,
            n_domains=6,
            domain_pool="max",
            encoder_pretrain=cfg["Architecture"].get("weights"),
        ).cuda()
        model.load_pretrain()

    if cfg["Train"].get("start_epoch") > 0:
        model.load_state_dict(
            torch.load(
                f"{cfg['Train'].get('save_path')}/epoch_{cfg['Train'].get('start_epoch')}.pth"
            )
        )
        print(f"Starting from epoch: {cfg['Train'].get('start_epoch')}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=1e-3
    )
    scaler = torch.cuda.amp.GradScaler()

    train_model(model, train_loader, val_loader, optimizer, scaler, cfg)


if __name__ == "__main__":
    cfg = parse_config()
    create_save_folders(cfg)
    main(cfg)
