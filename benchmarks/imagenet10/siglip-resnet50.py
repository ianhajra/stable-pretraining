"""SigLIP (Sigmoid Loss for Language-Image Pre-Training) on ImageNette (imagenet10).

Trains ResNet-50 from scratch using the SigLIP sigmoid contrastive loss.
Every element of the NxN pairwise similarity matrix is treated as an
independent binary classification (positive on the diagonal, negative
elsewhere), with a learnable temperature and bias.

Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
(arXiv:2303.15343).
"""

import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.callbacks.rankme import RankMe

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# ---------------------------------------------------------------------------
# Augmentation pipeline — two views, same recipe as SimCLR
# ---------------------------------------------------------------------------
siglip_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_dir = get_data_dir("imagenet10")

train_dataset = spt.data.HFDataset(
    "frgfm/imagenette",
    split="train",
    revision="refs/convert/parquet",
    cache_dir=str(data_dir),
    transform=siglip_transform,
)
val_dataset = spt.data.HFDataset(
    "frgfm/imagenette",
    split="validation",
    revision="refs/convert/parquet",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
backbone = spt.backbone.from_torchvision("resnet50", low_resolution=False)
backbone.fc = nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 2048, bias=False),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256, bias=False),
    spt.utils.BatchNorm1dNoBias(256),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.siglip_forward,
    siglip_loss=spt.losses.SigLIPLoss(
        init_logit_scale=2.302585,  # log(10)
        init_logit_bias=-10.0,
    ),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * batch_size / 256,
            "weight_decay": 1e-4,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(2048, 10),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=10000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=2048,
    k=20,
)

rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=2048,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
wandb_logger = WandbLogger(
    project="imagenet10-siglip",
    name="siglip-resnet50",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=200,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, rankme],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    devices=1,
    accelerator="gpu",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
