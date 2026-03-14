"""SigLIP 2 on ImageNette (imagenet10) — vision-only self-supervised adaptation.

Trains a ViT-B/16 backbone from scratch using SigLIP 2's multi-positive sigmoid
loss combined with EMA self-distillation (a momentum teacher, similar in spirit to
BYOL / DINO).

Key differences from SigLIP:
* EMA teacher (via TeacherStudentWrapper) provides stable training targets.
* Multi-positive SigLIP2Loss: all augmented crops from the *same* source image
  are treated as positives, enabling multi-view training.

Reference: Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders
with Improved Semantic Understanding, Localisation, and Dense Features"
(arXiv:2502.14786).
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
# Augmentation — two global views (multi-crop can be added trivially)
# ---------------------------------------------------------------------------
siglip2_transform = transforms.MultiViewTransform(
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
    transform=siglip2_transform,
)
val_dataset = spt.data.HFDataset(
    "frgfm/imagenette",
    split="validation",
    revision="refs/convert/parquet",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 128
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
# Model — ViT-B/16 with EMA teacher (TeacherStudentWrapper)
# ---------------------------------------------------------------------------
_embed_dim = 768  # ViT-B hidden size

# Build student ViT backbone and wrap with EMA teacher
_student_backbone = spt.backbone.vit_hf(
    "base", patch_size=16, image_size=224, pretrained=False
)
backbone = spt.TeacherStudentWrapper(
    _student_backbone,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

# Projection MLP — both student and teacher share architecture via wrapper
_student_projector = nn.Sequential(
    nn.Linear(_embed_dim, 2048, bias=False),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256, bias=False),
    spt.utils.BatchNorm1dNoBias(256),
)
projector = spt.TeacherStudentWrapper(
    _student_projector,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.siglip2_forward,
    siglip2_loss=spt.losses.SigLIP2Loss(
        init_logit_scale=2.302585,  # log(10)
        init_logit_bias=-10.0,
    ),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "warmup_epochs": 10,
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
    probe=nn.Linear(_embed_dim, 10),
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
    input_dim=_embed_dim,
    k=20,
)

rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=_embed_dim,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
wandb_logger = WandbLogger(
    project="imagenet10-siglip2",
    name="siglip2-vitb",
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
