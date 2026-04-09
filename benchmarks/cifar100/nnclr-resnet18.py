"""NNCLR training on CIFAR100."""

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import types

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.callbacks.queue import OnlineQueue
from stable_pretraining.data import transforms

nnclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR100),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR100),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR100),
)

train_dataset = spt.data.HFDataset(
    "uoft-cs/cifar100",
    split="train",
    transform=nnclr_transform,
    rename_columns={"img": "image", "fine_label": "label"},
    remove_columns=["coarse_label"],
)
val_dataset = spt.data.HFDataset(
    "uoft-cs/cifar100",
    split="test",
    transform=val_transform,
    rename_columns={"img": "image", "fine_label": "label"},
    remove_columns=["coarse_label"],
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=4,
    drop_last=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=2,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)

backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

predictor = nn.Sequential(
    nn.Linear(256, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 256),
)


module = spt.Module(
    backbone=backbone,
    projector=projector,
    predictor=predictor,
    # Corrected Usage: Use the imported 'forward' module
    forward=forward.nnclr_forward,
    nnclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {"type": "LARS", "lr": 5, "weight_decay": 1e-6},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    },
    hparams={
        "support_set_size": 6000,
        "projection_dim": 256,
    },
)

_orig_forward = module.forward


def _forward_with_views(self, batch, stage):
    out = _orig_forward(batch, stage)
    if self.training and "embedding" in out:
        if "views" in batch:
            n = batch["views"][0]["image"].shape[0]
        elif "image" not in batch:
            n = next(iter(batch.values()))["image"].shape[0]
        else:
            n = out["embedding"].shape[0] // 2
        out["embedding_view1"] = out["embedding"][:n]
        out["embedding_view2"] = out["embedding"][n : 2 * n]
    return out


module.forward = types.MethodType(_forward_with_views, module)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 100),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=512,
    k=10,
)

support_queue = OnlineQueue(
    key="nnclr_support_set",
    queue_length=module.hparams.support_set_size,
    dim=module.hparams.projection_dim,
)


rankme = spt.callbacks.RankMe(
    name="rankme",
    target="embedding",
    queue_length=8192,
    target_shape=512,
)

rerankme = spt.callbacks.ReRankMe(
    name="rerankme",
    target_view1="embedding_view1",
    target_view2="embedding_view2",
    queue_length=8192,
    target_shape=512,
)

wandb_logger = WandbLogger(
    entity="ianhajra-brown-university",
    project="rerankme",
    name="nnclr-resnet18-cifar100",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=400,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, support_queue, rankme, rerankme],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
