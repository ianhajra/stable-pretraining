"""VICReg training on CIFAR-10."""

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path

import types

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.callbacks.cifar10c import CIFAR10CCallback
from stable_pretraining.data import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "benchmarks"))
from utils import get_data_dir

RUN_NAME = "vicreg-resnet18-inv5"


vicreg_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=vicreg_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=2,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.vicreg_forward,
    vicreg_loss=spt.losses.VICRegLoss(
        sim_coeff=5.0,
        std_coeff=25.0,
        cov_coeff=1.0,
    ),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
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
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
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
    name=RUN_NAME,
    log_model=False,
    save_dir=str(Path.home() / "scratch" / "rerankme" / "logs"),
)

lr_monitor = LearningRateMonitor(logging_interval="step")

checkpoint_callback = ModelCheckpoint(
    dirpath=str(Path.home() / "scratch" / "rerankme" / "checkpoints" / RUN_NAME),
    filename="{epoch:03d}",
    save_top_k=-1,
    every_n_epochs=25,
    save_last=True,
)

trainer = pl.Trainer(
    max_epochs=300,
    num_sanity_val_steps=0,
    callbacks=[
        knn_probe,
        linear_probe,
        lr_monitor,
        rankme,
        rerankme,
        checkpoint_callback,
        CIFAR10CCallback(every_n_epochs=25),
    ],
    precision="16-mixed",
    logger=wandb_logger,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
