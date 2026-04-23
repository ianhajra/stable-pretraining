"""SimCLR single-transform ablation for financial imagery (ViT-Tiny backbone).

Augmentation pipeline is reduced to exactly one transform per run.
Window size is fixed at 63 for all runs in this ablation.
All other settings are identical to ablation_window_size.py.
"""

import argparse
import lightning as pl
import torch
import torchmetrics
from torch import nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import timm
import stable_pretraining as spt
from stable_pretraining.callbacks import RankMe
from stable_pretraining import forward
from datasets import load_from_disk

import stable_pretraining.data.transforms as spt_transforms
import stable_pretraining.data.transforms_custom as transforms_custom

# ----------------------
# Argument parsing
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR single-transform ablation for financial imagery (ViT-Tiny)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HFDataset root for the encoding and window size")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for probe (11 for S&P 500, 5 for FF)")
    parser.add_argument("--window_size", type=int, required=True, help="Window size (always 63 for this ablation)")
    parser.add_argument("--encoding", type=str, required=True, help="Encoding type (gaf_mtf, candlestick, heatmap)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (sp500, ff)")
    parser.add_argument(
        "--transform", type=str, required=True,
        choices=[
            "random_resized_crop",
            "horizontal_flip",
            "color_jitter",
            "random_grayscale",
            "gaussian_blur",
            "magnitude_scaling",
            "gaussian_noise",
            "temporal_masking",
        ],
        help="Single augmentation transform to ablate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()

args = parse_args()

# ----------------------
# Reproducibility
# ----------------------
pl.seed_everything(args.seed, workers=True)


# ----------------------
# Transforms
# ----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_view(transform_name):
    """Build a single-transform view pipeline for the given transform name.

    For transforms other than random_resized_crop, a plain Resize(224) is used
    as neutral preprocessing to satisfy the ViT-Tiny input size requirement.
    This is not counted as the ablated transform.
    """
    if transform_name == "random_resized_crop":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.RandomResizedCrop((224, 224)),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "horizontal_flip":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.RandomHorizontalFlip(p=0.5),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "color_jitter":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "random_grayscale":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.RandomGrayscale(p=0.2),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "gaussian_blur":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.PILGaussianBlur(sigma=[0.1, 2.0], p=1.0),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "magnitude_scaling":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.WrapTorchTransform(
                transforms_custom.MagnitudeScaling(scale_range=(0.8, 1.2), p=0.5)
            ),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "gaussian_noise":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.WrapTorchTransform(
                transforms_custom.GaussianNoiseInjection(sigma=0.05, p=0.5)
            ),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    elif transform_name == "temporal_masking":
        return spt_transforms.Compose(
            spt_transforms.RGB(),
            spt_transforms.Resize((224, 224)),
            spt_transforms.WrapTorchTransform(
                transforms_custom.RandomTemporalMasking(mask_ratio_range=(0.10, 0.20), p=0.5)
            ),
            spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


train_transform = spt_transforms.MultiViewTransform([
    _make_view(args.transform),
    _make_view(args.transform),
])

val_transform = spt_transforms.Compose(
    spt_transforms.RGB(),
    spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
)

# ----------------------
# Dataset: robust label selection
# ----------------------
rename_columns = None
if args.dataset == "ff":
    rename_columns = None
elif args.dataset == "sp500":
    rename_columns = None

# ----------------------
# Dataset: robust label selection
# ----------------------
class FinancialImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Convert FF factor labels from 1-indexed to 0-indexed if present (before transform)
        for ff_col in ['label_mktrf', 'label_smb', 'label_hml', 'label_rmw', 'label_cma']:
            if ff_col in sample:
                sample[ff_col] = sample[ff_col] - 1
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return len(self.dataset)

remove_columns = None
if args.dataset == "ff":
    remove_columns = ["start_date", "end_date"]
elif args.dataset == "sp500":
    remove_columns = ["ticker", "sector", "start_date", "end_date"]

# Load the full DatasetDict once, then select splits
full_dataset = load_from_disk(args.data_dir)
train_split = full_dataset["train"]
val_split = full_dataset["val"]

train_ds = FinancialImageDataset(train_split, transform=train_transform)
val_ds = FinancialImageDataset(val_split, transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
)

data = spt.data.DataModule(train=train_loader, val=val_loader)

# ----------------------
# Backbone & Projector
# ----------------------
backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
embed_dim = backbone.num_features  # 192 for ViT-Tiny

projector = nn.Sequential(
    nn.Linear(embed_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 128),
)

# ----------------------
# Module
# ----------------------
module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.simclr_forward,
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.07),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 1.5e-4,
            "weight_decay": 0.05,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# ----------------------
# Probes
# ----------------------
probes = []
if args.dataset == "ff":
    # Fama–French 5-factor: probe on each factor separately
    ff_factors = [
        ("mktrf", "label_mktrf"),
        ("smb", "label_smb"),
        ("hml", "label_hml"),
        ("rmw", "label_rmw"),
        ("cma", "label_cma"),
    ]
    for factor_name, col in ff_factors:
        probes.append(
            spt.callbacks.OnlineProbe(
                module,
                name=f"linear_probe_{factor_name}",
                input="embedding",
                target=col,
                probe=torch.nn.Linear(embed_dim, args.num_classes),
                loss=torch.nn.CrossEntropyLoss(),
                metrics={
                    "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
                        args.num_classes, average="macro"
                    ),
                },
            )
        )
else:
    probes.append(
        spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=torch.nn.Linear(embed_dim, args.num_classes),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={
                "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
                    args.num_classes, average="macro"
                ),
            },
        )
    )

if args.dataset == "ff":
    knn_target = "label_mktrf"
else:
    knn_target = "label"
knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target=knn_target,
    queue_length=10000,
    metrics={
        "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
            args.num_classes, average="macro"
        )
    },
    input_dim=embed_dim,
    k=20,
)

# ----------------------
# AverageFactorAccuracy callback
# ----------------------
if args.dataset == "ff":
    class AverageFactorAccuracy(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            keys = [
                "eval/linear_probe_mktrf_balanced_acc",
                "eval/linear_probe_smb_balanced_acc",
                "eval/linear_probe_hml_balanced_acc",
                "eval/linear_probe_rmw_balanced_acc",
                "eval/linear_probe_cma_balanced_acc",
            ]
            vals = [trainer.callback_metrics.get(k) for k in keys]
            # Robust float conversion
            def to_float(v):
                if v is None:
                    return float('nan')
                if hasattr(v, 'item'):
                    return float(v.item())
                return float(v)
            vals = [to_float(v) for v in vals]
            avg = float('nan') if any([v != v for v in vals]) else sum(vals) / len(vals)
            pl_module.log("eval/linear_probe_ff_avg_balanced_acc", avg, on_epoch=True, prog_bar=False, sync_dist=True)
    avg_factor_acc_cb = AverageFactorAccuracy()
else:
    avg_factor_acc_cb = None

# ----------------------
# RankMe callback
# ----------------------
rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=embed_dim,
)

# ----------------------
# Wandb logger
# ----------------------
run_name = f"ablation_transform_{args.transform}_{args.encoding}_{args.dataset}_w{args.window_size:03d}_seed{args.seed}"
wandb_logger = WandbLogger(
    project="thesis-ablations",
    name=run_name,
    log_model=False,
    config=vars(args),
)

# ----------------------
# Trainer
# ----------------------
lr_monitor = LearningRateMonitor(logging_interval="step")
trainer_callbacks = [knn_probe, *probes, rankme, lr_monitor]
if avg_factor_acc_cb is not None:
    trainer_callbacks.append(avg_factor_acc_cb)
trainer = pl.Trainer(
    max_epochs=100,
    num_sanity_val_steps=0,
    callbacks=trainer_callbacks,
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

# ----------------------
# Run
# ----------------------
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()