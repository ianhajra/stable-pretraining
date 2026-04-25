"""LeJEPA training for financial imagery (ViT-Tiny backbone).

Multi-view invariance + Epps-Pulley goodness-of-fit (SIGReg).
Uses 2 global views (224×224) + 6 local views (96×96).

Augmentation pipeline is built dynamically from --augmentations; dataset and
encoding are controlled entirely by CLI arguments.  Invoke via run_lejepa.sh
or directly:

    python lejepa.py \\
        --data_dir /oscar/scratch/ihajra/finance/sp500_encoded/gaf_mtf/w126 \\
        --num_classes 11 \\
        --window_size 126 \\
        --encoding gaf_mtf \\
        --dataset sp500 \\
        --augmentations random_resized_crop horizontal_flip color_jitter \\
                         random_grayscale gaussian_blur
"""

import argparse
import lightning as pl
import torch
import torchmetrics
from torch import nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import stable_pretraining as spt
from stable_pretraining.callbacks import RankMe
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput
from datasets import load_from_disk

import stable_pretraining.data.transforms as spt_transforms
import stable_pretraining.data.transforms_custom as transforms_custom


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LeJEPA for financial imagery (ViT-Tiny)"
    )
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--encoding",   type=str, required=True)
    parser.add_argument("--dataset",    type=str, required=True, choices=["sp500", "ff"])
    parser.add_argument(
        "--augmentations", type=str, nargs="+", required=True,
        choices=[
            "random_resized_crop", "horizontal_flip", "color_jitter",
            "random_grayscale",    "gaussian_blur",   "magnitude_scaling",
            "gaussian_noise",      "temporal_masking",
        ],
        help="Augmentations to include in each view (applied in canonical order).",
    )
    return parser.parse_args()


args = parse_args()

SEED        = 42
BATCH_SIZE  = 256
NUM_WORKERS = 4

# ── Reproducibility ────────────────────────────────────────────────────────────
pl.seed_everything(SEED, workers=True)


# ── Augmentation pipeline ──────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Canonical application order: spatial → photometric → domain-specific.
# Both view builders respect this order regardless of how --augmentations are listed.
_AUGMENTATION_ORDER = [
    "random_resized_crop",
    "horizontal_flip",
    "color_jitter",
    "random_grayscale",
    "gaussian_blur",
    "magnitude_scaling",
    "gaussian_noise",
    "temporal_masking",
]

N_GLOBAL = 2
N_LOCAL  = 6


def _append_photometric_and_domain(steps: list, aug_set: set) -> None:
    """Append photometric and domain-specific transforms to steps in-place."""
    for aug in _AUGMENTATION_ORDER[1:]:   # random_resized_crop handled by caller
        if aug not in aug_set:
            continue
        if aug == "horizontal_flip":
            steps.append(spt_transforms.RandomHorizontalFlip(p=0.5))
        elif aug == "color_jitter":
            steps.append(spt_transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ))
        elif aug == "random_grayscale":
            steps.append(spt_transforms.RandomGrayscale(p=0.2))
        elif aug == "gaussian_blur":
            steps.append(spt_transforms.PILGaussianBlur(sigma=[0.1, 2.0], p=1.0))
        elif aug == "magnitude_scaling":
            steps.append(spt_transforms.WrapTorchTransform(
                transforms_custom.MagnitudeScaling(scale_range=(0.8, 1.2), p=0.5)
            ))
        elif aug == "gaussian_noise":
            steps.append(spt_transforms.WrapTorchTransform(
                transforms_custom.GaussianNoiseInjection(sigma=0.05, p=0.5)
            ))
        elif aug == "temporal_masking":
            steps.append(spt_transforms.WrapTorchTransform(
                transforms_custom.RandomTemporalMasking(
                    mask_ratio_range=(0.10, 0.20), p=0.5
                )
            ))
        else:
            raise ValueError(f"Unknown transform: {aug}")


def _make_global_view(transform_names) -> spt_transforms.Compose:
    """224×224 global view. random_resized_crop is respected as an ablation knob.

    If absent, a plain Resize(224) is used so the ablation is clean.
    """
    aug_set = {transform_names} if isinstance(transform_names, str) else set(transform_names)
    steps: list = [spt_transforms.RGB()]
    if "random_resized_crop" in aug_set:
        steps.append(spt_transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)))
    else:
        steps.append(spt_transforms.Resize((224, 224)))
    _append_photometric_and_domain(steps, aug_set)
    steps.append(spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return spt_transforms.Compose(*steps)


def _make_local_view(transform_names) -> spt_transforms.Compose:
    """96×96 local view. Always uses RandomResizedCrop — this is architectural.

    The small-scale crop is what makes local views semantically different from
    global views; it is not an ablatable augmentation choice.
    """
    aug_set = {transform_names} if isinstance(transform_names, str) else set(transform_names)
    steps: list = [
        spt_transforms.RGB(),
        spt_transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
    ]
    _append_photometric_and_domain(steps, aug_set)
    steps.append(spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return spt_transforms.Compose(*steps)


train_transform = spt_transforms.MultiViewTransform({
    **{f"global_{i}": _make_global_view(args.augmentations) for i in range(N_GLOBAL)},
    **{f"local_{i}":  _make_local_view(args.augmentations)  for i in range(N_LOCAL)},
})

val_transform = spt_transforms.Compose(
    spt_transforms.RGB(),
    spt_transforms.ToImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
)


# ── Dataset ───────────────────────────────────────────────────────────────────
class FinancialImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset   = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for ff_col in ["label_mktrf", "label_smb", "label_hml", "label_rmw", "label_cma"]:
            if ff_col in sample:
                sample[ff_col] = sample[ff_col] - 1   # 1-indexed → 0-indexed
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)


full_dataset = load_from_disk(args.data_dir)

train_ds = FinancialImageDataset(full_dataset["train"], transform=train_transform)
val_ds   = FinancialImageDataset(full_dataset["val"],   transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

data = spt.data.DataModule(train=train_loader, val=val_loader)


# ── LeJEPA model ──────────────────────────────────────────────────────────────
lejepa_model = LeJEPA(
    encoder_name="vit_tiny_patch16_224",
    lamb=0.02,
    n_slices=1024,
    n_points=17,
)

embed_dim = lejepa_model.embed_dim   # 192 for ViT-Tiny


# ── Forward ───────────────────────────────────────────────────────────────────
# lejepa_forward is a plain function passed to spt.Module(forward=...), not
# monkey-patched, because LeJEPA bundles its own encoder/projector/sigreg.
def lejepa_forward(self, batch, stage):
    out    = {}
    images = batch.get("image")   # None during training (named-view batch)

    if stage == "fit":
        global_views = [batch[k]["image"] for k in batch if k.startswith("global")]
        local_views  = [batch[k]["image"] for k in batch if k.startswith("local")]
        first_view   = next(v for k, v in batch.items() if k.startswith("global"))

        output: LeJEPAOutput = self.model.forward(
            global_views=global_views, local_views=local_views, images=images
        )

        if "label" in first_view:
            out["label"] = first_view["label"].repeat(len(global_views))
        for ff_col in ["label_mktrf", "label_smb", "label_hml", "label_rmw", "label_cma"]:
            if ff_col in first_view:
                out[ff_col] = first_view[ff_col].repeat(len(global_views))
    else:
        output: LeJEPAOutput = self.model.forward(images=images)
        if "label" in batch:
            out["label"] = batch["label"].long()
        for ff_col in ["label_mktrf", "label_smb", "label_hml", "label_rmw", "label_cma"]:
            if ff_col in batch:
                out[ff_col] = batch[ff_col].long()

    out["loss"]      = output.loss
    out["embedding"] = output.embedding

    self.log(f"{stage}/sigreg", output.sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/inv",    output.inv_loss,    on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss",   output.loss,        on_step=True, on_epoch=True, sync_dist=True)
    return out


# ── Module ────────────────────────────────────────────────────────────────────
module = spt.Module(
    model=lejepa_model,
    forward=lejepa_forward,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 1.5e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)


# ── Probes ────────────────────────────────────────────────────────────────────
probes = []
if args.dataset == "ff":
    for factor_name, col in [
        ("mktrf", "label_mktrf"),
        ("smb",   "label_smb"),
        ("hml",   "label_hml"),
        ("rmw",   "label_rmw"),
        ("cma",   "label_cma"),
    ]:
        probes.append(spt.callbacks.OnlineProbe(
            module,
            name=f"linear_probe_{factor_name}",
            input="embedding",
            target=col,
            probe=nn.Linear(embed_dim, args.num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
                    args.num_classes, average="macro"
                ),
            },
        ))
else:
    probes.append(spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(embed_dim, args.num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
                args.num_classes, average="macro"
            ),
        },
    ))

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label_mktrf" if args.dataset == "ff" else "label",
    queue_length=10000,
    metrics={
        "balanced_acc": torchmetrics.classification.MulticlassAccuracy(
            args.num_classes, average="macro"
        ),
    },
    input_dim=embed_dim,
    k=20,
)


# ── AverageFactorAccuracy (FF only) ───────────────────────────────────────────
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
            def to_float(v):
                if v is None:          return float("nan")
                if hasattr(v, "item"): return float(v.item())
                return float(v)
            vals = [to_float(trainer.callback_metrics.get(k)) for k in keys]
            avg  = float("nan") if any(v != v for v in vals) else sum(vals) / len(vals)
            pl_module.log(
                "eval/linear_probe_ff_avg_balanced_acc",
                avg, on_epoch=True, prog_bar=False, sync_dist=True,
            )
    avg_factor_acc_cb = AverageFactorAccuracy()
else:
    avg_factor_acc_cb = None


# ── RankMe ────────────────────────────────────────────────────────────────────
rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=embed_dim,
)


# ── Logger ────────────────────────────────────────────────────────────────────
run_name = (
    f"lejepa_{args.encoding}_{args.dataset}"
    f"_w{args.window_size:03d}_seed{SEED}"
)
wandb_logger = WandbLogger(
    project="thesis-ablations",
    name=run_name,
    log_model=False,
    config={**vars(args), "seed": SEED, "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS},
)


# ── Trainer ───────────────────────────────────────────────────────────────────
lr_monitor        = LearningRateMonitor(logging_interval="step")
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


# ── Run ───────────────────────────────────────────────────────────────────────
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
