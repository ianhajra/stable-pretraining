import sys
import types
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE
from stable_pretraining.methods.adaptive_masking import AdaptiveMaskingEMA


class AdvanceEpochCallback(pl.Callback):
    """Ticks the AdaptiveMaskingEMA epoch counter at the end of each train epoch."""

    def __init__(self, masking_module: AdaptiveMaskingEMA) -> None:
        self.masking_module = masking_module

    def on_train_epoch_end(self, trainer, pl_module):
        self.masking_module.advance_epoch()


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 64
    max_epochs = 600

    # EMA decay of 0.9 gives an effective window of ~10 epochs per sample
    # (equal weighting approximation: 1 - 1/10 = 0.9)
    mask_ratio = 0.75

    def mae_forward(self, batch, stage):
        output = MAE.forward(self, batch["image"], indices=batch["sample_idx"])
        with torch.no_grad():
            features = self.encoder.forward_features(batch["image"])

        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )

        return {
            "loss": output.loss,
            "embedding": features[:, 1:].mean(dim=1).detach(),  # skip cls
            **({"label": batch["label"].long()} if "label" in batch else {}),
        }

    data_dir = str(get_data_dir("imagenet10"))

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 6),
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 6),
            persistent_workers=num_workers > 0,
        ),
    )

    adaptive_masking = AdaptiveMaskingEMA(
        m_base=mask_ratio,
        dataset_size=len(data.train.dataset),
        ema_decay=0.9,
        warmup_epochs=50,
    )

    module = MAE(
        model_or_model_name="vit_base_patch16_224",
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=mask_ratio,
        block_size=1,  # random masking
        norm_pix_loss=True,  # normalize pixel targets per patch
        loss_type="mse",
        pretrained=False,
        use_adaptive_masking=True,
    )

    # Replace the default AdaptiveMasking with AdaptiveMaskingEMA
    module.adaptive_masking = adaptive_masking

    module.forward = types.MethodType(mae_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": 5e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 40 / 600,
            "start_factor": 0.01,
            "end_lr": 5e-4 / 10,
            "total_steps": (len(data.train) // num_gpus) * 600,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            AdvanceEpochCallback(adaptive_masking),
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(768, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.025, "weight_decay": 0.0},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=768,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=768,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(
                    Path(__file__).parent / "checkpoints" / "mae-vitb-adaptive-ema"
                ),
                filename="mae-vitb-adaptive-ema-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=300,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="ianhajra-brown-university",
            project="imagenet10-methods",
            name="mae-vitb-adaptive-ema-inet10",
            log_model=False,
        ),
        precision="16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
