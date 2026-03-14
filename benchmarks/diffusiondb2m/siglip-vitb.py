"""SigLIP (Sigmoid Loss for Language-Image Pre-Training) on DiffusionDB-2M.

Trains a ViT-B/32 image-text model using the SigLIP sigmoid loss instead of the
CLIP softmax InfoNCE loss.  The key difference: every (image, text) pair in the
NxN batch similarity matrix is treated as an independent binary classification
(y=+1 on the diagonal, y=-1 everywhere else), with learnable logit_scale and
logit_bias, removing the need for row/column softmax normalisation.

Structure mirrors benchmarks/diffusiondb2m/clip-vitb.py — swap CLIPLoss for
SigLIPLoss and adapt the monitor metrics accordingly.

Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
(arXiv:2303.15343).
"""

import argparse

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

import stable_pretraining as spt
from functools import partial

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_devices", type=int, default=8)
parser.add_argument("--global_batch", type=int, default=4096)
parser.add_argument("--num_epochs", type=int, default=8)
parser.add_argument("--val_percent", type=float, default=0.10)
parser.add_argument("--resume_ckpt_path", type=str, default=None)
args = parser.parse_args()

lr = args.lr
num_devices = args.num_devices
global_batch = args.global_batch
batch_size = global_batch // num_devices
num_epochs = args.num_epochs
val_percent = args.val_percent
resume_ckpt_path = args.resume_ckpt_path

# ---------------------------------------------------------------------------
# Encoder backbones (same as CLIP benchmark)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)
text_model = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)


def tokenize(text: str, tokenizer: AutoTokenizer):
    data = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    return data["input_ids"].squeeze(0), data["attention_mask"].squeeze(0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
image_transform = spt.data.transforms.Compose(
    spt.data.transforms.Resize((224, 224)),
    spt.data.transforms.ToImage(
        mean=[0.481, 0.457, 0.408],
        std=[0.268, 0.261, 0.275],
    ),
    spt.data.transforms.LambdaTransform(
        fn=partial(tokenize, tokenizer=tokenizer),
        source="prompt",
        targets=("tokenized_prompt", "attention_mask"),
    ),
)

train_base = spt.data.HFDataset(
    "poloclub/diffusiondb",
    "2m_all",
    split="train",
    transform=image_transform,
    remove_columns=["timestamp", "user_name", "prompt_nsfw", "image_nsfw", "sampler"],
)

size = len(train_base)
val_n = int(size * val_percent)
val_dataset = spt.data.Subset(train_base, range(0, val_n))
train_dataset = spt.data.Subset(train_base, range(val_n, size))

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def forward(self: spt.Module, batch: dict, stage: str) -> dict:
    out = {}
    vision_outputs = self.vision_model(pixel_values=batch["image"])
    image_embeds = F.normalize(vision_outputs.image_embeds, dim=-1)

    text_outputs = self.text_model(
        input_ids=batch["tokenized_prompt"],
        attention_mask=batch["attention_mask"],
    )
    text_embeds = F.normalize(text_outputs.text_embeds, dim=-1)

    out["image_embeds"] = image_embeds
    out["text_embeds"] = text_embeds

    if self.training:
        # SigLIPLoss normalises internally, but pre-normalised inputs are fine
        out["loss"] = self.siglip_loss(image_embeds, text_embeds)
    return out


# ---------------------------------------------------------------------------
# Monitor callback — sigmoid-aware metrics
# ---------------------------------------------------------------------------
class SigLIPMonitor(pl.Callback):
    """Logs SigLIP-specific training metrics.

    Computes retrieval (R@1), sigmoid-loss statistics (positive probability,
    margin, effective temperature) and embedding alignment from image/text
    embeddings, and logs them during training and validation.
    """

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.every = log_every_n_steps

    @torch.no_grad()
    def _log(self, trainer: pl.Trainer, pl_module, outputs: dict, stage: str):
        img = outputs["image_embeds"]  # already L2-normalised
        txt = outputs["text_embeds"]

        # Use the learnable scale and bias from the loss module
        scale = pl_module.siglip_loss.logit_scale.exp().item()
        bias = pl_module.siglip_loss.logit_bias.item()

        logits = scale * (img @ txt.T) + bias  # [B, B]
        B = logits.size(0)
        diag = torch.arange(B, device=logits.device)

        # Sigmoid probabilities (positive pair probability)
        pos_logits = logits[diag, diag]
        pos_prob = torch.sigmoid(pos_logits).mean()

        # Hardest negative logit per row (image→text direction)
        neg_logits = logits.masked_fill(
            torch.eye(B, dtype=torch.bool, device=logits.device), float("-inf")
        )
        top_neg = neg_logits.max(dim=1).values
        margin = pos_logits - top_neg

        # R@1 via argmax on raw dot-product similarity (no bias for ranking)
        sim = img @ txt.T
        r1_i2t = (sim.argmax(dim=1) == diag).float().mean()
        r1_t2i = (sim.argmax(dim=0) == diag).float().mean()

        cos_pos = F.cosine_similarity(img, txt, dim=-1).mean()

        trainer.logger.log_metrics(
            {
                f"{stage}/retrieval/R@1_i2t": float(r1_i2t.cpu()),
                f"{stage}/retrieval/R@1_t2i": float(r1_t2i.cpu()),
                f"{stage}/contrast/pos_prob": float(pos_prob.cpu()),
                f"{stage}/contrast/margin": float(margin.mean().cpu()),
                f"{stage}/align/cos_pos": float(cos_pos.cpu()),
                f"{stage}/siglip/logit_scale": scale,
                f"{stage}/siglip/logit_bias": bias,
            },
            step=trainer.global_step,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            self._log(trainer, pl_module, outputs, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log(trainer, pl_module, outputs, "val")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
module = spt.Module(
    vision_model=vision_model,
    text_model=text_model,
    forward=forward,
    siglip_loss=spt.losses.SigLIPLoss(
        init_logit_scale=2.302585,  # log(10)
        init_logit_bias=-10.0,
    ),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 1.0e-6,
            "betas": (0.9, 0.98),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": (len(train_dataloader) // num_devices) * num_epochs,
            "peak_step": 0.1,
        },
        "interval": "step",
    },
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
wandb_logger = WandbLogger(
    entity="stable-pretraining",
    project="diffusiondb2m-siglip",
    name="siglip-vit-b32-diffusiondb2m",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(
            monitor="train/loss_step",
            mode="min",
            every_n_epochs=1,
            save_top_k=-1,
            dirpath="/your/path/to/checkpoints",
        ),
        LearningRateMonitor(logging_interval="step"),
        SigLIPMonitor(log_every_n_steps=10),
    ],
    precision="bf16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    devices=num_devices,
    accelerator="gpu",
    strategy="ddp",
)

manager = spt.Manager(
    trainer=trainer,
    module=module,
    data=data,
    ckpt_path=resume_ckpt_path,
)

manager()
