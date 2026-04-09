"""CIFAR-10-C corruption robustness callback.

Evaluates model robustness on all 19 CIFAR-10-C corruption types at all 5
severity levels using a frozen linear probe trained on clean CIFAR-10 features.
Logs a single ``eval/cifar10c_accuracy`` scalar.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from ..data.dataset_stats import CIFAR10 as CIFAR10_STATS

# All 19 CIFAR-10-C corruption types that correspond to the 19 splits of
# xinlinzz/cifar-10-c on Hugging Face.
_CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]

_N_SEVERITIES = 5
_N_SAMPLES_PER_SEVERITY = 10_000  # 50k total / 5 severities


class CIFAR10CCallback(Callback):
    """Evaluate corruption robustness on CIFAR-10-C.

    Downloads ``xinlinzz/cifar-10-c`` from Hugging Face at init time (cached
    after the first run) and evaluates all 19 corruption types at all 5
    severity levels.  A linear probe is fitted on clean CIFAR-10 training
    features using the current encoder (backbone) and then applied to the
    corruption test images.  Only the encoder output is used; the projector is
    never called.  A single scalar ``eval/cifar10c_accuracy`` is logged, equal
    to the mean accuracy across all 19 × 5 = 95 (corruption, severity) pairs.

    Args:
        every_n_epochs: Run the evaluation every this many epochs.  Default 1.
        encoder_attr: Name of the attribute on the LightningModule that holds
            the frozen encoder (backbone).  Default ``"backbone"``.
        batch_size: Batch size used when extracting features.  Default 256.
        probe_lr: Learning-rate for the linear probe SGD optimiser.
        probe_epochs: Number of full passes over the clean training features
            when fitting the probe.
        clean_cifar10_root: Root directory for caching the clean CIFAR-10
            dataset used to fit the probe.  Default ``"/tmp/cifar10_for_probe"``.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        encoder_attr: str = "backbone",
        batch_size: int = 256,
        probe_lr: float = 0.1,
        probe_epochs: int = 100,
        clean_cifar10_root: str = "/tmp/cifar10_for_probe",
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.encoder_attr = encoder_attr
        self.batch_size = batch_size
        self.probe_lr = probe_lr
        self.probe_epochs = probe_epochs

        # Shared preprocessing matching clean CIFAR-10 evaluation transforms.
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=CIFAR10_STATS["mean"],
                    std=CIFAR10_STATS["std"],
                ),
            ]
        )

        logging.info(
            "CIFAR10CCallback: loading xinlinzz/cifar-10-c from Hugging Face …"
        )
        from datasets import load_dataset  # noqa: PLC0415

        self._hf_splits: dict = {}
        for corruption in _CORRUPTIONS:
            self._hf_splits[corruption] = load_dataset(
                "xinlinzz/cifar-10-c",
                split=corruption,
            )

        logging.info("CIFAR10CCallback: loading clean CIFAR-10 train split for probe …")
        import torchvision.datasets as tv_datasets  # noqa: PLC0415

        self._clean_train = tv_datasets.CIFAR10(
            root=clean_cifar10_root,
            train=True,
            download=True,
        )

        logging.info("CIFAR10CCallback: initialised.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_encoder(self, pl_module: LightningModule) -> nn.Module:
        encoder = getattr(pl_module, self.encoder_attr)
        return encoder

    @torch.no_grad()
    def _extract_features(
        self,
        encoder: nn.Module,
        images_pil,
        labels,
        device: torch.device,
    ):
        """Return (features, labels) tensors for a list of PIL images."""
        all_feats = []
        all_labels = []
        n = len(images_pil)
        for start in range(0, n, self.batch_size):
            batch_imgs = images_pil[start : start + self.batch_size]
            batch_lbl = labels[start : start + self.batch_size]
            tensors = torch.stack([self._transform(img) for img in batch_imgs]).to(
                device
            )
            feats = encoder(tensors)
            # Handle tuple / dataclass outputs from some backbones.
            if not isinstance(feats, torch.Tensor):
                feats = (
                    feats[0]
                    if isinstance(feats, (tuple, list))
                    else feats.last_hidden_state
                )
            # Global-average-pool if the output has sequence / spatial dims.
            if feats.dim() == 3:
                feats = feats.mean(dim=1)
            elif feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            all_feats.append(feats.cpu())
            if isinstance(batch_lbl, torch.Tensor):
                all_labels.append(batch_lbl.cpu())
            else:
                all_labels.append(torch.tensor(batch_lbl, dtype=torch.long))
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    def _fit_linear_probe(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        feat_dim: int,
        n_classes: int,
        device: torch.device,
    ) -> nn.Linear:
        """Fit a linear probe with SGD on the pre-extracted clean features."""
        probe = nn.Linear(feat_dim, n_classes).to(device)
        X = X_train.to(device)
        y = y_train.to(device)
        optimiser = torch.optim.SGD(probe.parameters(), lr=self.probe_lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        n = X.shape[0]
        probe.train()
        for _ in range(self.probe_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimiser.zero_grad()
                loss_fn(probe(X[idx]), y[idx]).backward()
                optimiser.step()
        probe.eval()
        return probe

    @torch.no_grad()
    def _eval_probe(
        self,
        probe: nn.Linear,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> float:
        X = X.to(device)
        y = y.to(device)
        preds = probe(X).argmax(dim=1)
        return (preds == y).float().mean().item()

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.global_rank != 0:
            return

        device = pl_module.device
        encoder = self._get_encoder(pl_module)
        encoder.eval()

        logging.info(
            "CIFAR10CCallback: extracting clean CIFAR-10 train features for probe …"
        )
        clean_imgs = [self._clean_train[i][0] for i in range(len(self._clean_train))]
        clean_labels = [self._clean_train[i][1] for i in range(len(self._clean_train))]

        with torch.no_grad():
            X_train, y_train = self._extract_features(
                encoder, clean_imgs, clean_labels, device
            )

        feat_dim = X_train.shape[1]
        logging.info(
            f"CIFAR10CCallback: fitting linear probe (feat_dim={feat_dim}, "
            f"probe_epochs={self.probe_epochs}) …"
        )
        probe = self._fit_linear_probe(X_train, y_train, feat_dim, 10, device)

        logging.info(
            "CIFAR10CCallback: evaluating on 19 corruptions × 5 severity levels …"
        )
        total_acc = 0.0
        n_evals = 0

        for corruption in _CORRUPTIONS:
            hf_ds = self._hf_splits[corruption]
            all_imgs = hf_ds["image"]  # list of PIL Images, length 50 000
            all_labels = hf_ds["label"]  # list of ints, length 50 000

            for sev in range(_N_SEVERITIES):
                start_idx = sev * _N_SAMPLES_PER_SEVERITY
                end_idx = start_idx + _N_SAMPLES_PER_SEVERITY
                imgs_sev = all_imgs[start_idx:end_idx]
                lbls_sev = all_labels[start_idx:end_idx]

                with torch.no_grad():
                    X_corr, y_corr = self._extract_features(
                        encoder, imgs_sev, lbls_sev, device
                    )

                acc = self._eval_probe(probe, X_corr, y_corr, device)
                total_acc += acc
                n_evals += 1

        mean_acc = total_acc / n_evals
        logging.info(f"CIFAR10CCallback: eval/cifar10c_accuracy = {mean_acc:.4f}")
        pl_module.log(
            "eval/cifar10c_accuracy",
            mean_acc,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
