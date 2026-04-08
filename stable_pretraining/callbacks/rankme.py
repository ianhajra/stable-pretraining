from typing import Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .queue import find_or_create_queue_callback


def _rankme_score(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute the RankMe score (effective rank via singular value entropy).

    Computes singular values of the embedding matrix, normalizes them to a
    probability distribution, and returns exp(entropy).

    Args:
        embeddings: Matrix of shape (N, D).

    Returns:
        Scalar RankMe score.
    """
    embeddings = embeddings - embeddings.mean(dim=0)
    s = torch.linalg.svdvals(embeddings)
    p = (s / torch.sum(s, axis=0)) + 1e-5
    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy)


class RankMe(Callback):
    """RankMe (effective rank) monitor using queue discovery.

    RankMe measures the effective rank of feature representations by computing
    the exponential of the entropy of normalized singular values. This metric
    helps detect dimensional collapse in self-supervised learning.

    Args:
        name: Unique name for this callback instance
        target: Key in batch dict containing the feature embeddings to monitor
        queue_length: Required queue length
        target_shape: Shape of the target embeddings (e.g., 768 for 768-dim features)
    """

    def __init__(
        self,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
    ) -> None:
        super().__init__()

        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape

        self._target_queue = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"RankMe[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create the queue callback for target features."""
        if self._target_queue is None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for target '{self.target}'")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute RankMe metric on the first validation batch only."""
        if batch_idx > 0:
            return

        logging.info(f"{self.name}: Computing RankMe on first validation batch")

        embeddings = self._target_queue.data

        if embeddings is None:
            logging.warning(
                f"{self.name}: Queue data not available (not in validation?)"
            )
            return

        if embeddings.numel() == 0:
            logging.warning(
                f"{self.name}: Queue data is empty, skipping RankMe computation"
            )
            return

        if trainer.global_rank == 0:
            with torch.no_grad():
                rankme = _rankme_score(embeddings)
                pl_module.log(self.name, rankme.item())


class ReRankMe(Callback):
    """ReRankMe monitor: RankMe corrected for augmentation invariance.

    Computes RankMe(Z_1) - λ · RankMe(ΔZ), where ΔZ = Z_1 - Z_2 are the
    embedding matrices of two independently augmented views of the same N images.
    A good representation has high RankMe (rich structure) but low RankMe(ΔZ)
    (invariant to augmentation noise), yielding a high ReRankMe score.

    All computation — SVD, normalization, entropy — is identical to RankMe.
    The only change is the input matrix for the correction term.

    Args:
        name: Unique name for this callback instance
        target_view1: Key in batch dict for the first augmented view embeddings
        target_view2: Key in batch dict for the second augmented view embeddings
        queue_length: Required queue length (same for both views)
        target_shape: Shape of the embeddings (e.g., 768 for 768-dim features)
        lam: Weight of the augmentation-invariance penalty (default: 1.0)
    """

    def __init__(
        self,
        name: str,
        target_view1: str,
        target_view2: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        lam: float = 1.0,
    ) -> None:
        super().__init__()

        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target_view1 = target_view1
        self.target_view2 = target_view2
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.lam = lam

        self._queue_view1 = None
        self._queue_view2 = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"ReRankMe[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create queue callbacks for both views."""
        if self._queue_view1 is None:
            self._queue_view1 = find_or_create_queue_callback(
                trainer,
                self.target_view1,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for view1 '{self.target_view1}'")

        if self._queue_view2 is None:
            self._queue_view2 = find_or_create_queue_callback(
                trainer,
                self.target_view2,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for view2 '{self.target_view2}'")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute ReRankMe metric on the first validation batch only."""
        if batch_idx > 0:
            return

        logging.info(f"{self.name}: Computing ReRankMe on first validation batch")

        z1 = self._queue_view1.data
        z2 = self._queue_view2.data

        if z1 is None or z2 is None:
            logging.warning(
                f"{self.name}: Queue data not available (not in validation?)"
            )
            return

        if z1.numel() == 0 or z2.numel() == 0:
            logging.warning(
                f"{self.name}: Queue data is empty, skipping ReRankMe computation"
            )
            return

        if z1.shape != z2.shape:
            logging.warning(
                f"{self.name}: View queue shapes mismatch "
                f"({z1.shape} vs {z2.shape}), skipping ReRankMe computation"
            )
            return

        if trainer.global_rank == 0:
            with torch.no_grad():
                delta_z = z1 - z2
                rankme_z1 = _rankme_score(z1)
                rankme_delta = _rankme_score(delta_z)
                rerankme = rankme_z1 - self.lam * rankme_delta
                pl_module.log(self.name, rerankme.item())
                pl_module.log(f"{self.name}/rankme_z1", rankme_z1.item())
                pl_module.log(f"{self.name}/rankme_delta", rankme_delta.item())
