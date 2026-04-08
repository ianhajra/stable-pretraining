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
    s = torch.linalg.svdvals(embeddings) + 1e-8
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy)


def _rerankme_score(
    z1: torch.Tensor, z2: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Compute the ReRankMe score via spectral ratio entropy.

    Measures how much each intrinsic direction of Z1 changes under augmentation,
    after removing dependence on global scale and anisotropy via spectral ratio.

    Args:
        z1: Embedding matrix of shape (N, D) for the first augmented view.
        z2: Embedding matrix of shape (N, D) for the second augmented view.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar ReRankMe score in [0, 1]. Higher means more augmentation-invariant.
    """
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}"
        )

    z1 = z1 - z1.mean(dim=0)
    delta_z = z1 - z2
    delta_z = delta_z - delta_z.mean(dim=0)

    s1 = torch.linalg.svdvals(z1)
    s_delta = torch.linalg.svdvals(delta_z)

    # Align lengths (truncate to the shorter of the two)
    d = min(s1.shape[0], s_delta.shape[0])
    s1 = s1[:d]
    s_delta = s_delta[:d]

    # Spectral ratio: how much each direction changes relative to Z1
    r = s_delta / (s1 + epsilon)

    # Normalize into a probability distribution
    r_sum = r.sum()
    p = r / (r_sum + epsilon)

    # Entropy of the spectral ratio distribution
    entropy = -torch.sum(p * torch.log(p + epsilon))

    # Normalize by maximum entropy
    h_max = torch.log(torch.tensor(d, dtype=torch.float32, device=z1.device))
    rerankme = 1.0 - (entropy / (h_max + epsilon))

    return rerankme


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
    """ReRankMe monitor: augmentation-invariance score via spectral ratio entropy.

    Computes ReRankMe using spectral ratio entropy: measures how much each
    intrinsic direction of Z1 changes under augmentation, normalized by the
    magnitude of Z1 in that direction. A score near 1 indicates strong
    augmentation invariance; near 0 indicates high sensitivity.

    Args:
        name: Unique name for this callback instance
        target_view1: Key in batch dict for the first augmented view embeddings
        target_view2: Key in batch dict for the second augmented view embeddings
        queue_length: Required queue length (same for both views)
        target_shape: Shape of the embeddings (e.g., 768 for 768-dim features)
        lam: Weight of the augmentation-invariance penalty (default: 1.0, unused)
        epsilon: Small constant added to denominator for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        name: str,
        target_view1: str,
        target_view2: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        lam: float = 1.0,
        epsilon: float = 1e-8,
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
        self.epsilon = epsilon

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
                rerankme = _rerankme_score(z1, z2, self.epsilon)
                pl_module.log(self.name, rerankme.item())
