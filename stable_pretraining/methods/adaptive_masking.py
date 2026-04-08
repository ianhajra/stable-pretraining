"""Adaptive masking ratio adjustment based on per-sample reconstruction losses.

Provides two classes for truly per-sample adaptive masking.  Both track losses
keyed by **dataset sample index**, so the difficulty of one image never directly
affects another (beyond sharing the epoch-level mean/variance used for
normalisation).

The adjustment formula for sample *i* is::

    m_i = clip(m_base - (m_base * (1 - m_base) / Var(e)) * (e_i - mean(e)), 0.1, 0.9)

where ``e`` is the full set of per-sample losses committed at the end of the
*previous* epoch and ``mean`` / ``Var`` are computed across all dataset samples
seen so far.

:class:`AdaptiveMasking`
    Uses each sample's loss from the single most-recent completed epoch.

:class:`AdaptiveMaskingEMA`
    Uses an EMA over several epochs per sample.  ``ema_decay=0.9`` gives an
    effective window of roughly 10 epochs.

Both classes share the same interface:

* :meth:`update(losses, indices)` — call after every training step with the
  per-sample losses and their dataset indices for that batch.
* :meth:`get_ratios(indices)` — call before masking with the dataset indices
  for the upcoming batch; returns per-sample ratios.
* :meth:`advance_epoch()` — call once at the end of every training epoch (e.g.
  from a ``pl.Callback``) to commit the accumulated losses.

.. note::
    In multi-GPU (DDP) runs each process maintains its own loss store and only
    updates the samples it processes.  Samples not seen by a given process
    fall back to ``m_base``.  Full cross-process synchronisation is not
    implemented.
"""

from __future__ import annotations

import torch

__all__ = ["AdaptiveMasking", "AdaptiveMaskingEMA"]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _compute_ratios(
    m_base: float,
    e_i: torch.Tensor,
    all_losses: torch.Tensor,
) -> torch.Tensor:
    """Apply the adaptive formula using epoch-level statistics.

    Normalisation (mean / variance) is computed across *all* samples in
    ``all_losses`` that have been seen at least once (i.e. are not NaN).
    This keeps the average ratio across the batch close to ``m_base``.

    Samples whose stored loss is NaN (never seen) are assigned ``m_base``.

    :param m_base: Base masking ratio.
    :param e_i: Per-sample losses for the current batch, shape ``[B]``.
        May contain NaN for samples never seen before.
    :param all_losses: Full per-sample loss buffer for the dataset, shape
        ``[N]``, may contain NaN.
    :return: Ratios, shape ``[B]``, clipped to ``[0.1, 0.9]``.
    """
    valid = all_losses[~torch.isnan(all_losses)]
    if valid.numel() == 0:
        return torch.full((e_i.shape[0],), m_base)

    e_mean = valid.mean()
    e_var = valid.var(unbiased=False)

    if e_var < 1e-8:
        return torch.full((e_i.shape[0],), m_base, device=all_losses.device)

    scale = m_base * (1.0 - m_base) / e_var
    ratios = m_base - scale * (e_i - e_mean)
    # Unseen samples fall back to m_base
    ratios = torch.where(torch.isnan(e_i), torch.full_like(ratios, m_base), ratios)
    return ratios.clamp(0.1, 0.9)


class AdaptiveMasking:
    """Per-sample adaptive masking using the previous epoch's loss per image.

    On epoch *k*, each sample is masked using its own loss recorded in epoch
    *k-1*.  Normalisation uses the mean and variance across all samples seen
    so far, keeping the average ratio close to ``m_base``.

    Usage pattern::

        # In a Lightning Callback:
        def on_train_epoch_end(self, trainer, pl_module):
            pl_module.adaptive_masking.advance_epoch()


        # In the model forward (training step):
        ratios = self.adaptive_masking.get_ratios(batch["sample_idx"])
        # ... compute loss ...
        self.adaptive_masking.update(per_sample_losses, batch["sample_idx"])

    :param m_base: Base masking ratio, in the open interval ``(0, 1)``.
    :param dataset_size: Total number of samples in the training dataset.
    :param warmup_epochs: Use uniform ``m_base`` for this many epochs before
        enabling adaptation (minimum 1, since epoch 0 has no loss history yet).
    """

    def __init__(
        self,
        m_base: float,
        dataset_size: int,
        warmup_epochs: int = 1,
    ) -> None:
        if not (0.0 < m_base < 1.0):
            raise ValueError(f"m_base must be in (0, 1), got {m_base}")
        self.m_base = m_base
        self.dataset_size = dataset_size
        self.warmup_epochs = max(warmup_epochs, 1)
        self._current_epoch: int = 0
        # Previous epoch's per-sample losses; NaN = sample not yet seen
        self._sample_losses: torch.Tensor = torch.full((dataset_size,), float("nan"))
        # Accumulates losses seen during the current in-progress epoch
        self._accumulator: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, losses: torch.Tensor, indices: torch.Tensor) -> None:
        """Record per-sample losses from the current training step.

        Call this after the forward/backward pass.

        :param losses: Per-sample scalar losses, shape ``[B]``.
        :param indices: Dataset indices for each sample in the batch, shape ``[B]``.
        """
        if losses.dim() != 1 or indices.dim() != 1:
            raise ValueError("losses and indices must be 1-D tensors")
        if losses.shape != indices.shape:
            raise ValueError("losses and indices must have the same length")
        for loss_val, idx in zip(losses.detach().float().tolist(), indices.tolist()):
            self._accumulator[int(idx)] = float(loss_val)

    def advance_epoch(self) -> None:
        """Commit this epoch's accumulated losses and advance the epoch counter.

        Call once at the end of every training epoch.
        """
        for idx, loss_val in self._accumulator.items():
            self._sample_losses[idx] = loss_val
        self._accumulator.clear()
        self._current_epoch += 1

    def get_ratios(self, indices: torch.Tensor) -> torch.Tensor:
        """Return per-sample masking ratios for the upcoming training step.

        Returns uniform ``m_base`` during warmup or for samples not yet seen.

        :param indices: Dataset indices for the upcoming batch, shape ``[B]``.
        :return: Masking ratios, shape ``[B]``, values in ``[0.1, 0.9]``.
        """
        if self._current_epoch < self.warmup_epochs:
            return torch.full((indices.shape[0],), self.m_base)
        e_i = self._sample_losses[indices.cpu().long()]
        return _compute_ratios(self.m_base, e_i, self._sample_losses)


class AdaptiveMaskingEMA:
    """Per-sample adaptive masking with EMA-smoothed loss history per image.

    Like :class:`AdaptiveMasking`, but the stored loss for each sample is an
    exponential moving average over the last several epochs rather than just
    the most-recent one.  ``ema_decay=0.9`` gives an effective window of
    roughly 10 epochs::

        stored[i] = ema_decay * stored[i] + (1 - ema_decay) * new_loss[i]

    The first time a sample is seen its stored loss is initialised directly
    (no blending with NaN).

    :param m_base: Base masking ratio, in the open interval ``(0, 1)``.
    :param dataset_size: Total number of samples in the training dataset.
    :param ema_decay: Per-epoch EMA decay factor, in ``[0, 1)``.
        ``0.9`` ≈ 10-epoch effective window.
    :param warmup_epochs: Use uniform ``m_base`` for this many epochs before
        enabling adaptation.  Default ``50``.
    """

    def __init__(
        self,
        m_base: float,
        dataset_size: int,
        ema_decay: float = 0.9,
        warmup_epochs: int = 50,
    ) -> None:
        if not (0.0 < m_base < 1.0):
            raise ValueError(f"m_base must be in (0, 1), got {m_base}")
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.m_base = m_base
        self.dataset_size = dataset_size
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs
        self._current_epoch: int = 0
        # EMA-smoothed per-sample losses; NaN = sample not yet seen
        self._sample_losses: torch.Tensor = torch.full((dataset_size,), float("nan"))
        # Accumulates losses seen during the current in-progress epoch
        self._accumulator: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, losses: torch.Tensor, indices: torch.Tensor) -> None:
        """Record per-sample losses from the current training step.

        :param losses: Per-sample scalar losses, shape ``[B]``.
        :param indices: Dataset indices for each sample in the batch, shape ``[B]``.
        """
        if losses.dim() != 1 or indices.dim() != 1:
            raise ValueError("losses and indices must be 1-D tensors")
        if losses.shape != indices.shape:
            raise ValueError("losses and indices must have the same length")
        for loss_val, idx in zip(losses.detach().float().tolist(), indices.tolist()):
            self._accumulator[int(idx)] = float(loss_val)

    def advance_epoch(self) -> None:
        """EMA-blend this epoch's losses into the per-sample store.

        For each sample seen this epoch::

            stored[i] = ema_decay * stored[i] + (1 - ema_decay) * epoch_loss[i]

        Samples seen for the first time are initialised directly.
        Call once at the end of every training epoch.
        """
        for idx, loss_val in self._accumulator.items():
            current = self._sample_losses[idx]
            if torch.isnan(current):
                self._sample_losses[idx] = loss_val
            else:
                self._sample_losses[idx] = (
                    self.ema_decay * current.item() + (1.0 - self.ema_decay) * loss_val
                )
        self._accumulator.clear()
        self._current_epoch += 1

    def get_ratios(self, indices: torch.Tensor) -> torch.Tensor:
        """Return per-sample masking ratios for the upcoming training step.

        Returns uniform ``m_base`` during warmup or for samples not yet seen.

        :param indices: Dataset indices for the upcoming batch, shape ``[B]``.
        :return: Masking ratios, shape ``[B]``, values in ``[0.1, 0.9]``.
        """
        if self._current_epoch < self.warmup_epochs:
            return torch.full((indices.shape[0],), self.m_base)
        e_i = self._sample_losses[indices.cpu().long()]
        return _compute_ratios(self.m_base, e_i, self._sample_losses)
