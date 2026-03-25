"""Adaptive masking ratio adjustment based on per-sample reconstruction losses.

Provides :class:`AdaptiveMasking`, a drop-in module that adjusts the masking
ratio per sample using the previous step's per-sample losses.  It is fully
self-contained: all state is a single stored loss tensor from the most recent
call to :meth:`update`.

The adjustment formula for sample *i* in a batch is::

    m_i = m_base - (m_base * (1 - m_base) / Var(e)) * (e_i - mean(e))

where ``e`` is the vector of per-sample losses stored from the previous step.
On the very first step, when no losses have been stored yet, :meth:`get_ratios`
returns ``m_base`` uniformly for all samples.

Output ratios are clipped to ``[0.1, 0.9]`` to prevent degenerate masking.
"""

from __future__ import annotations

import torch

__all__ = ["AdaptiveMasking"]


class AdaptiveMasking:
    """Per-sample adaptive masking ratio.

    Intended to be owned by a model that has a ``use_adaptive_masking`` flag.
    Typical usage (from a training script or forward hook)::

        # -- end of step t --
        per_sample_losses = ...  # torch.Tensor, shape [B]
        model.adaptive_masking.update(per_sample_losses)

        # -- start of step t+1, before masks are sampled --
        ratios = model.adaptive_masking.get_ratios(batch_size)
        # pass ratios to the mask sampler

    :param m_base: Base masking ratio that the model uses (e.g. ``mask_ratio``
        for MAE).  Must be in the open interval ``(0, 1)``.
    :param batch_size: Expected batch size.  Stored as metadata; the actual
        size used at runtime is taken from the argument to :meth:`get_ratios`.
    """

    def __init__(self, m_base: float, batch_size: int) -> None:
        if not (0.0 < m_base < 1.0):
            raise ValueError(f"m_base must be in (0, 1), got {m_base}")
        self.m_base = m_base
        self.batch_size = batch_size
        self._stored_losses: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, losses: torch.Tensor) -> None:
        """Store per-sample losses from the current training step.

        Call this *after* the forward pass, using a 1-D tensor of
        per-sample scalar losses (one value per image in the batch).

        :param losses: Per-sample losses, shape ``[B]``.
        """
        if losses.dim() != 1:
            raise ValueError(
                f"losses must be 1-D (one value per sample), "
                f"got shape {tuple(losses.shape)}"
            )
        self._stored_losses = losses.detach().float()

    def get_ratios(self, batch_size: int) -> torch.Tensor:
        """Return per-sample masking ratios for the upcoming training step.

        On the first call (no previous losses stored) or when the stored
        losses have negligible variance, returns a uniform tensor of
        ``m_base``.  Otherwise applies the adaptive formula and clips
        the result to ``[0.1, 0.9]``.

        :param batch_size: Number of samples in the upcoming batch.
        :return: Masking ratios, shape ``[batch_size]``, values in
            ``[0.1, 0.9]``.
        """
        if self._stored_losses is None:
            return torch.full((batch_size,), self.m_base)

        e = self._stored_losses
        e_mean = e.mean()
        e_var = e.var(unbiased=False)

        if e_var < 1e-8:
            return torch.full((batch_size,), self.m_base, device=e.device)

        scale = self.m_base * (1.0 - self.m_base) / e_var
        ratios = self.m_base - scale * (e - e_mean)
        return ratios.clamp(0.1, 0.9)
