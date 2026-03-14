"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP, SigLIP, and SigLIP2.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import all_gather
from .joint_embedding import InfoNCELoss


class CLIPLoss(InfoNCELoss):
    """CLIP loss (symmetric bidirectional InfoNCE).

    As used in CLIP :cite:`radford2021learning`.
    Computes symmetric cross-entropy over image-text and text-image logits.

    Args:
        temperature (float, optional): Softmax temperature. Default is 0.07.
            (If you use a learnable logit_scale in your model, pass it to
            forward(...) and this temperature will be ignored.)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__(temperature=temperature)

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        # for CLIP, targets are always the diagonal
        targets = torch.arange(feats_i.size(0), device=feats_i.device)

        # calculate loss in both directions
        loss_i = self._compute(
            anchors=feats_i,
            candidates=feats_j,
            targets=targets,
            logit_scale=logit_scale,
        )
        loss_j = self._compute(
            anchors=feats_j,
            candidates=feats_i,
            targets=targets,
            logit_scale=logit_scale,
        )

        return 0.5 * (loss_i + loss_j)


class SigLIPLoss(nn.Module):
    """Sigmoid loss for language-image pre-training (SigLIP).

    Replaces the softmax-based contrastive loss in CLIP with element-wise sigmoid
    binary cross-entropy over the full NxN pairwise similarity matrix.  Every pair
    is independently treated as a binary classification: *positive* (same-image, i.e.
    the diagonal) or *negative* (different images, off-diagonal).

    This removes the need for row/column normalisation that is inherent to softmax
    and scales better to very large batch sizes.

    The logit for each pair (i, j) is computed as::

        l_ij = exp(logit_scale) * (z_i · z_j) + logit_bias

    where ``z_i``, ``z_j`` are L2-normalised embeddings, ``logit_scale`` is a
    learnable scalar (initialised to ``log(10) ≈ 2.303``) and ``logit_bias`` is
    a learnable scalar bias (initialised to ``-10``).

    The loss over an N×N batch is::

        L = −(1/N) Σ_i Σ_j log σ(y_ij · l_ij)

    where ``y_ij = +1`` for the positive (diagonal) and ``−1`` for all negatives.

    Args:
        init_logit_scale (float): Initial value of the log-temperature.
            Default: ``math.log(10)`` (≈ 2.303).
        init_logit_bias (float): Initial value of the bias. Default: ``-10.0``.
        logit_scale_max (float): Upper clamp for ``logit_scale`` (prevents
            degenerate collapse). Default: ``math.log(100)``.

    References:
        Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
        (arXiv:2303.15343).
    """

    def __init__(
        self,
        init_logit_scale: float = math.log(10),
        init_logit_bias: float = -10.0,
        logit_scale_max: float = math.log(100),
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        self.logit_scale_max = logit_scale_max

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SigLIP loss.

        Args:
            feats_i (torch.Tensor): Embeddings of the first view, shape ``[N, D]``.
            feats_j (torch.Tensor): Embeddings of the second view, shape ``[N, D]``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        feats_i = torch.cat(all_gather(F.normalize(feats_i, dim=-1)), 0)
        feats_j = torch.cat(all_gather(F.normalize(feats_j, dim=-1)), 0)

        logit_scale = self.logit_scale.clamp(max=self.logit_scale_max).exp()
        logits = logit_scale * (feats_i @ feats_j.T) + self.logit_bias

        n = feats_i.size(0)
        # Labels: +1 on the diagonal (positive pairs), −1 everywhere else
        labels = 2 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1

        # log σ(y · l) = logsigmoid(y · l); sum over all N² pairs, normalise by N
        loss = -F.logsigmoid(labels * logits).sum() / n
        return loss


class SigLIP2Loss(nn.Module):
    """Multi-positive sigmoid loss for SigLIP 2 (vision-only adaptation).

    Extends :class:`SigLIPLoss` to support **multiple positives per anchor**.
    When several augmented crops of the *same* source image appear in the batch,
    each pair of same-source embeddings is treated as a positive.

    This is the key algorithmic contribution of SigLIP 2 that enables
    self-distillation and multi-view training::

        l_ij = exp(logit_scale) * (z_i · z_j) + logit_bias

        L = −(1/N) Σ_i Σ_j log σ(y_ij · l_ij)

    where ``y_ij = +1`` if sample ``i`` and sample ``j`` share the same source
    image, and ``−1`` otherwise.

    Args:
        init_logit_scale (float): Initial log-temperature. Default: ``math.log(10)``.
        init_logit_bias (float): Initial bias. Default: ``-10.0``.
        logit_scale_max (float): Upper clamp for ``logit_scale``. Default: ``math.log(100)``.

    References:
        Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with
        Improved Semantic Understanding, Localisation, and Dense Features"
        (arXiv:2502.14786).
    """

    def __init__(
        self,
        init_logit_scale: float = math.log(10),
        init_logit_bias: float = -10.0,
        logit_scale_max: float = math.log(100),
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        self.logit_scale_max = logit_scale_max

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
        source_ids_i: Optional[torch.Tensor] = None,
        source_ids_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the SigLIP 2 loss.

        Args:
            feats_i (torch.Tensor): Embeddings of the first set, shape ``[N, D]``.
            feats_j (torch.Tensor): Embeddings of the second set, shape ``[M, D]``.
            source_ids_i (torch.Tensor, optional): Integer source-image index for
                each sample in ``feats_i``, shape ``[N]``.  When ``None``, falls back
                to identity-matrix positives (standard SigLIP behaviour).
            source_ids_j (torch.Tensor, optional): Integer source-image index for
                each sample in ``feats_j``, shape ``[M]``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        feats_i = torch.cat(all_gather(F.normalize(feats_i, dim=-1)), 0)
        feats_j = torch.cat(all_gather(F.normalize(feats_j, dim=-1)), 0)

        logit_scale = self.logit_scale.clamp(max=self.logit_scale_max).exp()
        logits = logit_scale * (feats_i @ feats_j.T) + self.logit_bias

        ni, nj = feats_i.size(0), feats_j.size(0)

        if source_ids_i is not None and source_ids_j is not None:
            source_ids_i = torch.cat(all_gather(source_ids_i), 0)
            source_ids_j = torch.cat(all_gather(source_ids_j), 0)
            # pos_mask[i,j] = 1 iff sample i and sample j come from the same image
            pos_mask = (source_ids_i.unsqueeze(1) == source_ids_j.unsqueeze(0)).float()
        else:
            pos_mask = torch.eye(ni, nj, device=logits.device, dtype=logits.dtype)

        # y_ij ∈ {−1, +1}
        labels = 2 * pos_mask - 1

        loss = -F.logsigmoid(labels * logits).sum() / ni
        return loss
