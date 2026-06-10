"""Task heads + ordinal (CORAL/CORN) losses for the cross-encoder.

Three task families mirror ``tasks.REGISTRY``:

* **regression**     — a single linear unit, MSE loss, raw output.
* **classification** — ``n_classes`` logits, cross-entropy, argmax.
* **ordinal**        — a rank-monotonic head, either:
    - **CORAL** (Cao et al. 2020): one shared weight vector + ``K-1`` independent
      biases → ``K-1`` ordered binary tasks "is y > k?".
    - **CORN** (Shi et al. 2021): ``K-1`` independent logits trained with the
      *conditional* chain so the implied rank probabilities stay monotone without
      the weight-sharing constraint.

Ordinal scores are arbitrary (e.g. {0,1,2,3} per ASAP-SAS prompt); the trainer
maps the sorted unique training scores to contiguous ranks ``0..K-1`` and inverts
on prediction, so these losses only ever see contiguous integer ranks.

The loss/predict functions are deliberately free of any model state so they can be
unit-tested on hand-built logits without a transformer (see ``tests/test_neural``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- ordinal: CORAL ------------------------------

class CoralHead(nn.Module):
    """Shared weight + ``K-1`` independent biases → ``K-1`` ordered logits."""

    def __init__(self, hidden: int, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"ordinal head needs >=2 classes, got {num_classes}")
        self.num_classes = num_classes
        self.weight = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (N, K-1)
        return self.weight(x) + self.bias


def levels_from_ranks(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``y`` (N,) integer ranks → extended binary levels (N, K-1), 1 iff y > k."""
    k = torch.arange(num_classes - 1, device=y.device)
    return (y.unsqueeze(1) > k.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    levels = levels_from_ranks(y, num_classes)
    # numerically-stable BCE-with-logits, summed over tasks then mean over batch
    val = -torch.sum(
        F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels),
        dim=1,
    )
    return val.mean()


# --------------------------- ordinal: CORN -------------------------------

def corn_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Conditional ordinal loss (Shi et al. 2021).

    Task ``i`` predicts ``y > i`` but is trained only on the conditional subset
    ``y >= i`` — so the chained (cumulative-product) probabilities are monotone.
    """
    num_tasks = num_classes - 1
    total_loss = logits.new_tensor(0.0)
    n_examples = 0
    for i in range(num_tasks):
        mask = y >= i                     # conditional set for task i
        if mask.sum() == 0:
            continue
        pred = logits[mask, i]
        label = (y[mask] > i).float()
        total_loss = total_loss - torch.sum(
            F.logsigmoid(pred) * label + (F.logsigmoid(pred) - pred) * (1 - label)
        )
        n_examples += int(mask.sum())
    return total_loss / max(n_examples, 1)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """CORN rank: cumulative product of sigmoids, count of P(y>i) > 0.5."""
    probas = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (probas > 0.5).sum(dim=1)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """CORAL rank: count of independent P(y>k) > 0.5 (already monotone by design)."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


# --------------------------- dispatch helpers ----------------------------

def ordinal_loss(name: str, logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if name == "coral":
        return coral_loss(logits, y, num_classes)
    if name == "corn":
        return corn_loss(logits, y, num_classes)
    raise ValueError(f"unknown ordinal head {name!r} (expected 'coral'|'corn')")


def ordinal_predict(name: str, logits: torch.Tensor) -> torch.Tensor:
    return coral_predict(logits) if name == "coral" else corn_predict(logits)
