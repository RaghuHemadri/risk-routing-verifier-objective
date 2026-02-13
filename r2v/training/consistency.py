"""
Tool-Consistency Regularization.

For contexts where the agent can re-query a tool, enforces invariance
across stochastic tool outputs. Given two tool outcomes (y, y') under
different seeds producing contexts (x, x'), encourage consistent
high-level action choice:

  L_cons(θ) = E_{(x,x')} KL(π_θ(·|x) || π_θ(·|x'))

This makes the SLM policy less brittle to stochastic tool outputs,
which is central to the "tool noise" axis.

Refinement: We use a symmetric KL (Jensen-Shannon divergence) for
stability, and only penalize divergence in the *action type* distribution
(not exact token sequences) for a more meaningful invariance signal.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def symmetric_kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute symmetric KL divergence (Jensen-Shannon) between two distributions.

    Uses temperature scaling for sharper/softer distributions.
    More stable than asymmetric KL for regularization.
    """
    log_probs_a = F.log_softmax(logits_a / temperature, dim=-1)
    log_probs_b = F.log_softmax(logits_b / temperature, dim=-1)
    probs_a = F.softmax(logits_a / temperature, dim=-1)
    probs_b = F.softmax(logits_b / temperature, dim=-1)

    kl_ab = F.kl_div(log_probs_b, probs_a, reduction="batchmean")
    kl_ba = F.kl_div(log_probs_a, probs_b, reduction="batchmean")

    return 0.5 * (kl_ab + kl_ba)


def consistency_loss(
    model: nn.Module,
    input_ids_a: torch.Tensor,
    attention_mask_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    attention_mask_b: torch.Tensor,
    temperature: float = 2.0,
    use_last_n_positions: int = 5,
) -> torch.Tensor:
    """Compute tool-consistency regularization loss.

    Gets the next-token distribution for the last few positions of both
    contexts and penalizes divergence. Using last-N positions captures
    the action decision space more broadly than just the final token.

    Args:
        model: Policy model
        input_ids_a: Context under tool seed z
        input_ids_b: Context under tool seed z'
        temperature: Softmax temperature (higher = softer, more stable KL)
        use_last_n_positions: Number of positions to compute KL over

    Returns:
        Scalar consistency loss
    """
    outputs_a = model(input_ids=input_ids_a, attention_mask=attention_mask_a)
    outputs_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b)

    logits_a = outputs_a.logits if hasattr(outputs_a, 'logits') else outputs_a["logits"]
    logits_b = outputs_b.logits if hasattr(outputs_b, 'logits') else outputs_b["logits"]

    # Use last N non-padding positions
    seq_len_a = attention_mask_a.sum(dim=1).min().item()
    seq_len_b = attention_mask_b.sum(dim=1).min().item()

    n = min(use_last_n_positions, int(seq_len_a), int(seq_len_b))
    if n < 1:
        return torch.tensor(0.0, device=logits_a.device, requires_grad=True)

    logits_a_tail = logits_a[:, -n:, :]
    logits_b_tail = logits_b[:, -n:, :]

    # Flatten positions and compute KL
    batch_size = logits_a_tail.size(0)
    logits_a_flat = logits_a_tail.reshape(batch_size * n, -1)
    logits_b_flat = logits_b_tail.reshape(batch_size * n, -1)

    return symmetric_kl_divergence(logits_a_flat, logits_b_flat, temperature)


class ConsistencyRegularizer:
    """Manages consistency regularization during training.

    Computes L_cons on batches from the ConsistencyDataset and
    provides the weighted loss term for the overall objective.
    """

    def __init__(self, config: dict[str, Any]):
        self.lambda_cons = config.get("lambda_cons", 0.1)
        self.temperature = config.get("temperature", 2.0)
        self.last_n_positions = config.get("last_n_positions", 5)
        self.enabled = config.get("enabled", True)

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted consistency loss for a batch.

        Args:
            model: Policy model
            batch: Dict with keys input_ids_a, attention_mask_a,
                   input_ids_b, attention_mask_b

        Returns:
            λ_cons * L_cons
        """
        if not self.enabled:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = consistency_loss(
            model,
            batch["input_ids_a"],
            batch["attention_mask_a"],
            batch["input_ids_b"],
            batch["attention_mask_b"],
            temperature=self.temperature,
            use_last_n_positions=self.last_n_positions,
        )

        return self.lambda_cons * loss
