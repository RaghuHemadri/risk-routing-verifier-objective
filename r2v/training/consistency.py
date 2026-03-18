"""
Consistency Regularizer for robustness against tool noise.

Implements the Jensen-Shannon Divergence (JSD) loss between policy
distributions on perturbed contexts (x, x') to enforce invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyRegularizer(nn.Module):
    """
    Computes Jensen-Shannon Divergence (JSD) between probability distributions
    output by the policy on two different views of the same state.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q) is the mixture distribution.

    This loss encourages the model to be invariant to stochastic tool outputs.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, 
        logits1: torch.Tensor, 
        logits2: torch.Tensor, 
        attention_mask1: torch.Tensor = None, 
        attention_mask2: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute symmetric KL (Jensen-Shannon) divergence between two batches of logits.
        
        Args:
            logits1: (batch, seq_len1, vocab)
            logits2: (batch, seq_len2, vocab)
            attention_mask1: (batch, seq_len1) 1 for valid tokens, 0 for pad
            attention_mask2: (batch, seq_len2) 1 for valid tokens, 0 for pad

        Returns:
            Scalar JSD loss averaged over valid tokens.
        """
        # 1. Align sequence lengths (truncate to min length of the generated portion)
        # Note: In practice, we usually compare the distribution over the *generated action*
        # so the sequences might be different lengths if generation is unconstrained.
        # But for consistency, we typically force-teach the same action on perturbed observations
        # or compare the *next token distribution* at each step.
        # Here we assume the logits correspond to the same target action tokens.
        
        min_len = min(logits1.size(1), logits2.size(1))
        logits1 = logits1[:, :min_len, :]
        logits2 = logits2[:, :min_len, :]
        
        if attention_mask1 is not None:
             attention_mask1 = attention_mask1[:, :min_len]
        if attention_mask2 is not None:
             attention_mask2 = attention_mask2[:, :min_len]

        # 2. Compute probabilities with temperature scaling
        # (batch, seq, vocab)
        p = F.softmax(logits1 / self.temperature, dim=-1)
        q = F.softmax(logits2 / self.temperature, dim=-1)

        # 3. Compute Mixture M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # 4. Compute KL(P || M) and KL(Q || M)
        # KL(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)} = \sum P(x) (\log P(x) - \log Q(x))
        # We use log_softmax for numerical stability of log terms
        log_p = F.log_softmax(logits1 / self.temperature, dim=-1)
        log_q = F.log_softmax(logits2 / self.temperature, dim=-1)
        log_m = torch.log(m + 1e-10) # Avoid log(0)

        # KL terms (sum over vocab dimension)
        kl_pm = torch.sum(p * (log_p - log_m), dim=-1) # (batch, seq)
        kl_qm = torch.sum(q * (log_q - log_m), dim=-1) # (batch, seq)

        # 5. JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        jsd = 0.5 * (kl_pm + kl_qm)  # (batch, seq)

        # 6. Apply masking
        if attention_mask1 is not None and attention_mask2 is not None:
            # Only consider positions where BOTH sequences have valid tokens
            # attention_mask is 1 for valid, 0 for pad
            valid_mask = (attention_mask1 * attention_mask2).float()
            
            # Mask out invalid positions
            jsd = jsd * valid_mask
            
            # Average over total number of valid tokens
            num_valid = valid_mask.sum()
            if num_valid > 0:
                return jsd.sum() / num_valid
            else:
                return torch.tensor(0.0, device=jsd.device, requires_grad=True)
        else:
            return jsd.mean()
