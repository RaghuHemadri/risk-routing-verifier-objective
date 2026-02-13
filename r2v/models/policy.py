"""
SLM Policy model: decoder-only LM with optional LoRA fine-tuning.

The policy π_θ(a|x) generates action token sequences given the agent
context x_t = (G, o_≤t, a_<t, y_<t).

Supports:
- LoRA fine-tuning via PEFT
- 4-bit quantization via bitsandbytes
- Flash Attention 2
- Gradient checkpointing
- Sampling K candidates with log-probabilities (for verifier scoring)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    """SLM policy backbone with LoRA support."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config["model_name"]
        self.max_seq_len = config.get("max_seq_len", 4096)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quant_config = None
        quant_settings = config.get("quantization", {})
        if quant_settings.get("load_in_4bit", False):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_settings.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
                bnb_4bit_quant_type=quant_settings.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=True,
            )

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        if config.get("quantization", {}).get("load_in_4bit"):
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Apply LoRA if configured
        lora_config = config.get("lora", {})
        if lora_config.get("enabled", False):
            self._apply_lora(lora_config)

        # Enable gradient checkpointing for memory efficiency
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()

        logger.info(
            f"PolicyModel initialized: {self.model_name}, "
            f"params={sum(p.numel() for p in self.model.parameters()):,}, "
            f"trainable={sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

    def _apply_lora(self, lora_config: dict):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("alpha", 128),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training (returns loss + logits)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for given sequences.

        Used for DPO preference loss and consistency regularization.
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding and prompt tokens (where labels == -100)
        mask = (shift_labels != -100) & (shift_labels != self.tokenizer.pad_token_id)
        token_log_probs = token_log_probs * mask.float()

        # Sum log probs per sequence
        return token_log_probs.sum(dim=-1)

    def compute_action_distribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Get the next-token distribution for entropy/KL computation."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # Use last non-padding position
        seq_lens = attention_mask.sum(dim=1) - 1
        last_logits = outputs.logits[
            torch.arange(input_ids.size(0)), seq_lens
        ]
        return torch.distributions.Categorical(logits=last_logits)

    @torch.no_grad()
    def generate_candidates(
        self,
        context: str,
        num_candidates: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Generate K candidate actions with log-probabilities.

        Returns list of dicts with 'text', 'log_prob', 'tokens'.
        """
        prompt = f"{context}\nAction:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_seq_len - max_new_tokens,
        ).to(self.model.device)

        candidates = []
        for _ in range(num_candidates):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Compute log probability of the generated sequence
            log_prob = 0.0
            if outputs.scores:
                for t, score in enumerate(outputs.scores):
                    if t >= len(generated_ids):
                        break
                    token_id = generated_ids[t]
                    log_probs = F.log_softmax(score[0], dim=-1)
                    log_prob += log_probs[token_id].item()

            candidates.append({
                "text": generated_text,
                "log_prob": log_prob,
                "num_tokens": len(generated_ids),
            })

        return candidates

    @torch.no_grad()
    def compute_entropy(self, context: str) -> float:
        """Compute entropy H(π_θ(·|x)) of next-token distribution."""
        prompt = f"{context}\nAction:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_seq_len,
        ).to(self.model.device)

        dist = self.compute_action_distribution(
            inputs["input_ids"], inputs["attention_mask"]
        )
        return dist.entropy().item()

    def get_hidden_state(
        self, context: str, layer: int = -1
    ) -> torch.Tensor:
        """Extract hidden state summary for router input."""
        inputs = self.tokenizer(
            context, return_tensors="pt", truncation=True,
            max_length=self.max_seq_len,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True
            )

        hidden_states = outputs.hidden_states[layer]
        # Mean pool over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.squeeze(0)

    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load model weights from checkpoint."""
        from peft import PeftModel
        if hasattr(self.model, 'peft_config'):
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(
                self.model.base_model.model, path
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path)
