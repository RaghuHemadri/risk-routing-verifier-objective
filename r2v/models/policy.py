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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

_HF_TOKEN = os.environ.get("HF_TOKEN", None)

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
            self.model_name, trust_remote_code=True, token=_HF_TOKEN
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        # NOTE: 4-bit quantization (bitsandbytes) is incompatible with
        # Accelerate DDP multi-GPU training — quantized weights can't be
        # wrapped in DistributedDataParallel. When WORLD_SIZE > 1, skip
        # quantization entirely and use bf16 (H200 80GB has plenty of VRAM).
        quant_config = None
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        quant_settings = config.get("quantization", {})
        if quant_settings.get("load_in_4bit", False) and world_size <= 1:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_settings.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
                bnb_4bit_quant_type=quant_settings.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=True,
            )
        elif quant_settings.get("load_in_4bit", False) and world_size > 1:
            logger.info(
                "Skipping 4-bit quantization for multi-GPU DDP training "
                f"(WORLD_SIZE={world_size}). Using bf16 instead."
            )

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "token": _HF_TOKEN,
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Sync pad_token_id into model config to suppress
        # "Setting pad_token_id to eos_token_id" warning during generate()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Apply LoRA if configured
        lora_config = config.get("lora", {})
        if lora_config.get("enabled", False):
            self._apply_lora(lora_config)

        # Enable gradient checkpointing for memory efficiency
        if config.get("gradient_checkpointing", True):
            # use_reentrant=False is required for LoRA/PEFT — the default
            # (reentrant) silently drops gradients when base-model params are frozen.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # PEFT helper: make embedding outputs require grad so checkpointing
            # can propagate through frozen → trainable boundaries.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

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

        Uses num_return_sequences to generate all K candidates in a
        single forward pass instead of looping K times.

        Returns list of dicts with 'text', 'log_prob', 'num_tokens'.
        """
        prompt = f"{context}\nAction:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_seq_len - max_new_tokens,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_candidates,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prompt_len = inputs["input_ids"].shape[1]
        candidates = []
        for i in range(num_candidates):
            generated_ids = outputs.sequences[i, prompt_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Compute log probability of the generated sequence
            log_prob = 0.0
            if outputs.scores:
                for t, score_t in enumerate(outputs.scores):
                    if t >= len(generated_ids):
                        break
                    token_id = generated_ids[t]
                    log_probs = F.log_softmax(score_t[i], dim=-1)
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
