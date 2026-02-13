"""
DPO-style Preference Trainer for verifier-guided distillation.

For each context x, samples K candidates from π_θ, scores with V_φ,
and trains with DPO preference loss:

  L_pref(θ) = -E_x[log σ(β(log π_θ(a+|x) - log π_θ(a-|x)))]

where a+ = argmax V_φ(x, a^(k)), a- = argmin V_φ(x, a^(k)).

This directly implements "small models need strong verifiers" as a
training signal, converting verifier scores into dense preference pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler

logger = logging.getLogger(__name__)


class PreferenceTrainer:
    """DPO preference distillation trainer."""

    def __init__(
        self,
        policy,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] = None,
        output_dir: str = "experiments/checkpoints/preference",
    ):
        self.policy = policy
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DPO hyperparameters
        self.beta = self.config.get("beta", 0.1)
        self.epochs = self.config.get("epochs", 2)
        self.batch_size = self.config.get("batch_size", 2)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 16)
        self.lr = self.config.get("learning_rate", 5e-6)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)

        # Reference model: frozen copy for DPO
        self.use_reference = self.config.get("use_reference_model", True)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss.

        DPO loss:
        L = -log σ(β * ((log π(a+|x) - log π_ref(a+|x))
                       - (log π(a-|x) - log π_ref(a-|x))))

        Returns:
            Tuple of (loss, metrics dict)
        """
        # Log-ratio differences
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # DPO implicit reward difference
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # DPO loss
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (logits > 0).float().mean().item()

        metrics = {
            "dpo_loss": loss.item(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss, metrics

    def train(self, accelerator=None) -> dict[str, Any]:
        """Run DPO preference training."""
        from accelerate import Accelerator

        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_steps,
                mixed_precision="bf16",
            )

        # Create reference model (frozen policy copy)
        if self.use_reference:
            import copy
            ref_model = copy.deepcopy(self.policy.model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
        else:
            ref_model = None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        trainable_params = [
            p for p in self.policy.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.policy.model, optimizer, train_loader, scheduler
        )
        if ref_model:
            ref_model = accelerator.prepare(ref_model)

        global_step = 0
        metrics_history = []

        for epoch in range(self.epochs):
            model.train()
            epoch_metrics = {"dpo_loss": 0, "accuracy": 0, "reward_margin": 0}
            num_batches = 0

            for batch in train_loader:
                with accelerator.accumulate(model):
                    # Compute policy log probs for chosen and rejected
                    policy_chosen_logps = self._compute_logps(
                        model, batch["chosen_input_ids"],
                        batch["chosen_attention_mask"], batch["chosen_labels"]
                    )
                    policy_rejected_logps = self._compute_logps(
                        model, batch["rejected_input_ids"],
                        batch["rejected_attention_mask"], batch["rejected_labels"]
                    )

                    # Compute reference log probs
                    if ref_model is not None:
                        with torch.no_grad():
                            ref_chosen_logps = self._compute_logps(
                                ref_model, batch["chosen_input_ids"],
                                batch["chosen_attention_mask"], batch["chosen_labels"]
                            )
                            ref_rejected_logps = self._compute_logps(
                                ref_model, batch["rejected_input_ids"],
                                batch["rejected_attention_mask"],
                                batch["rejected_labels"]
                            )
                    else:
                        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

                    loss, metrics = self.compute_dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_params, self.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                num_batches += 1

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % 10 == 0:
                        avg = {k: v / num_batches for k, v in epoch_metrics.items()}
                        accelerator.print(
                            f"[DPO] Epoch {epoch+1} Step {global_step} "
                            f"Loss={avg['dpo_loss']:.4f} "
                            f"Acc={avg['accuracy']:.3f} "
                            f"Margin={avg['reward_margin']:.3f}"
                        )

            avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = epoch + 1
            metrics_history.append(avg_metrics)

        if accelerator.is_main_process:
            save_path = self.output_dir / "final"
            accelerator.save_state(str(save_path))

        return {"history": metrics_history, "total_steps": global_step}

    def _compute_logps(
        self, model, input_ids, attention_mask, labels
    ) -> torch.Tensor:
        """Compute per-sequence log probabilities."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs["logits"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != -100).float()
        return (token_log_probs * mask).sum(dim=-1)
