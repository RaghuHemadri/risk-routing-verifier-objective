"""
Verifier Trainer: stepwise + final-outcome supervision.

Trains V_φ to predict success/progress for (x, a) pairs:

  L_V(φ) = -E[y^final log V_φ(x,a) + (1-y^final) log(1-V_φ(x,a))]

Optionally multi-tasks with step-level labels y^step_t (agent analogue
of process-supervised reward modeling).
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


class VerifierTrainer:
    """Trainer for the step/outcome verifier."""

    def __init__(
        self,
        verifier,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] = None,
        output_dir: str = "experiments/checkpoints/verifier",
    ):
        self.verifier = verifier
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = self.config.get("epochs", 5)
        self.batch_size = self.config.get("batch_size", 8)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)
        self.lr = self.config.get("learning_rate", 1e-5)
        self.weight_decay = self.config.get("weight_decay", 0.01)

        # Multi-task weights
        mt_config = self.config.get("multitask", {})
        self.step_weight = mt_config.get("step_weight", 0.3)
        self.final_weight = mt_config.get("final_weight", 1.0)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task verifier loss.

        L = w_final * BCE(V_final, y_final) + w_step * BCE(V_step, y_step)
        """
        metrics = {}

        # Final outcome loss (always computed)
        final_loss = F.binary_cross_entropy(
            outputs["final_score"],
            batch["final_label"],
            reduction="mean",
        )
        total_loss = self.final_weight * final_loss
        metrics["final_loss"] = final_loss.item()

        # Step-level loss (only for samples with step labels)
        if "step_score" in outputs and "has_step_label" in batch:
            has_label = batch["has_step_label"].bool()
            if has_label.any():
                step_loss = F.binary_cross_entropy(
                    outputs["step_score"][has_label],
                    batch["step_label"][has_label],
                    reduction="mean",
                )
                total_loss = total_loss + self.step_weight * step_loss
                metrics["step_loss"] = step_loss.item()

        metrics["total_loss"] = total_loss.item()

        # Accuracy metrics
        with torch.no_grad():
            final_preds = (outputs["final_score"] > 0.5).float()
            metrics["final_accuracy"] = (
                final_preds == batch["final_label"]
            ).float().mean().item()

        return total_loss, metrics

    def train(self, accelerator=None) -> dict[str, Any]:
        """Run verifier training."""
        from accelerate import Accelerator

        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_steps,
                mixed_precision="bf16",
            )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # Only train the classification heads (backbone is frozen)
        trainable_params = [
            p for p in self.verifier.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.verifier, optimizer, train_loader, scheduler
        )

        global_step = 0
        metrics_history = []

        for epoch in range(self.epochs):
            model.train()
            epoch_metrics = {}
            num_batches = 0

            for batch in train_loader:
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss, metrics = self.compute_loss(outputs, batch)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                num_batches += 1

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % 20 == 0:
                        avg = {k: v / num_batches for k, v in epoch_metrics.items()}
                        accelerator.print(
                            f"[Verifier] Epoch {epoch+1} Step {global_step} "
                            f"Loss={avg.get('total_loss', 0):.4f} "
                            f"Acc={avg.get('final_accuracy', 0):.3f}"
                        )

            avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = epoch + 1
            metrics_history.append(avg_metrics)

        if accelerator.is_main_process:
            save_path = self.output_dir / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.verifier.state_dict(), save_path / "verifier.pt")

        return {"history": metrics_history, "total_steps": global_step}
