"""
Behavior Cloning (BC) Trainer for the SLM policy.

Trains π_θ to imitate teacher actions on successful trajectories:
  L_BC(θ) = -E_{(x,a*)~D_T}[log π_θ(a*|x)]

Uses HuggingFace Accelerate + DeepSpeed for distributed training,
with LoRA for parameter-efficient fine-tuning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

logger = logging.getLogger(__name__)


class BCTrainer:
    """Behavior cloning trainer for the SLM policy."""

    def __init__(
        self,
        policy,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] = None,
        output_dir: str = "experiments/checkpoints/bc",
    ):
        self.policy = policy
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.epochs = self.config.get("epochs", 3)
        self.batch_size = self.config.get("batch_size", 4)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 8)
        self.lr = self.config.get("learning_rate", 2e-5)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.warmup_ratio = self.config.get("warmup_ratio", 0.05)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.scheduler_type = self.config.get("scheduler", "cosine")

        # Logging
        self.log_interval = self.config.get("log_interval", 10)
        self.eval_interval = self.config.get("eval_interval", 100)
        self.save_interval = self.config.get("save_interval", 500)

    def train(self, accelerator=None) -> dict[str, Any]:
        """Run behavior cloning training loop.

        Args:
            accelerator: HuggingFace Accelerator for distributed training.
                        If None, creates a default single-GPU one.

        Returns:
            Training metrics dictionary.
        """
        from accelerate import Accelerator

        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_steps,
                mixed_precision="bf16",
            )

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        # Optimizer (only train LoRA params if applicable)
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Prepare with accelerator
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.policy.model, optimizer, train_loader, scheduler
        )
        if eval_loader:
            eval_loader = accelerator.prepare(eval_loader)

        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        metrics_history = []

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_params, self.max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

                if accelerator.sync_gradients:
                    global_step += 1

                    # Logging
                    if global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        lr_current = scheduler.get_last_lr()[0]
                        accelerator.print(
                            f"[BC] Epoch {epoch+1}/{self.epochs} "
                            f"Step {global_step}/{total_steps} "
                            f"Loss={avg_loss:.4f} LR={lr_current:.2e}"
                        )

                        # wandb logging
                        if accelerator.is_main_process:
                            try:
                                import wandb
                                if wandb.run is not None:
                                    wandb.log({
                                        "bc/train_loss": avg_loss,
                                        "bc/learning_rate": lr_current,
                                        "bc/epoch": epoch + 1,
                                        "bc/global_step": global_step,
                                    }, step=global_step)
                            except ImportError:
                                pass

                    # Evaluation
                    if eval_loader and global_step % self.eval_interval == 0:
                        eval_loss = self._evaluate(model, eval_loader, accelerator)
                        accelerator.print(f"[BC] Eval Loss={eval_loss:.4f}")

                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            if accelerator.is_main_process:
                                self._save_checkpoint(
                                    accelerator, "best", global_step
                                )

                    # Periodic save
                    if (
                        accelerator.is_main_process
                        and global_step % self.save_interval == 0
                    ):
                        self._save_checkpoint(
                            accelerator, f"step_{global_step}", global_step
                        )

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "global_step": global_step,
            })

        # Final save
        if accelerator.is_main_process:
            self._save_checkpoint(accelerator, "final", global_step)

        return {
            "final_train_loss": avg_epoch_loss,
            "best_eval_loss": best_eval_loss,
            "total_steps": global_step,
            "history": metrics_history,
        }

    @torch.no_grad()
    def _evaluate(self, model, eval_loader, accelerator) -> float:
        """Run evaluation and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

        model.train()
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, accelerator, name: str, step: int):
        """Save model checkpoint."""
        save_path = self.output_dir / name
        accelerator.save_state(str(save_path))
        logger.info(f"[BC] Saved checkpoint to {save_path} (step {step})")
