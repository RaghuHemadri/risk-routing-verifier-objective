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
        collate_fn=None,
        resume_state_path: str | None = None,
    ):
        self.policy = policy
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collate_fn = collate_fn
        self.resume_state_path = resume_state_path

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

        # DataLoader — num_workers=2 prefetches batches on CPU while GPU trains,
        # pin_memory speeds up CPU→GPU transfer on CUDA devices.
        _use_workers = self.config.get("num_workers", 2)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=_use_workers,
            pin_memory=True,
            persistent_workers=_use_workers > 0,
            prefetch_factor=2 if _use_workers > 0 else None,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        eval_loader = None
        if self.eval_dataset:
            eval_batch_size = self.config.get("eval_batch_size", self.batch_size)
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=_use_workers,
                pin_memory=True,
                persistent_workers=_use_workers > 0,
                prefetch_factor=2 if _use_workers > 0 else None,
                collate_fn=self.collate_fn,
            )

        # Optimizer (only train LoRA params if applicable)
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Prepare with accelerator first (handles model → GPU, DataLoader sharding)
        # so len(train_loader) reflects per-process shard size.
        model, optimizer, train_loader = accelerator.prepare(
            self.policy.model, optimizer, train_loader
        )

        # LR scheduler — compute total_steps AFTER prepare so len(train_loader)
        # reflects the per-GPU shard (dataset_size / num_processes / batch_size).
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )
        scheduler = accelerator.prepare(scheduler)
        if eval_loader:
            eval_loader = accelerator.prepare(eval_loader)

        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        metrics_history = []

        # Resume full trainer state (model + optimizer + scheduler + RNG)
        if self.resume_state_path:
            ckpt_path = Path(self.resume_state_path)
            accelerator.print(f"[BC] Loading accelerate state from {ckpt_path}")
            accelerator.load_state(str(ckpt_path))
            name = ckpt_path.name
            if name.startswith("step_"):
                try:
                    global_step = int(name.split("_")[-1])
                except ValueError:
                    global_step = 0
            accelerator.print(f"[BC] Resumed at global_step={global_step}")

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    if (batch["labels"] != -100).sum().item() == 0:
                        accelerator.print(
                            "[BC] Warning: skipping batch with no supervised tokens"
                        )
                        optimizer.zero_grad()
                        continue

                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]

                    if not torch.isfinite(loss):
                        accelerator.print(
                            "[BC] Warning: non-finite loss encountered; skipping batch"
                        )
                        optimizer.zero_grad()
                        continue

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
                        if eval_loss is None:
                            accelerator.print(
                                "[BC] Eval skipped due to OOM; continuing training"
                            )
                        else:
                            accelerator.print(f"[BC] Eval Loss={eval_loss:.4f}")

                        if eval_loss is not None and eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            if accelerator.is_main_process:
                                self._save_checkpoint(
                                    accelerator, "best", global_step
                                )


            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "global_step": global_step,
            })

            if accelerator.is_main_process:
                self._save_checkpoint(
                    accelerator, f"epoch_{epoch + 1}", global_step
                )

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
    def _evaluate(self, model, eval_loader, accelerator) -> float | None:
        """Run evaluation and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]
                total_loss += loss.item()
                num_batches += 1
            except torch.OutOfMemoryError:
                # Eval is optional; recover instead of killing the full run.
                torch.cuda.empty_cache()
                model.train()
                return None

        model.train()
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, accelerator, name: str, step: int):
        """Save model checkpoint."""
        save_path = self.output_dir / name
        accelerator.save_state(str(save_path))
        logger.info(f"[BC] Saved checkpoint to {save_path} (step {step})")
