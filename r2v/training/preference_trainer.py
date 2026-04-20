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
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler

from r2v.training.consistency import ConsistencyRegularizer

logger = logging.getLogger(__name__)


class PreferenceTrainer:
    """DPO preference distillation trainer."""

    def __init__(
        self,
        policy,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] | None = None,
        output_dir: str = "experiments/checkpoints/preference",
        collate_fn=None,
        start_epoch: int = 0,
        resume_state_path: str | None = None,
        save_epoch_checkpoints: bool = True,
    ):
        self.policy = policy
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collate_fn = collate_fn
        self.start_epoch = start_epoch
        self.resume_state_path = resume_state_path
        self.save_epoch_checkpoints = save_epoch_checkpoints

        # DPO hyperparameters
        self.beta = self.config.get("beta", 0.1)
        self.epochs = self.config.get("epochs", 2)
        self.batch_size = self.config.get("batch_size", 2)
        self.grad_accum_steps = self.config.get("gradient_accumulation_steps", 16)
        self.lr = self.config.get("learning_rate", 5e-6)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.gpu_keepalive_interval = float(self.config.get("gpu_keepalive_interval", 0.0))
        # Memory-safe by default: separate chosen/rejected passes.
        self.concat_pairs = self.config.get("concat_pairs", False)

        # Reference model: frozen copy for DPO
        self.use_reference = self.config.get("use_reference_model", True)

        # Consistency regularization (R-Drop style: two dropout views of chosen)
        cons_cfg = self.config.get("consistency", {})
        self.consistency_enabled = bool(cons_cfg.get("enabled", False))
        self.lambda_cons = float(cons_cfg.get("lambda_cons", 0.1))
        _cons_temp = float(cons_cfg.get("temperature", 2.0))
        self.consistency_reg = ConsistencyRegularizer(temperature=_cons_temp) if self.consistency_enabled else None
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None

    def _start_gpu_keepalive(self):
        if self.gpu_keepalive_interval <= 0 or not torch.cuda.is_available():
            return
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()

        def _worker():
            device = torch.device("cuda")
            a = torch.randn((512, 512), device=device, dtype=torch.float16)
            b = torch.randn((512, 512), device=device, dtype=torch.float16)
            while not self._keepalive_stop.is_set():
                try:
                    _ = a @ b
                    torch.cuda.synchronize()
                except Exception:
                    break
                self._keepalive_stop.wait(self.gpu_keepalive_interval)

        self._keepalive_thread = threading.Thread(target=_worker, daemon=True)
        self._keepalive_thread.start()
        logger.info(
            "[DPO] GPU keepalive enabled (interval=%.2fs)", self.gpu_keepalive_interval
        )

    def _stop_gpu_keepalive(self):
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None

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

        # DataLoader — num_workers=2 prefetches on CPU, pin_memory for fast transfers.
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
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=_use_workers,
                pin_memory=True,
                persistent_workers=_use_workers > 0,
                prefetch_factor=2 if _use_workers > 0 else None,
                collate_fn=self.collate_fn,
            )

        trainable_params = [
            p for p in self.policy.model.parameters() if p.requires_grad
        ]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found for preference training. "
                "If resuming a LoRA checkpoint, ensure adapters are loaded as trainable "
                "(e.g., PeftModel.from_pretrained(..., is_trainable=True))."
            )
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
        if eval_loader is not None:
            eval_loader = accelerator.prepare(eval_loader)

        if self.resume_state_path:
            logger.info(f"[DPO] Restoring Accelerate state from {self.resume_state_path}")
            accelerator.load_state(self.resume_state_path)
            logger.info(f"[DPO] State restored; resuming from epoch {self.start_epoch + 1}")

        global_step = 0
        best_eval_acc = float("-inf")
        metrics_history = []

        self._start_gpu_keepalive()
        try:
            for epoch in range(self.start_epoch, self.epochs):
                model.train()
                epoch_metrics = {"dpo_loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0, "cons_loss": 0.0}
                num_batches = 0

                for batch in train_loader:
                    with accelerator.accumulate(model):
                        if self.concat_pairs:
                            # Faster path, but uses higher peak memory.
                            concat_ids = torch.cat(
                                [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
                            )
                            concat_mask = torch.cat(
                                [batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0
                            )
                            concat_labels = torch.cat(
                                [batch["chosen_labels"], batch["rejected_labels"]], dim=0
                            )
                            bs = batch["chosen_input_ids"].size(0)

                            if self.consistency_enabled:
                                policy_logps, all_logits = self._compute_logps(
                                    model, concat_ids, concat_mask, concat_labels,
                                    return_logits=True,
                                )
                                chosen_logits_v1 = all_logits[:bs]
                            else:
                                policy_logps = self._compute_logps(
                                    model, concat_ids, concat_mask, concat_labels
                                )
                            policy_chosen_logps = policy_logps[:bs]
                            policy_rejected_logps = policy_logps[bs:]

                            if ref_model is not None:
                                with torch.no_grad():
                                    ref_logps = self._compute_logps(
                                        ref_model, concat_ids, concat_mask, concat_labels
                                    )
                                    ref_chosen_logps = ref_logps[:bs]
                                    ref_rejected_logps = ref_logps[bs:]
                            else:
                                ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                                ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                        else:
                            # Memory-safe path: separate chosen/rejected passes.
                            if self.consistency_enabled:
                                policy_chosen_logps, chosen_logits_v1 = self._compute_logps(
                                    model,
                                    batch["chosen_input_ids"],
                                    batch["chosen_attention_mask"],
                                    batch["chosen_labels"],
                                    return_logits=True,
                                )
                            else:
                                policy_chosen_logps = self._compute_logps(
                                    model,
                                    batch["chosen_input_ids"],
                                    batch["chosen_attention_mask"],
                                    batch["chosen_labels"],
                                )
                            policy_rejected_logps = self._compute_logps(
                                model,
                                batch["rejected_input_ids"],
                                batch["rejected_attention_mask"],
                                batch["rejected_labels"],
                            )

                            if ref_model is not None:
                                with torch.no_grad():
                                    ref_chosen_logps = self._compute_logps(
                                        ref_model,
                                        batch["chosen_input_ids"],
                                        batch["chosen_attention_mask"],
                                        batch["chosen_labels"],
                                    )
                                    ref_rejected_logps = self._compute_logps(
                                        ref_model,
                                        batch["rejected_input_ids"],
                                        batch["rejected_attention_mask"],
                                        batch["rejected_labels"],
                                    )
                            else:
                                ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                                ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

                        loss, metrics = self.compute_dpo_loss(
                            policy_chosen_logps, policy_rejected_logps,
                            ref_chosen_logps, ref_rejected_logps,
                        )

                        if self.consistency_enabled:
                            if "alt_input_ids" in batch:
                                # True perturbation consistency: same action, different perturbed context
                                _, alt_logits = self._compute_logps(
                                    model,
                                    batch["alt_input_ids"],
                                    batch["alt_attention_mask"],
                                    batch["alt_labels"],
                                    return_logits=True,
                                )
                                chosen_shift_mask = (batch["chosen_labels"][:, 1:] != -100).float()
                                alt_shift_mask = (batch["alt_labels"][:, 1:] != -100).float()
                                cons_loss = self.consistency_reg(
                                    chosen_logits_v1, alt_logits,
                                    chosen_shift_mask, alt_shift_mask,
                                )
                            else:
                                # R-Drop fallback: second dropout pass on chosen
                                _, alt_logits = self._compute_logps(
                                    model,
                                    batch["chosen_input_ids"],
                                    batch["chosen_attention_mask"],
                                    batch["chosen_labels"],
                                    return_logits=True,
                                )
                                shift_mask = (batch["chosen_labels"][:, 1:] != -100).float()
                                cons_loss = self.consistency_reg(
                                    chosen_logits_v1, alt_logits, shift_mask, shift_mask,
                                )
                            total_loss = loss + self.lambda_cons * cons_loss
                            metrics["cons_loss"] = cons_loss.item()
                        else:
                            total_loss = loss

                        accelerator.backward(total_loss)
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
                                + (f" Cons={avg['cons_loss']:.4f}" if self.consistency_enabled else "")
                            )

                avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
                avg_metrics["epoch"] = epoch + 1

                if eval_loader is not None:
                    eval_metrics = self._evaluate(model, ref_model, eval_loader, accelerator)
                    if eval_metrics is not None:
                        eval_loss = float(eval_metrics["eval_loss"])
                        eval_acc = float(eval_metrics["eval_accuracy"])
                        avg_metrics["eval_loss"] = eval_loss
                        avg_metrics["eval_accuracy"] = eval_acc
                        accelerator.print(
                            f"[DPO] Epoch {epoch+1} Eval Loss={eval_loss:.4f} Eval Acc={eval_acc:.4f}"
                        )
                        if eval_acc > best_eval_acc:
                            best_eval_acc = eval_acc
                            if accelerator.is_main_process:
                                self._save_checkpoint(accelerator, model, "best", global_step)
                    else:
                        accelerator.print("[DPO] Eval skipped due to OOM; continuing training")

                metrics_history.append(avg_metrics)

                if accelerator.is_main_process:
                    self._save_checkpoint(accelerator, model, f"epoch_{epoch + 1}", global_step)

            if accelerator.is_main_process:
                self._save_checkpoint(accelerator, model, "final", global_step)
        finally:
            self._stop_gpu_keepalive()

        return {
            "history": metrics_history,
            "total_steps": global_step,
            "best_eval_accuracy": best_eval_acc,
        }

    def _save_checkpoint(self, accelerator, model, name: str, step: int):
        """Save checkpoint.

        Epoch checkpoints (``epoch_*``) write the full Accelerate state
        (model + optimizer + scheduler + RNG) so training can be resumed.

        Best / final checkpoints write only the PEFT adapter weights,
        adapter_config.json, and tokenizer — the minimal set needed to
        reload via ``PeftModel.from_pretrained`` / ``AutoModelForCausalLM``.
        """
        save_path = self.output_dir / name
        if name.startswith("epoch_"):
            if not self.save_epoch_checkpoints:
                logger.info(f"[DPO] Skipping epoch checkpoint {name} (save_epoch_checkpoints=False)")
                return
            save_path.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(str(save_path))
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(save_path))
            self.policy.tokenizer.save_pretrained(str(save_path))
        logger.info(f"[DPO] Saved checkpoint to {save_path} (step {step})")

    @torch.no_grad()
    def _evaluate(self, model, ref_model, eval_loader, accelerator) -> dict[str, float] | None:
        """Run evaluation and return average DPO loss + preference accuracy."""
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in eval_loader:
            try:
                policy_chosen_logps = self._compute_logps(
                    model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_rejected_logps = self._compute_logps(
                    model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

                if ref_model is not None:
                    ref_chosen_logps = self._compute_logps(
                        ref_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    )
                    ref_rejected_logps = self._compute_logps(
                        ref_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                    )
                else:
                    ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

                loss, metrics = self.compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                )
                total_loss += loss.item()
                total_acc += float(metrics["accuracy"])
                num_batches += 1
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                model.train()
                return None

        model.train()
        denom = max(num_batches, 1)
        return {
            "eval_loss": total_loss / denom,
            "eval_accuracy": total_acc / denom,
        }

    def _compute_logps(
        self, model, input_ids, attention_mask, labels, return_logits: bool = False
    ):
        """Compute per-sequence log probabilities.

        Args:
            return_logits: If True, also return shifted logits (batch, seq-1, vocab)
                           for use in the consistency regularizer.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs["logits"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Memory-efficient token log-prob computation:
        # log p(y_t) = z_{y_t} - logsumexp(z)
        target_logits = shift_logits.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        normalizer = torch.logsumexp(shift_logits, dim=-1)
        token_log_probs = target_logits - normalizer

        mask = (shift_labels != -100).float()
        logps = (token_log_probs * mask).sum(dim=-1)
        if return_logits:
            return logps, shift_logits
        return logps
