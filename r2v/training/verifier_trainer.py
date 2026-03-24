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


def _binary_clf_metrics(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy, and AUROC for binary predictions.

    Args:
        scores: predicted probabilities in [0, 1]
        labels: ground-truth binary labels (0 or 1)
        threshold: decision boundary for precision/recall/F1/accuracy
    """
    preds = (scores > threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # AUROC — sort-based, no sklearn dependency
    auroc = _compute_auroc(scores, labels)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auroc": auroc,
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
    }


def _find_best_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_thr: float = 0.05,
    max_thr: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """Grid-search threshold that maximizes F1 on provided scores/labels."""
    best_thr = 0.5
    best_f1 = -1.0

    n = int(round((max_thr - min_thr) / step))
    for i in range(n + 1):
        thr = min_thr + i * step
        f1 = _binary_clf_metrics(scores, labels, threshold=thr)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr), float(best_f1)


def _compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUROC via the trapezoidal rule (no external dependencies)."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # undefined when only one class is present

    desc_idx = scores.argsort(descending=True)
    sorted_labels = labels[desc_idx].float()
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0.0, 0.0
    auc = 0.0

    for label_val in sorted_labels:
        if label_val == 1.0:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (tpr + tpr_prev) * (fpr - fpr_prev)
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)


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

        # Class-imbalance handling for final outcome loss.
        # Positive samples are weighted by this factor in BCE.
        # If auto_class_balance=True and no explicit weight is provided,
        # it is estimated as (#neg / #pos) from the training dataset.
        self.auto_class_balance = bool(self.config.get("auto_class_balance", True))
        self.positive_class_weight = self.config.get("positive_class_weight", None)
        if self.positive_class_weight is not None:
            self.positive_class_weight = float(self.positive_class_weight)
        self.max_class_weight = float(self.config.get("max_class_weight", 5.0))

        # Optional focal modulation on top of BCE for hard-example emphasis.
        # When disabled, loss reduces to (weighted) BCE as before.
        self.use_focal_loss = bool(self.config.get("use_focal_loss", False))
        self.focal_gamma = float(self.config.get("focal_gamma", 2.0))
        # Optional alpha balancing for focal loss. If null, no alpha term is used.
        focal_alpha = self.config.get("focal_alpha", None)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

        # Use a learned validation threshold instead of hard-coded 0.5.
        self.decision_threshold = float(self.config.get("decision_threshold", 0.5))
        self.tune_threshold_on_val = bool(self.config.get("tune_threshold_on_val", True))

    def _ensure_class_weight(self):
        """Infer positive class weight from train dataset when requested."""
        if self.positive_class_weight is not None:
            return

        if not self.auto_class_balance:
            self.positive_class_weight = 1.0
            return

        examples = getattr(self.train_dataset, "examples", None)
        if not examples:
            self.positive_class_weight = 1.0
            logger.warning(
                "Could not infer class balance from train dataset; using positive_class_weight=1.0"
            )
            return

        labels = [float(ex.get("final_label", 0.0)) for ex in examples]
        n = len(labels)
        n_pos = sum(labels)
        n_neg = n - n_pos

        if n_pos <= 0 or n_neg <= 0:
            self.positive_class_weight = 1.0
            logger.warning(
                "Single-class training labels detected (pos=%s, neg=%s); "
                "using positive_class_weight=1.0",
                int(n_pos), int(n_neg),
            )
            return

        weight = n_neg / max(n_pos, 1e-8)
        weight = min(weight, self.max_class_weight)
        self.positive_class_weight = float(weight)
        logger.info(
            "Auto class balance: n=%d, pos=%d, neg=%d, positive_class_weight=%.4f",
            n, int(n_pos), int(n_neg), self.positive_class_weight,
        )

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task verifier loss.

        L = w_final * BCE(V_final, y_final) + w_step * BCE(V_step, y_step)
        """
        metrics = {}

        pos_w = float(self.positive_class_weight or 1.0)

        # Final outcome loss (always computed)
        final_weights = torch.where(
            batch["final_label"] > 0.5,
            torch.full_like(batch["final_label"], pos_w),
            torch.ones_like(batch["final_label"]),
        )
        final_bce = F.binary_cross_entropy(
            outputs["final_score"],
            batch["final_label"],
            weight=final_weights,
            reduction="none",
        )

        if self.use_focal_loss:
            probs = outputs["final_score"].clamp(min=1e-6, max=1 - 1e-6)
            pt = torch.where(batch["final_label"] > 0.5, probs, 1.0 - probs)
            focal_factor = (1.0 - pt).pow(self.focal_gamma)

            if self.focal_alpha is not None:
                alpha_factor = torch.where(
                    batch["final_label"] > 0.5,
                    torch.full_like(batch["final_label"], self.focal_alpha),
                    torch.full_like(batch["final_label"], 1.0 - self.focal_alpha),
                )
            else:
                alpha_factor = torch.ones_like(batch["final_label"])

            final_loss = (final_bce * focal_factor * alpha_factor).mean()
        else:
            final_loss = final_bce.mean()

        total_loss = self.final_weight * final_loss
        metrics["final_loss"] = final_loss.item()
        metrics["positive_class_weight"] = pos_w
        metrics["use_focal_loss"] = float(self.use_focal_loss)
        metrics["focal_gamma"] = float(self.focal_gamma)
        if self.focal_alpha is not None:
            metrics["focal_alpha"] = float(self.focal_alpha)

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

        # Classification metrics at the trainer's current decision threshold.
        with torch.no_grad():
            clf = _binary_clf_metrics(
                outputs["final_score"],
                batch["final_label"],
                threshold=self.decision_threshold,
            )
            for k, v in clf.items():
                metrics[f"final_{k}"] = v
            metrics["decision_threshold"] = self.decision_threshold

        return total_loss, metrics

    @torch.no_grad()
    def evaluate(self, model, eval_loader, accelerator) -> dict[str, float]:
        """Run full validation pass and return aggregated metrics."""
        model.eval()

        all_scores = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss, _ = self.compute_loss(outputs, batch)
            total_loss += loss.item()
            num_batches += 1

            all_scores.append(outputs["final_score"])
            all_labels.append(batch["final_label"])

        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Gather across processes for multi-GPU
        if accelerator.num_processes > 1:
            all_scores = accelerator.gather(all_scores)
            all_labels = accelerator.gather(all_labels)

        if self.tune_threshold_on_val:
            tuned_thr, tuned_f1 = _find_best_threshold(all_scores, all_labels)
        else:
            tuned_thr, tuned_f1 = self.decision_threshold, -1.0

        clf = _binary_clf_metrics(all_scores, all_labels, threshold=tuned_thr)
        clf["loss"] = total_loss / max(num_batches, 1)
        clf["n_samples"] = len(all_scores)
        clf["pos_rate"] = all_labels.float().mean().item()
        clf["threshold"] = float(tuned_thr)
        clf["best_f1_threshold_sweep"] = float(tuned_f1)

        model.train()
        return clf

    def train(self, accelerator=None) -> dict[str, Any]:
        """Run verifier training with per-epoch validation."""
        from accelerate import Accelerator

        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_steps,
                mixed_precision="bf16",
            )

        self._ensure_class_weight()

        # Use dynamic-padding collate_fn if available (VerifierDataset provides one)
        collate_fn = getattr(self.train_dataset, 'collate_fn', None)
        if collate_fn is None and hasattr(self.train_dataset, 'dataset'):
            collate_fn = getattr(self.train_dataset.dataset, 'collate_fn', None)

        eval_collate_fn = None
        if self.eval_dataset is not None:
            eval_collate_fn = getattr(self.eval_dataset, 'collate_fn', None)
            if eval_collate_fn is None and hasattr(self.eval_dataset, 'dataset'):
                eval_collate_fn = getattr(self.eval_dataset.dataset, 'collate_fn', None)
            if eval_collate_fn is None:
                eval_collate_fn = collate_fn

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        eval_loader = None
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                collate_fn=eval_collate_fn,
            )

        # Only train the classification heads (backbone is frozen)
        trainable_params = [
            p for p in self.verifier.parameters() if p.requires_grad
        ]

        total_params = sum(p.numel() for p in self.verifier.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        logger.info(
            f"Verifier params: total={total_params:,}, "
            f"trainable={trainable_count:,} ({100*trainable_count/total_params:.2f}%), "
            f"frozen={frozen_count:,}"
        )

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        prepare_args = [self.verifier, optimizer, train_loader, scheduler]
        if eval_loader is not None:
            prepare_args.append(eval_loader)
            model, optimizer, train_loader, scheduler, eval_loader = accelerator.prepare(*prepare_args)
        else:
            model, optimizer, train_loader, scheduler = accelerator.prepare(*prepare_args)

        global_step = 0
        best_val_f1 = -1.0
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
                            f"Acc={avg.get('final_accuracy', 0):.3f} "
                            f"F1={avg.get('final_f1', 0):.3f}"
                        )

            # ── End-of-epoch train summary ──
            avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = epoch + 1

            accelerator.print(
                f"\n[Verifier] ── Epoch {epoch+1}/{self.epochs} Train ──\n"
                f"  Loss={avg_metrics.get('total_loss', 0):.4f}  "
                f"Acc={avg_metrics.get('final_accuracy', 0):.3f}  "
                f"F1={avg_metrics.get('final_f1', 0):.3f}  "
                f"P={avg_metrics.get('final_precision', 0):.3f}  "
                f"R={avg_metrics.get('final_recall', 0):.3f}  "
                f"AUROC={avg_metrics.get('final_auroc', 0):.3f}"
            )

            # ── Validation ──
            if eval_loader is not None:
                val_metrics = self.evaluate(model, eval_loader, accelerator)
                self.decision_threshold = float(
                    val_metrics.get("threshold", self.decision_threshold)
                )
                for k, v in val_metrics.items():
                    avg_metrics[f"val_{k}"] = v

                accelerator.print(
                    f"[Verifier] ── Epoch {epoch+1}/{self.epochs} Val ──\n"
                    f"  Loss={val_metrics['loss']:.4f}  "
                    f"Acc={val_metrics['accuracy']:.3f}  "
                    f"F1={val_metrics['f1']:.3f}  "
                    f"P={val_metrics['precision']:.3f}  "
                    f"R={val_metrics['recall']:.3f}  "
                    f"AUROC={val_metrics['auroc']:.3f}  "
                    f"Thr={val_metrics.get('threshold', self.decision_threshold):.3f}\n"
                    f"  TP={val_metrics['tp']:.0f}  FP={val_metrics['fp']:.0f}  "
                    f"FN={val_metrics['fn']:.0f}  TN={val_metrics['tn']:.0f}  "
                    f"N={val_metrics['n_samples']}  "
                    f"pos_rate={val_metrics['pos_rate']:.2%}"
                )

                # Save best model by val F1
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    if accelerator.is_main_process:
                        best_path = self.output_dir / "best"
                        best_path.mkdir(parents=True, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(unwrapped.state_dict(), best_path / "verifier.pt")
                        (best_path / "decision_threshold.txt").write_text(
                            f"{self.decision_threshold:.6f}\n"
                        )
                        accelerator.print(
                            f"  ✓ New best model saved (val F1={best_val_f1:.4f})"
                        )

            # ── Save epoch checkpoint ──
            if accelerator.is_main_process:
                epoch_path = self.output_dir / f"epoch_{epoch + 1}"
                epoch_path.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), epoch_path / "verifier.pt")
                accelerator.print(
                    f"  Saved epoch {epoch + 1} checkpoint to {epoch_path}"
                )

            metrics_history.append(avg_metrics)

        # Save final model
        if accelerator.is_main_process:
            save_path = self.output_dir / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), save_path / "verifier.pt")
            (save_path / "decision_threshold.txt").write_text(
                f"{self.decision_threshold:.6f}\n"
            )
            accelerator.print(
                f"\n[Verifier] Final model saved to {save_path}\n"
                f"[Verifier] Best val F1={best_val_f1:.4f} "
                f"(checkpoint at {self.output_dir / 'best'})"
            )

        return {
            "history": metrics_history,
            "total_steps": global_step,
            "best_val_f1": best_val_f1,
            "decision_threshold": self.decision_threshold,
            "positive_class_weight": float(self.positive_class_weight or 1.0),
        }
