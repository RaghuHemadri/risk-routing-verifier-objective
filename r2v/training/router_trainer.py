"""
Router Trainer: risk-calibrated cost-robustness optimization.

Trains the router r_ψ using the Lagrangian formulation:
  min_ψ max_λ  E[cost(d)] + λ(CVaR_α(1 - S_z) - ε)

The router learns to fall back to the LLM teacher when:
1. The verifier score is low (SLM action likely wrong)
2. The policy entropy is high (SLM is uncertain)
3. The combined risk exceeds the learned threshold

Calibration is enforced via Brier score and post-hoc temperature scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from r2v.models.router import Router, RouterLoss, TemperatureScaling

logger = logging.getLogger(__name__)


class RouterTrainer:
    """Trainer for the risk-calibrated router."""

    def __init__(
        self,
        router: Router,
        train_dataset,
        eval_dataset=None,
        config: dict[str, Any] = None,
        output_dir: str = "experiments/checkpoints/router",
    ):
        self.router = router
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = self.config.get("epochs", 20)
        self.batch_size = self.config.get("batch_size", 64)
        self.lr = self.config.get("learning_rate", 1e-3)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.lagrangian_lr = self.config.get("lagrangian_lr", 0.01)

        self.loss_fn = RouterLoss(self.config)
        self.temp_scaling = TemperatureScaling(
            num_bins=self.config.get("num_bins", 15)
        )

    def train(self) -> dict[str, Any]:
        """Run router training with Lagrangian dual optimization."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router = self.router.to(device)
        self.router.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=2,
            )

        # Separate optimizers for primal (router params) and dual (λ)
        router_params = [
            p for name, p in self.router.named_parameters()
            if p.requires_grad and name != "log_lambda"
        ]
        primal_optimizer = torch.optim.AdamW(
            router_params, lr=self.lr, weight_decay=self.weight_decay
        )
        dual_optimizer = torch.optim.Adam(
            [self.router.log_lambda], lr=self.lagrangian_lr
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            primal_optimizer, T_max=self.epochs
        )

        metrics_history = []
        best_eval_metric = float("inf")

        for epoch in range(self.epochs):
            epoch_metrics = {}
            num_batches = 0

            for batch in train_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                success = batch["success"].to(device)
                seeds = batch["perturbation_seed"].to(device)

                # Forward pass
                fallback_probs = self.router(features)

                # Compute loss
                losses = self.loss_fn(
                    fallback_probs, labels, success, seeds,
                    self.router.lagrange_multiplier,
                )

                # Primal step (minimize router loss)
                primal_optimizer.zero_grad()
                losses["total_loss"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(router_params, 1.0)
                primal_optimizer.step()

                # Dual step (maximize λ * constraint violation)
                dual_optimizer.zero_grad()
                losses["dual_loss"].backward()
                dual_optimizer.step()

                # Accumulate metrics
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        epoch_metrics[k] = epoch_metrics.get(k, 0) + v.item()
                num_batches += 1

            scheduler.step()

            # Epoch summary
            avg_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = epoch + 1
            avg_metrics["lambda"] = self.router.lagrange_multiplier.item()
            avg_metrics["temperature"] = self.router.temperature.item()
            metrics_history.append(avg_metrics)

            logger.info(
                f"[Router] Epoch {epoch+1}/{self.epochs} "
                f"Cost={avg_metrics.get('cost_loss', 0):.4f} "
                f"Robust={avg_metrics.get('robustness_loss', 0):.4f} "
                f"Calib={avg_metrics.get('calibration_loss', 0):.4f} "
                f"λ={avg_metrics['lambda']:.4f} "
                f"T={avg_metrics['temperature']:.4f}"
            )

            # Evaluation
            if eval_loader and (epoch + 1) % 5 == 0:
                eval_metrics = self._evaluate(eval_loader, device)
                logger.info(
                    f"[Router] Eval: ECE={eval_metrics['ece']:.4f} "
                    f"Brier={eval_metrics['brier']:.4f} "
                    f"Acc={eval_metrics['accuracy']:.4f}"
                )
                if eval_metrics["brier"] < best_eval_metric:
                    best_eval_metric = eval_metrics["brier"]
                    self._save("best")

        # Post-hoc temperature scaling on eval set
        if eval_loader:
            self._calibrate(eval_loader, device)

        self._save("final")

        return {
            "history": metrics_history,
            "best_eval_brier": best_eval_metric,
        }

    @torch.no_grad()
    def _evaluate(self, eval_loader, device) -> dict[str, float]:
        """Evaluate router on held-out data."""
        self.router.eval()
        all_probs = []
        all_labels = []

        for batch in eval_loader:
            features = batch["features"].to(device)
            probs = self.router(features)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        brier = np.mean((all_probs - all_labels) ** 2)
        ece = self.temp_scaling.compute_ece(all_probs, all_labels)
        accuracy = np.mean((all_probs > 0.5) == all_labels)

        self.router.train()
        return {"brier": brier, "ece": ece, "accuracy": accuracy}

    def _calibrate(self, eval_loader, device):
        """Apply post-hoc temperature scaling."""
        self.router.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                features = batch["features"].to(device)
                logits = self.router.mlp(features).squeeze(-1)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["label"].numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        self.temp_scaling.fit(all_logits, all_labels)

        # Update router temperature
        with torch.no_grad():
            self.router.temperature.fill_(self.temp_scaling.temperature)

        logger.info(f"[Router] Post-hoc calibration temperature: {self.temp_scaling.temperature:.4f}")

    def _save(self, name: str):
        """Save router checkpoint."""
        save_path = self.output_dir / f"{name}.pt"
        torch.save({
            "router_state_dict": self.router.state_dict(),
            "temperature": self.temp_scaling.temperature,
        }, save_path)
        logger.info(f"[Router] Saved to {save_path}")
