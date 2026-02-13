"""
PyTorch datasets for R2V-Agent training.

Provides:
- BCDataset: behavior cloning from teacher trajectories
- PreferenceDataset: DPO pairs from verifier-scored candidates
- ConsistencyDataset: paired observations under different tool seeds
- VerifierDataset: (context, action, label) for verifier training
- RouterDataset: features + routing labels for router training
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from r2v.data.trajectory import (
    Episode, Step, CandidateActions, TrajectoryStore, PerturbationType
)


def build_context_string(goal: str, steps: list[Step], current_step_idx: int) -> str:
    """Build the agent context x_t = (G, o_<=t, a_<t, y_<t).

    Constructs a linearized context string from the goal and trajectory history.
    """
    parts = [f"Goal: {goal}\n"]
    for i in range(current_step_idx + 1):
        s = steps[i]
        parts.append(f"--- Step {i + 1} ---")
        parts.append(f"Observation: {s.observation.raw_text}")
        if s.observation.url:
            parts.append(f"URL: {s.observation.url}")
        if i < current_step_idx:
            parts.append(f"Action: {s.action.raw_text}")
            if s.action.plan_tag:
                parts.append(f"Plan: {s.action.plan_tag}")
    return "\n".join(parts)


# ============================================================
# Behavior Cloning Dataset
# ============================================================

@dataclass
class BCExample:
    context: str
    target_action: str
    episode_id: str
    step_idx: int


class BCDataset(Dataset):
    """Dataset for behavior cloning on teacher trajectories."""

    def __init__(
        self,
        trajectory_path: str,
        tokenizer,
        max_seq_len: int = 4096,
        max_episodes: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[BCExample] = []

        store = TrajectoryStore(trajectory_path)
        episodes = store.load_episodes(max_count=max_episodes)

        for ep in episodes:
            if not ep.success:
                continue  # Only learn from successful teacher trajectories
            for i, step in enumerate(ep.steps):
                ctx = build_context_string(ep.metadata.goal, ep.steps, i)
                self.examples.append(BCExample(
                    context=ctx,
                    target_action=step.action.raw_text,
                    episode_id=ep.episode_id,
                    step_idx=i,
                ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        # Format as instruction-following: context → action
        prompt = f"{ex.context}\nAction:"
        full_text = f"{prompt} {ex.target_action}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels: mask the prompt tokens with -100
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = encoding["input_ids"].clone().squeeze(0)
        labels[:prompt_len] = -100  # Don't compute loss on prompt

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ============================================================
# Preference Dataset (DPO)
# ============================================================

@dataclass
class PreferenceExample:
    context: str
    chosen_action: str        # a+ (verifier-preferred)
    rejected_action: str      # a- (verifier-rejected)
    chosen_score: float
    rejected_score: float


class PreferenceDataset(Dataset):
    """Dataset for DPO-style preference distillation using verifier scores."""

    def __init__(
        self,
        candidates_path: str,
        tokenizer,
        max_seq_len: int = 4096,
        min_score_gap: float = 0.1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[PreferenceExample] = []

        # Load candidate action data
        import jsonlines
        with jsonlines.open(candidates_path, mode="r") as reader:
            for obj in reader:
                best_idx = max(range(len(obj["verifier_scores"])),
                               key=lambda i: obj["verifier_scores"][i])
                worst_idx = min(range(len(obj["verifier_scores"])),
                                key=lambda i: obj["verifier_scores"][i])
                score_gap = obj["verifier_scores"][best_idx] - obj["verifier_scores"][worst_idx]
                if score_gap < min_score_gap:
                    continue
                self.examples.append(PreferenceExample(
                    context=obj["context"],
                    chosen_action=obj["candidates"][best_idx],
                    rejected_action=obj["candidates"][worst_idx],
                    chosen_score=obj["verifier_scores"][best_idx],
                    rejected_score=obj["verifier_scores"][worst_idx],
                ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt = f"{ex.context}\nAction:"

        chosen_text = f"{prompt} {ex.chosen_action}"
        rejected_text = f"{prompt} {ex.rejected_action}"

        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        prompt_enc = self.tokenizer(
            prompt, max_length=self.max_seq_len,
            truncation=True, return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        chosen_labels = chosen_enc["input_ids"].clone().squeeze(0)
        chosen_labels[:prompt_len] = -100
        rejected_labels = rejected_enc["input_ids"].clone().squeeze(0)
        rejected_labels[:prompt_len] = -100

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels,
        }


# ============================================================
# Consistency Dataset
# ============================================================

@dataclass
class ConsistencyExample:
    context_a: str   # Context with tool output under seed z
    context_b: str   # Context with tool output under seed z'
    episode_id: str
    step_idx: int


class ConsistencyDataset(Dataset):
    """Paired observations under different tool seeds for KL regularization."""

    def __init__(
        self,
        paired_path: str,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[ConsistencyExample] = []

        import jsonlines
        with jsonlines.open(paired_path, mode="r") as reader:
            for obj in reader:
                self.examples.append(ConsistencyExample(**obj))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt_a = f"{ex.context_a}\nAction:"
        prompt_b = f"{ex.context_b}\nAction:"

        enc_a = self.tokenizer(
            prompt_a, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        enc_b = self.tokenizer(
            prompt_b, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
        }


# ============================================================
# Verifier Dataset
# ============================================================

class VerifierDataset(Dataset):
    """Dataset for training the step/outcome verifier V_φ."""

    def __init__(
        self,
        trajectory_path: str,
        tokenizer,
        max_seq_len: int = 4096,
        use_step_labels: bool = True,
        max_episodes: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []

        store = TrajectoryStore(trajectory_path)
        episodes = store.load_episodes(max_count=max_episodes)

        for ep in episodes:
            for i, step in enumerate(ep.steps):
                ctx = build_context_string(ep.metadata.goal, ep.steps, i)
                text = f"{ctx}\nAction: {step.action.raw_text}"

                # Final outcome label (always available)
                entry = {
                    "text": text,
                    "final_label": float(ep.success),
                    "episode_id": ep.episode_id,
                    "step_idx": i,
                    "perturbation_type": ep.perturbation_type.value,
                }

                # Step-level label (if available)
                if use_step_labels and step.label is not None:
                    entry["step_label"] = float(
                        step.label.is_correct if step.label.is_correct is not None
                        else step.label.is_progress if step.label.is_progress is not None
                        else ep.success
                    )
                else:
                    entry["step_label"] = None

                self.examples.append(entry)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"], max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "final_label": torch.tensor(ex["final_label"], dtype=torch.float32),
        }
        if ex["step_label"] is not None:
            item["step_label"] = torch.tensor(ex["step_label"], dtype=torch.float32)
            item["has_step_label"] = torch.tensor(1.0)
        else:
            item["step_label"] = torch.tensor(0.0)
            item["has_step_label"] = torch.tensor(0.0)

        return item


# ============================================================
# Router Dataset
# ============================================================

@dataclass
class RouterExample:
    """Feature vector + label for router training."""
    features: list[float]   # [verifier_score, entropy, step_pct, token_pct, ...]
    label: float            # 1.0 = should fallback to LLM
    success: float          # Episode success (for CVaR computation)
    perturbation_seed: int
    cost: float


class RouterDataset(Dataset):
    """Dataset for training the risk-calibrated router."""

    def __init__(self, examples: list[RouterExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "features": torch.tensor(ex.features, dtype=torch.float32),
            "label": torch.tensor(ex.label, dtype=torch.float32),
            "success": torch.tensor(ex.success, dtype=torch.float32),
            "perturbation_seed": torch.tensor(ex.perturbation_seed, dtype=torch.long),
            "cost": torch.tensor(ex.cost, dtype=torch.float32),
        }

    @classmethod
    def from_episodes(
        cls,
        episodes: list[Episode],
        verifier_scores: dict[str, list[float]],
        entropies: dict[str, list[float]],
        cost_slm: float = 1.0,
        cost_llm: float = 50.0,
        risk_threshold: float = 0.5,
    ) -> "RouterDataset":
        """Build router dataset from evaluated episodes with verifier scores."""
        examples = []
        for ep in episodes:
            ep_scores = verifier_scores.get(ep.episode_id, [])
            ep_entropies = entropies.get(ep.episode_id, [])

            for i, step in enumerate(ep.steps):
                v_score = ep_scores[i] if i < len(ep_scores) else 0.5
                entropy = ep_entropies[i] if i < len(ep_entropies) else 1.0
                step_pct = (i + 1) / max(len(ep.steps), 1)
                risk = (1.0 - v_score) * 0.7 + entropy * 0.3  # risk estimate ρ(x_t)

                features = [v_score, entropy, step_pct, risk]
                should_fallback = float(risk > risk_threshold)
                cost = cost_llm if should_fallback else cost_slm

                examples.append(RouterExample(
                    features=features,
                    label=should_fallback,
                    success=float(ep.success),
                    perturbation_seed=ep.perturbation_seed or 0,
                    cost=cost,
                ))
        return cls(examples)
