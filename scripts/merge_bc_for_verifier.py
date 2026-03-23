#!/usr/bin/env python3
"""
Merge a BC-trained Accelerate/PEFT checkpoint into a standalone HuggingFace
model directory that can be loaded by AutoModelForCausalLM.from_pretrained().

Usage:
    python scripts/merge_bc_for_verifier.py \
        --hf-model-id Qwen/Qwen2.5-Coder-7B-Instruct \
        --bc-checkpoint outputs/policy/humaneval_noisy/bc_qwen_coder/best \
        --output outputs/merged/qwen_coder_7b \
        --lora-r 32 --lora-alpha 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HF_TOKEN = os.environ.get("HF_TOKEN", None)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge BC checkpoint into a standalone HF model directory"
    )
    parser.add_argument("--hf-model-id", type=str, required=True,
                        help="Original HuggingFace model ID (for config/tokenizer/architecture)")
    parser.add_argument("--bc-checkpoint", type=str, required=True,
                        help="Path to BC checkpoint dir (e.g. .../best) containing model.safetensors")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the merged HF model")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.10)
    parser.add_argument("--target-modules", nargs="*",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bc_path = Path(args.bc_checkpoint)
    output_path = Path(args.output)

    safetensors_file = bc_path / "model.safetensors"
    if not safetensors_file.exists():
        logger.error(f"No model.safetensors found in {bc_path}")
        sys.exit(1)

    if output_path.exists() and (output_path / "config.json").exists():
        logger.info(f"Merged model already exists at {output_path} — skipping")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    # 1. Load the state dict from BC checkpoint
    logger.info(f"Loading BC state dict from {safetensors_file}")
    bc_state = load_file(str(safetensors_file))

    sample_key = next(iter(bc_state))
    has_peft_prefix = sample_key.startswith("base_model.model.")
    logger.info(f"State dict sample key: {sample_key!r} (PEFT prefix: {has_peft_prefix})")

    if has_peft_prefix:
        # PEFT/Accelerate state dict — need to load base model, apply LoRA,
        # restore weights, then merge.
        logger.info(f"Loading base model: {args.hf_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=_HF_TOKEN,
        )

        from peft import LoraConfig, get_peft_model, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        logger.info(f"Applying LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
        peft_model = get_peft_model(base_model, peft_config)

        # Load the BC-trained state dict into the PEFT model
        logger.info("Loading BC weights into PEFT model")
        missing, unexpected = peft_model.load_state_dict(bc_state, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # Merge LoRA into base weights
        logger.info("Merging LoRA adapters into base model")
        merged_model = peft_model.merge_and_unload()
    else:
        # Plain state dict without PEFT prefix — load directly
        logger.info(f"Loading base model: {args.hf_model_id}")
        merged_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=_HF_TOKEN,
        )
        logger.info("Loading BC weights directly (no PEFT prefix)")
        missing, unexpected = merged_model.load_state_dict(bc_state, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Save the complete merged model
    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer from the original HF model
    logger.info("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_id, trust_remote_code=True, token=_HF_TOKEN
    )
    tokenizer.save_pretrained(output_path)

    logger.info(f"Done — merged model saved to {output_path}")


if __name__ == "__main__":
    main()
