"""Test script to analyze expert selection in FLH with alpha=0.1."""

import sys
import torch
import torch.nn.functional as F
from collections import Counter
import numpy as np
import wandb

from flh_prompts.data.streaming import StreamingSST2
from flh_prompts.models.frozen_backbone import FrozenBERTWithPrompt
from flh_prompts.models.flh_pool import FLHPromptPool


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def run_expert_analysis(
    total_steps: int = 2000,
    alpha: float = 0.1,
    birth_interval: int = 500,
    flip_interval: int = 1000,
):
    """Run FLH and track expert selection."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="flh-prompts",
        name=f"expert_analysis_alpha{alpha}",
        config={
            "total_steps": total_steps,
            "alpha": alpha,
            "birth_interval": birth_interval,
            "flip_interval": flip_interval,
        },
    )

    # Setup
    log("Loading model...")
    model = FrozenBERTWithPrompt(
        model_name="bert-base-uncased",
        num_labels=2,
        device=device,
    )

    log("Setting up data stream...")
    data_stream = StreamingSST2(
        flip_every_n=flip_interval,
        batch_size=32,
        tokenizer_name="bert-base-uncased",
    )

    log(f"Initializing FLH pool with alpha={alpha}...")
    pool = FLHPromptPool(
        prompt_length=20,
        embed_dim=768,
        alpha=alpha,
        device=device,
    )
    pool.birth_prompt()

    optimizer = torch.optim.AdamW(pool.get_parameters(), lr=0.001)

    # Tracking
    expert_selections = []  # Which expert had highest weight each step
    weight_history = []     # Full weight distribution each step
    regime_history = []     # Regime at each step
    accuracy_history = []

    log(f"\nStarting {total_steps} step training run...")
    log("=" * 80)

    for batch in data_stream:
        step = batch["step"]
        if step >= total_steps:
            break

        # Move to device
        batch_gpu = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }

        # Birth new prompt at intervals
        if step > 0 and step % birth_interval == 0:
            new_prompt = pool.birth_prompt()
            optimizer.add_param_group({"params": [new_prompt]})
            log(f"\n[Step {step}] BIRTH: New expert #{pool.num_prompts()-1} created")
            log(f"           Weights after birth: {format_weights(pool.weights)}")

        # FLH weight update
        losses = pool.get_all_losses(model, batch_gpu)
        pool.update_weights(losses)

        # Track selection
        weights = pool.get_weight_distribution()
        top_expert = np.argmax(weights)
        expert_selections.append(top_expert)
        weight_history.append(weights.copy())
        regime_history.append(batch["regime"])

        # Train step
        prompt = pool.get_prompt(mode="weighted_sum")
        logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"], prompt)
        loss = F.cross_entropy(logits, batch_gpu["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == batch_gpu["labels"]).float().mean().item()
            accuracy_history.append(acc)

        # Log to wandb
        log_dict = {
            "step": step,
            "accuracy": acc,
            "loss": loss.item(),
            "regime": batch["regime"],
            "top_expert": int(top_expert),
            "num_experts": pool.num_prompts(),
            "weight_entropy": pool.get_entropy(),
        }
        # Log per-expert weights
        for i, w in enumerate(weights):
            log_dict[f"weight_{i}"] = w
        # Log per-expert losses
        for i, l in enumerate(losses):
            log_dict[f"loss_{i}"] = l
        wandb.log(log_dict)

        # Print detailed info at key points
        if step % 100 == 0 or step == flip_interval - 1 or step == flip_interval or step == flip_interval + 1:
            log(f"\n[Step {step:4d}] Regime={batch['regime']} | Top expert: #{top_expert} | Acc: {acc:.2%}")
            log(f"           Weights: {format_weights(weights)}")
            log(f"           Losses:  {format_losses(losses)}")

    log("\n" + "=" * 80)
    log("ANALYSIS COMPLETE")
    log("=" * 80)

    # Summary statistics
    log("\n--- Expert Selection Summary ---")
    selection_counts = Counter(expert_selections)
    total = len(expert_selections)
    log(f"Total steps: {total}")
    log(f"Number of experts: {pool.num_prompts()}")
    log("\nExpert activation counts:")
    for expert_id in range(pool.num_prompts()):
        count = selection_counts.get(expert_id, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        log(f"  Expert #{expert_id}: {count:4d} ({pct:5.1f}%) {bar}")

    # Per-regime analysis
    log("\n--- Expert Selection by Regime ---")
    regimes = sorted(set(regime_history))
    for regime in regimes:
        regime_indices = [i for i, r in enumerate(regime_history) if r == regime]
        regime_selections = [expert_selections[i] for i in regime_indices]
        regime_counts = Counter(regime_selections)
        log(f"\nRegime {regime} ({len(regime_indices)} steps):")
        for expert_id in range(pool.num_prompts()):
            count = regime_counts.get(expert_id, 0)
            pct = count / len(regime_indices) * 100 if regime_indices else 0
            bar = "#" * int(pct / 2)
            log(f"  Expert #{expert_id}: {count:4d} ({pct:5.1f}%) {bar}")

    # Weight distribution over time
    log("\n--- Weight Distribution at Key Points ---")
    key_steps = [0, 499, 500, 999, 1000, 1001, 1499, 1500, 1999]
    for s in key_steps:
        if s < len(weight_history):
            log(f"Step {s:4d} (regime {regime_history[s]}): {format_weights(weight_history[s])}")

    # Final weights
    log(f"\n--- Final State ---")
    log(f"Final weights: {format_weights(pool.weights)}")
    log(f"Final accuracy (last 100 steps): {np.mean(accuracy_history[-100:]):.2%}")

    wandb.finish()

    return {
        "expert_selections": expert_selections,
        "weight_history": weight_history,
        "regime_history": regime_history,
        "accuracy_history": accuracy_history,
    }


def format_weights(weights):
    """Format weights for display."""
    return "[" + ", ".join(f"{w:.3f}" for w in weights) + "]"


def format_losses(losses):
    """Format losses for display."""
    return "[" + ", ".join(f"{l:.3f}" for l in losses) + "]"


if __name__ == "__main__":
    results = run_expert_analysis(
        total_steps=2000,
        alpha=0.1,
        birth_interval=500,
        flip_interval=1000,
    )
