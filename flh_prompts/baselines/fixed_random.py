"""Fixed random baseline - K prompts at init, random selection per batch."""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from flh_prompts.data.streaming import StreamingSST2, StreamingAmazonMultiDomain
from flh_prompts.models.frozen_backbone import FrozenBERTWithPrompt
from flh_prompts.training.trainer import TrainConfig


def train_fixed_random(config: TrainConfig, num_prompts: int = 10) -> dict:
    """Train using fixed pool with random selection.

    This baseline initializes K prompts at the start and randomly
    selects one to train on each batch. No weight updates.

    Args:
        config: Training configuration.
        num_prompts: Number of prompts to initialize.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or "fixed_random",
        config={**config.__dict__, "num_prompts": num_prompts},
    )

    # Setup model
    print("Loading model...")
    model = FrozenBERTWithPrompt(
        model_name=config.backbone,
        num_labels=2,
        device=config.device,
    )

    # Setup data
    print("Setting up data stream...")
    if config.dataset == "amazon":
        data_stream = StreamingAmazonMultiDomain(
            steps_per_domain=config.steps_per_domain,
            batch_size=config.batch_size,
            tokenizer_name=config.backbone,
            domains=config.amazon_domains,
            balance_labels=config.balance_labels,
        )
    else:  # sst2
        data_stream = StreamingSST2(
            flip_every_n=config.flip_interval,
            batch_size=config.batch_size,
            tokenizer_name=config.backbone,
        )

    # Fixed pool of prompts
    print(f"Initializing {num_prompts} prompts...")
    prompts = [
        nn.Parameter(
            torch.randn(config.prompt_length, config.embed_dim, device=config.device) * 0.02
        )
        for _ in range(num_prompts)
    ]

    # Optimizer for all prompts
    optimizer = torch.optim.AdamW(prompts, lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Results
    results = {
        "steps": [],
        "losses": [],
        "accuracies": [],
        "regimes": [],
        "selected_prompts": [],
        "domains": [],  # For Amazon dataset
        "domain_idxs": [],  # For Amazon dataset
    }

    # Training loop
    print(f"\nStarting training for {config.total_steps} steps...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Training", total=config.total_steps)

        for batch in data_stream:
            step = batch["step"]

            if step >= config.total_steps:
                break

            # Move to device - common fields
            batch_device = {
                "input_ids": batch["input_ids"].to(config.device),
                "attention_mask": batch["attention_mask"].to(config.device),
                "labels": batch["labels"].to(config.device),
                "step": batch["step"],
                "regime": batch["regime"],
            }

            # Add dataset-specific fields
            if config.dataset == "amazon":
                batch_device["domain"] = batch["domain"]
                batch_device["domain_idx"] = batch["domain_idx"]
            else:
                batch_device["flipped"] = batch["flipped"]

            # Randomly select a prompt
            selected_idx = random.randint(0, num_prompts - 1)
            prompt = prompts[selected_idx]

            # Forward pass
            logits = model(batch_device["input_ids"], batch_device["attention_mask"], prompt)
            loss = F.cross_entropy(logits, batch_device["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy (using randomly selected prompt)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == batch_device["labels"]).float().mean().item()

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss.item())
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch_device["regime"])
            results["selected_prompts"].append(selected_idx)

            # Dataset-specific results
            if config.dataset == "amazon":
                results["domains"].append(batch_device["domain"])
                results["domain_idxs"].append(batch_device["domain_idx"])

            # Log to wandb
            log_dict = {
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "regime": batch_device["regime"],
                "selected_prompt": selected_idx,
            }

            if config.dataset == "amazon":
                log_dict["domain"] = batch_device["domain"]
                log_dict["domain_idx"] = batch_device["domain_idx"]
            else:
                log_dict["flipped"] = batch_device["flipped"]

            wandb.log(log_dict)

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"random_step{step}.pt"
                torch.save({
                    "step": step,
                    "prompts": [p.data.clone() for p in prompts],
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"random_final.pt"
    torch.save({
        "step": config.total_steps,
        "prompts": [p.data.clone() for p in prompts],
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
    }, checkpoint_path)

    wandb.finish()
    print(f"\n✓ Training complete! Final checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    config = TrainConfig(
        total_steps=100,
        flip_interval=50,
        checkpoint_interval=50,
        wandb_run_name="random_test",
    )
    results = train_fixed_random(config, num_prompts=5)
    print(f"\nFinal accuracy: {results['accuracies'][-1]:.4f}")
