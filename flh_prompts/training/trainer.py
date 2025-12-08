"""Training loop for FLH-Prompts and baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from tqdm import tqdm

from flh_prompts.data.streaming import StreamingSST2
from flh_prompts.models.frozen_backbone import FrozenBERTWithPrompt
from flh_prompts.models.flh_pool import FLHPromptPool


@dataclass
class TrainConfig:
    """Training configuration."""
    # Dataset
    dataset: str = "sst2"
    backbone: str = "bert-base-uncased"

    # Prompt settings
    prompt_length: int = 20
    embed_dim: int = 768

    # FLH parameters
    alpha: float = 0.1
    birth_interval: int = 500

    # Data
    flip_interval: int = 1000
    batch_size: int = 32

    # Training
    total_steps: int = 10000
    lr: float = 0.001
    train_mode: Literal["weighted_only", "all_prompts"] = "weighted_only"

    # Wandb
    wandb_project: str = "flh-prompts"
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "cuda"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000


def train_step(
    model: FrozenBERTWithPrompt,
    pool: FLHPromptPool,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    train_mode: str = "weighted_only",
) -> tuple[float, float]:
    """Execute a single training step.

    Args:
        model: Frozen backbone model.
        pool: FLH prompt pool.
        batch: Batch dict with input_ids, attention_mask, labels.
        optimizer: Optimizer for prompt parameters.
        train_mode: "weighted_only" or "all_prompts".

    Returns:
        Tuple of (loss, accuracy).
    """
    device = batch["input_ids"].device

    if train_mode == "weighted_only":
        # Train only the weighted/selected prompt
        prompt = pool.get_prompt(mode="weighted_sum")
        logits = model(batch["input_ids"], batch["attention_mask"], prompt)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        loss_value = loss.item()

    else:  # all_prompts
        # Train all prompts in pool
        total_loss = 0.0
        for prompt in pool.prompts:
            logits = model(batch["input_ids"], batch["attention_mask"], prompt)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            total_loss += loss.item()
        loss_value = total_loss / len(pool.prompts)

    optimizer.step()
    optimizer.zero_grad()

    # Compute accuracy using weighted prompt
    with torch.no_grad():
        prompt = pool.get_prompt(mode="weighted_sum")
        logits = model(batch["input_ids"], batch["attention_mask"], prompt)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == batch["labels"]).float().mean().item()

    return loss_value, accuracy


def train_flh(config: TrainConfig) -> dict:
    """Train using FLH prompt pooling.

    Args:
        config: Training configuration.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"flh_{config.train_mode}",
        config=config.__dict__,
    )

    # Setup model and data
    print("Loading model...")
    model = FrozenBERTWithPrompt(
        model_name=config.backbone,
        num_labels=2,
        device=config.device,
    )

    print("Setting up data stream...")
    data_stream = StreamingSST2(
        flip_every_n=config.flip_interval,
        batch_size=config.batch_size,
        tokenizer_name=config.backbone,
    )

    # Initialize prompt pool with first prompt
    print("Initializing FLH prompt pool...")
    pool = FLHPromptPool(
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        alpha=config.alpha,
        device=config.device,
    )
    pool.birth_prompt()

    # Setup optimizer
    optimizer = torch.optim.AdamW(pool.get_parameters(), lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training results
    results = {
        "steps": [],
        "losses": [],
        "accuracies": [],
        "regimes": [],
        "weight_entropies": [],
        "num_prompts": [],
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

            # Move batch to device
            batch = {
                "input_ids": batch["input_ids"].to(config.device),
                "attention_mask": batch["attention_mask"].to(config.device),
                "labels": batch["labels"].to(config.device),
                "step": batch["step"],
                "regime": batch["regime"],
                "flipped": batch["flipped"],
            }

            # Birth new prompt at intervals
            if step > 0 and step % config.birth_interval == 0:
                new_prompt = pool.birth_prompt()
                # Add new prompt to optimizer
                optimizer.add_param_group({"params": [new_prompt]})

            # FLH weight update
            losses_all = pool.get_all_losses(model, batch)
            pool.update_weights(losses_all)

            # Training step
            loss, accuracy = train_step(model, pool, batch, optimizer, config.train_mode)

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss)
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch["regime"])
            results["weight_entropies"].append(pool.get_entropy())
            results["num_prompts"].append(pool.num_prompts())

            # Log to wandb
            wandb.log({
                "step": step,
                "loss": loss,
                "accuracy": accuracy,
                "regime": batch["regime"],
                "flipped": batch["flipped"],
                "weight_entropy": pool.get_entropy(),
                "num_prompts": pool.num_prompts(),
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"flh_step{step}.pt"
                torch.save({
                    "step": step,
                    "pool_state": pool.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"flh_final.pt"
    torch.save({
        "step": config.total_steps,
        "pool_state": pool.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
    }, checkpoint_path)

    wandb.finish()
    print(f"\n✓ Training complete! Final checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    # Quick test with minimal steps
    config = TrainConfig(
        total_steps=100,
        birth_interval=50,
        flip_interval=50,
        checkpoint_interval=50,
        wandb_run_name="test_run",
    )
    results = train_flh(config)
    print(f"\nFinal accuracy: {results['accuracies'][-1]:.4f}")
