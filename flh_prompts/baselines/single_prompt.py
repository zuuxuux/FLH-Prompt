"""Single prompt baseline - one prompt, standard training."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from flh_prompts.data.streaming import StreamingSST2
from flh_prompts.models.frozen_backbone import FrozenBERTWithPrompt
from flh_prompts.training.trainer import TrainConfig


def train_single_prompt(config: TrainConfig) -> dict:
    """Train using a single prompt (no FLH).

    This baseline uses one prompt that is trained throughout,
    with no adaptation to distribution shifts.

    Args:
        config: Training configuration.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or "single_prompt",
        config=config.__dict__,
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
    data_stream = StreamingSST2(
        flip_every_n=config.flip_interval,
        batch_size=config.batch_size,
        tokenizer_name=config.backbone,
    )

    # Single prompt
    print("Initializing single prompt...")
    prompt = nn.Parameter(
        torch.randn(config.prompt_length, config.embed_dim, device=config.device) * 0.02
    )

    # Optimizer
    optimizer = torch.optim.AdamW([prompt], lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Results
    results = {
        "steps": [],
        "losses": [],
        "accuracies": [],
        "regimes": [],
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

            # Move to device
            batch = {
                "input_ids": batch["input_ids"].to(config.device),
                "attention_mask": batch["attention_mask"].to(config.device),
                "labels": batch["labels"].to(config.device),
                "step": batch["step"],
                "regime": batch["regime"],
                "flipped": batch["flipped"],
            }

            # Forward pass
            logits = model(batch["input_ids"], batch["attention_mask"], prompt)
            loss = F.cross_entropy(logits, batch["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == batch["labels"]).float().mean().item()

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss.item())
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch["regime"])

            # Log to wandb
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "regime": batch["regime"],
                "flipped": batch["flipped"],
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"single_step{step}.pt"
                torch.save({
                    "step": step,
                    "prompt": prompt.data.clone(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"single_final.pt"
    torch.save({
        "step": config.total_steps,
        "prompt": prompt.data.clone(),
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
        wandb_run_name="single_test",
    )
    results = train_single_prompt(config)
    print(f"\nFinal accuracy: {results['accuracies'][-1]:.4f}")
