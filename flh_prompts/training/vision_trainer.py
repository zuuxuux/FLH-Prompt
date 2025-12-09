"""Training loop for FLH-Prompts with vision backbone (ViT + CIFAR-10)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from flh_prompts.data.streaming_vision import StreamingCIFAR10, StreamingCIFAR10Binary
from flh_prompts.models.frozen_vit import FrozenViTWithPrompt
from flh_prompts.models.flh_pool import FLHPromptPool


@dataclass
class VisionTrainConfig:
    """Vision training configuration."""
    # Dataset
    dataset: Literal["cifar10", "cifar10_binary"] = "cifar10"
    num_classes: int = 10  # 10 for full CIFAR-10, 2 for binary

    # Backbone
    backbone: str = "google/vit-base-patch16-224"
    image_size: int = 224

    # Prompt settings
    prompt_length: int = 20
    embed_dim: int = 768  # ViT-Base hidden size

    # FLH parameters
    alpha: float = 0.1
    birth_interval: int = 500

    # Data - label rotation settings
    rotate_interval: int = 1000  # Steps between label rotations
    batch_size: int = 32
    rotation_type: str = "shift"  # "shift" or "swap_pairs"

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


def train_step_vision(
    model: FrozenViTWithPrompt,
    pool: FLHPromptPool,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    train_mode: str = "weighted_only",
) -> tuple[float, float]:
    """Execute a single training step for vision.

    Args:
        model: Frozen ViT backbone model.
        pool: FLH prompt pool.
        batch: Batch dict with pixel_values, labels.
        optimizer: Optimizer for prompt parameters.
        train_mode: "weighted_only" or "all_prompts".

    Returns:
        Tuple of (loss, accuracy).
    """
    if train_mode == "weighted_only":
        # Train only the weighted/selected prompt
        prompt = pool.get_prompt(mode="weighted_sum")
        logits = model(batch["pixel_values"], prompt)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        loss_value = loss.item()

    else:  # all_prompts
        # Train all prompts in pool
        total_loss = 0.0
        for prompt in pool.prompts:
            logits = model(batch["pixel_values"], prompt)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            total_loss += loss.item()
        loss_value = total_loss / len(pool.prompts)

    optimizer.step()
    optimizer.zero_grad()

    # Compute accuracy using weighted prompt
    with torch.no_grad():
        prompt = pool.get_prompt(mode="weighted_sum")
        logits = model(batch["pixel_values"], prompt)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == batch["labels"]).float().mean().item()

    return loss_value, accuracy


def get_all_losses_vision(
    model: FrozenViTWithPrompt,
    pool: FLHPromptPool,
    batch: dict,
) -> list[float]:
    """Compute loss for each prompt in the pool (for FLH weight update).

    Args:
        model: Frozen ViT backbone.
        pool: FLH prompt pool.
        batch: Batch dict with pixel_values, labels.

    Returns:
        List of loss values for each prompt.
    """
    losses = []
    with torch.no_grad():
        for prompt in pool.prompts:
            logits = model(batch["pixel_values"], prompt)
            loss = F.cross_entropy(logits, batch["labels"])
            losses.append(loss.item())
    return losses


def train_flh_vision(config: VisionTrainConfig) -> dict:
    """Train using FLH prompt pooling with vision backbone.

    Args:
        config: Training configuration.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"flh_vision_{config.dataset}",
        config=config.__dict__,
    )

    # Setup model
    print("Loading ViT model...")
    model = FrozenViTWithPrompt(
        model_name=config.backbone,
        num_labels=config.num_classes,
        device=config.device,
    )

    # Setup data
    print("Setting up CIFAR-10 data stream...")
    if config.dataset == "cifar10_binary":
        data_stream = StreamingCIFAR10Binary(
            flip_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
    else:  # cifar10
        data_stream = StreamingCIFAR10(
            rotate_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            rotation_type=config.rotation_type,
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
            batch_device = {
                "pixel_values": batch["pixel_values"].to(config.device),
                "labels": batch["labels"].to(config.device),
                "step": batch["step"],
                "regime": batch["regime"],
            }

            # Birth new prompt at intervals
            if step > 0 and step % config.birth_interval == 0:
                new_prompt = pool.birth_prompt()
                optimizer.add_param_group({"params": [new_prompt]})

            # FLH weight update
            losses_all = get_all_losses_vision(model, pool, batch_device)
            pool.update_weights(losses_all)

            # Training step
            loss, accuracy = train_step_vision(
                model, pool, batch_device, optimizer, config.train_mode
            )

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss)
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch_device["regime"])
            results["weight_entropies"].append(pool.get_entropy())
            results["num_prompts"].append(pool.num_prompts())

            # Log to wandb
            log_dict = {
                "step": step,
                "loss": loss,
                "accuracy": accuracy,
                "regime": batch_device["regime"],
                "weight_entropy": pool.get_entropy(),
                "num_prompts": pool.num_prompts(),
            }
            wandb.log(log_dict)

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"flh_vision_step{step}.pt"
                torch.save({
                    "step": step,
                    "pool_state": pool.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"flh_vision_final.pt"
    torch.save({
        "step": config.total_steps,
        "pool_state": pool.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
    }, checkpoint_path)

    wandb.finish()
    print(f"\n Training complete! Final checkpoint: {checkpoint_path}")

    return results


def train_single_prompt_vision(config: VisionTrainConfig) -> dict:
    """Train with a single prompt (baseline) on vision task.

    Args:
        config: Training configuration.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"single_vision_{config.dataset}",
        config=config.__dict__,
    )

    # Setup model
    print("Loading ViT model...")
    model = FrozenViTWithPrompt(
        model_name=config.backbone,
        num_labels=config.num_classes,
        device=config.device,
    )

    # Setup data
    print("Setting up CIFAR-10 data stream...")
    if config.dataset == "cifar10_binary":
        data_stream = StreamingCIFAR10Binary(
            flip_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
    else:
        data_stream = StreamingCIFAR10(
            rotate_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            rotation_type=config.rotation_type,
        )

    # Single learnable prompt
    print("Initializing single prompt...")
    prompt = nn.Parameter(
        torch.randn(config.prompt_length, config.embed_dim, device=config.device) * 0.02
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW([prompt], lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training results
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

            # Move batch to device
            batch_device = {
                "pixel_values": batch["pixel_values"].to(config.device),
                "labels": batch["labels"].to(config.device),
            }

            # Forward pass
            logits = model(batch_device["pixel_values"], prompt)
            loss = F.cross_entropy(logits, batch_device["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == batch_device["labels"]).float().mean().item()

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
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"single_vision_step{step}.pt"
                torch.save({
                    "step": step,
                    "prompt": prompt.data,
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"single_vision_final.pt"
    torch.save({
        "step": config.total_steps,
        "prompt": prompt.data,
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
    }, checkpoint_path)

    wandb.finish()
    print(f"\n Training complete! Final checkpoint: {checkpoint_path}")

    return results


def train_random_pool_vision(config: VisionTrainConfig, num_prompts: int = 10) -> dict:
    """Train with random prompt selection (baseline) on vision task.

    Args:
        config: Training configuration.
        num_prompts: Number of prompts in the pool.

    Returns:
        Dictionary of training results.
    """
    import random

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"random_vision_{config.dataset}",
        config={**config.__dict__, "num_prompts": num_prompts},
    )

    # Setup model
    print("Loading ViT model...")
    model = FrozenViTWithPrompt(
        model_name=config.backbone,
        num_labels=config.num_classes,
        device=config.device,
    )

    # Setup data
    print("Setting up CIFAR-10 data stream...")
    if config.dataset == "cifar10_binary":
        data_stream = StreamingCIFAR10Binary(
            flip_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
    else:
        data_stream = StreamingCIFAR10(
            rotate_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            rotation_type=config.rotation_type,
        )

    # Initialize prompt pool
    print(f"Initializing {num_prompts} random prompts...")
    prompts = nn.ParameterList([
        nn.Parameter(torch.randn(config.prompt_length, config.embed_dim, device=config.device) * 0.02)
        for _ in range(num_prompts)
    ])

    # Setup optimizer
    optimizer = torch.optim.AdamW(list(prompts), lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training results
    results = {
        "steps": [],
        "losses": [],
        "accuracies": [],
        "regimes": [],
        "selected_prompts": [],
    }

    # Set random seed for reproducibility
    random.seed(42)

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
            batch_device = {
                "pixel_values": batch["pixel_values"].to(config.device),
                "labels": batch["labels"].to(config.device),
            }

            # Random prompt selection
            selected_idx = random.randint(0, num_prompts - 1)
            prompt = prompts[selected_idx]

            # Forward pass
            logits = model(batch_device["pixel_values"], prompt)
            loss = F.cross_entropy(logits, batch_device["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == batch_device["labels"]).float().mean().item()

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss.item())
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch["regime"])
            results["selected_prompts"].append(selected_idx)

            # Log to wandb
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "regime": batch["regime"],
                "selected_prompt": selected_idx,
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"random_vision_step{step}.pt"
                torch.save({
                    "step": step,
                    "prompts_state": {i: p.data for i, p in enumerate(prompts)},
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"random_vision_final.pt"
    torch.save({
        "step": config.total_steps,
        "prompts_state": {i: p.data for i, p in enumerate(prompts)},
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
    }, checkpoint_path)

    wandb.finish()
    print(f"\n Training complete! Final checkpoint: {checkpoint_path}")

    return results


class VisionPromptPoolWithKeys(nn.Module):
    """Prompt pool with learnable keys for input-based selection (vision version).

    Each prompt has an associated key vector. Selection is done by
    computing cosine similarity between the input embedding (CLS token)
    and all keys, then selecting the prompt with highest similarity.
    """

    def __init__(
        self,
        num_prompts: int,
        prompt_length: int,
        embed_dim: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.device = device

        # Initialize prompts
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(prompt_length, embed_dim, device=device) * 0.02)
            for _ in range(num_prompts)
        ])

        # Keys will be initialized from data
        self.keys = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, device=device), requires_grad=False)
            for _ in range(num_prompts)
        ])
        self._keys_initialized = False

    def initialize_keys_from_data(
        self,
        model: FrozenViTWithPrompt,
        data_stream,
        num_samples: int = 100,
    ) -> None:
        """Initialize keys from CLS embeddings of actual data samples."""
        if self._keys_initialized:
            return

        embeddings = []
        samples_collected = 0
        for batch in data_stream:
            if samples_collected >= num_samples:
                break
            pixel_values = batch["pixel_values"].to(self.device)
            emb = model.get_patch_embedding(pixel_values)
            embeddings.append(emb)
            samples_collected += pixel_values.size(0)

        all_embeddings = torch.cat(embeddings, dim=0)[:num_samples]

        # Use k-means++ initialization
        keys = []
        idx = torch.randint(0, all_embeddings.size(0), (1,)).item()
        keys.append(all_embeddings[idx])

        for _ in range(1, self.num_prompts):
            key_stack = torch.stack(keys)
            dists = torch.cdist(all_embeddings, key_stack)
            min_dists = dists.min(dim=1).values
            probs = min_dists ** 2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            keys.append(all_embeddings[idx])

        for i, key in enumerate(keys):
            self.keys[i].data = F.normalize(key, dim=0)

        self._keys_initialized = True

    def select_prompt(
        self,
        query: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select prompt based on cosine similarity to keys."""
        keys = torch.stack([k for k in self.keys])
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(keys, dim=-1)
        similarities = query_norm @ keys_norm.T
        per_sample_idx = similarities.argmax(dim=1)

        vote_counts = torch.zeros(self.num_prompts, device=query.device)
        for idx in per_sample_idx:
            vote_counts[idx] += 1

        selected_idx = vote_counts.argmax().item()
        pull_loss = torch.tensor(0.0, device=query.device)

        return selected_idx, self.prompts[selected_idx], pull_loss

    def get_parameters(self) -> list[nn.Parameter]:
        """Return all trainable parameters (only prompts)."""
        return list(self.prompts)


def train_similarity_vision(config: VisionTrainConfig, num_prompts: int = 10) -> dict:
    """Train using input similarity selection on vision task.

    Args:
        config: Training configuration.
        num_prompts: Number of prompts in the pool.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"similarity_vision_{config.dataset}",
        config={**config.__dict__, "num_prompts": num_prompts},
    )

    # Setup model
    print("Loading ViT model...")
    model = FrozenViTWithPrompt(
        model_name=config.backbone,
        num_labels=config.num_classes,
        device=config.device,
    )

    # Setup data
    print("Setting up CIFAR-10 data stream...")
    if config.dataset == "cifar10_binary":
        data_stream = StreamingCIFAR10Binary(
            flip_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
    else:
        data_stream = StreamingCIFAR10(
            rotate_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            rotation_type=config.rotation_type,
        )

    # Prompt pool with keys
    print(f"Initializing {num_prompts} prompts with keys...")
    pool = VisionPromptPoolWithKeys(
        num_prompts=num_prompts,
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        device=config.device,
    )

    # Initialize keys from data
    print("Initializing keys from data...")
    if config.dataset == "cifar10_binary":
        init_stream = StreamingCIFAR10Binary(
            flip_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
        )
    else:
        init_stream = StreamingCIFAR10(
            rotate_every_n=config.rotate_interval,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            rotation_type=config.rotation_type,
        )
    pool.initialize_keys_from_data(model, init_stream, num_samples=200)

    # Optimizer
    optimizer = torch.optim.AdamW(pool.get_parameters(), lr=config.lr)

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
    }

    # Selection counts
    selection_counts = [0] * num_prompts

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
            batch_device = {
                "pixel_values": batch["pixel_values"].to(config.device),
                "labels": batch["labels"].to(config.device),
            }

            # Get input embedding for selection
            input_embed = model.get_patch_embedding(batch_device["pixel_values"])

            # Select prompt based on similarity
            selected_idx, prompt, _ = pool.select_prompt(input_embed)
            selection_counts[selected_idx] += 1

            # Forward pass
            logits = model(batch_device["pixel_values"], prompt)
            loss = F.cross_entropy(logits, batch_device["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == batch_device["labels"]).float().mean().item()

            # Store results
            results["steps"].append(step)
            results["losses"].append(loss.item())
            results["accuracies"].append(accuracy)
            results["regimes"].append(batch["regime"])
            results["selected_prompts"].append(selected_idx)

            # Log to wandb
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "regime": batch["regime"],
                "selected_prompt": selected_idx,
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"similarity_vision_step{step}.pt"
                torch.save({
                    "step": step,
                    "pool_state": pool.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "results": results,
                    "selection_counts": selection_counts,
                }, checkpoint_path)

            progress.update(task, advance=1)

    # Final checkpoint
    checkpoint_path = checkpoint_dir / f"similarity_vision_final.pt"
    torch.save({
        "step": config.total_steps,
        "pool_state": pool.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__,
        "results": results,
        "selection_counts": selection_counts,
    }, checkpoint_path)

    # Log selection distribution
    print("\nPrompt selection distribution:")
    for i, count in enumerate(selection_counts):
        print(f"  Prompt {i}: {count} ({100*count/sum(selection_counts):.1f}%)")

    wandb.finish()
    print(f"\n Training complete! Final checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    # Quick test with minimal steps
    config = VisionTrainConfig(
        total_steps=100,
        birth_interval=50,
        rotate_interval=50,
        checkpoint_interval=50,
        wandb_run_name="vision_test",
    )
    results = train_flh_vision(config)
    print(f"\nFinal accuracy: {results['accuracies'][-1]:.4f}")
