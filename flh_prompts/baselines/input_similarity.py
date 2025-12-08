"""Input similarity baseline (mini-L2P) - selection via cosine similarity to keys."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from flh_prompts.data.streaming import StreamingSST2
from flh_prompts.models.frozen_backbone import FrozenBERTWithPrompt
from flh_prompts.training.trainer import TrainConfig


class PromptPoolWithKeys(nn.Module):
    """Prompt pool with learnable keys for input-based selection.

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

        # Initialize keys (one per prompt)
        self.keys = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, device=device) * 0.02)
            for _ in range(num_prompts)
        ])

    def get_input_embedding(
        self,
        model: FrozenBERTWithPrompt,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get the [CLS] token embedding for input-based selection.

        Args:
            model: The frozen backbone model.
            input_ids: Token IDs of shape [batch, seq_len].

        Returns:
            CLS embeddings of shape [batch, embed_dim].
        """
        with torch.no_grad():
            # Get word embeddings (without position embeddings for simpler matching)
            embeddings = model.embeddings.word_embeddings(input_ids)
            # Use mean of all tokens as input representation
            return embeddings.mean(dim=1)

    def select_prompt(
        self,
        query: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """Select prompt based on cosine similarity to keys.

        Args:
            query: Query vector of shape [batch, embed_dim].

        Returns:
            Tuple of (selected_index, selected_prompt).
        """
        # Stack keys: [num_prompts, embed_dim]
        keys = torch.stack([k for k in self.keys])

        # Mean query across batch
        query_mean = query.mean(dim=0)  # [embed_dim]

        # Compute cosine similarities
        query_norm = F.normalize(query_mean.unsqueeze(0), dim=-1)  # [1, embed_dim]
        keys_norm = F.normalize(keys, dim=-1)  # [num_prompts, embed_dim]

        similarities = (query_norm @ keys_norm.T).squeeze(0)  # [num_prompts]

        # Select prompt with highest similarity
        selected_idx = similarities.argmax().item()

        return selected_idx, self.prompts[selected_idx]

    def get_parameters(self) -> list[nn.Parameter]:
        """Return all parameters (prompts + keys) for optimizer."""
        return list(self.prompts) + list(self.keys)


def train_input_similarity(config: TrainConfig, num_prompts: int = 10) -> dict:
    """Train using input similarity selection (mini-L2P).

    This baseline selects prompts based on cosine similarity between
    the input embedding and learned key vectors.

    Args:
        config: Training configuration.
        num_prompts: Number of prompts to initialize.

    Returns:
        Dictionary of training results.
    """
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or "input_similarity",
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
    data_stream = StreamingSST2(
        flip_every_n=config.flip_interval,
        batch_size=config.batch_size,
        tokenizer_name=config.backbone,
    )

    # Prompt pool with keys
    print(f"Initializing {num_prompts} prompts with keys...")
    pool = PromptPoolWithKeys(
        num_prompts=num_prompts,
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        device=config.device,
    )

    # Optimizer for prompts and keys
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

    # Selection counts for analysis
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
            batch = {
                "input_ids": batch["input_ids"].to(config.device),
                "attention_mask": batch["attention_mask"].to(config.device),
                "labels": batch["labels"].to(config.device),
                "step": batch["step"],
                "regime": batch["regime"],
                "flipped": batch["flipped"],
            }

            # Get input embedding for selection
            input_embed = pool.get_input_embedding(model, batch["input_ids"])

            # Select prompt based on similarity
            selected_idx, prompt = pool.select_prompt(input_embed)
            selection_counts[selected_idx] += 1

            # Forward pass
            logits = model(batch["input_ids"], batch["attention_mask"], prompt)
            loss = F.cross_entropy(logits, batch["labels"])

            # Backward pass (updates both prompt and its key)
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
            results["selected_prompts"].append(selected_idx)

            # Log to wandb
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "regime": batch["regime"],
                "flipped": batch["flipped"],
                "selected_prompt": selected_idx,
            })

            # Checkpoint
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"similarity_step{step}.pt"
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
    checkpoint_path = checkpoint_dir / f"similarity_final.pt"
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
    print(f"\n✓ Training complete! Final checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    config = TrainConfig(
        total_steps=100,
        flip_interval=50,
        checkpoint_interval=50,
        wandb_run_name="similarity_test",
    )
    results = train_input_similarity(config, num_prompts=5)
    print(f"\nFinal accuracy: {results['accuracies'][-1]:.4f}")
