"""Input similarity baseline (mini-L2P) - selection via cosine similarity to keys."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from flh_prompts.data.streaming import StreamingSST2, StreamingAmazonMultiDomain
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

        # Keys will be initialized from data in initialize_keys_from_data()
        # Start with placeholder random keys
        self.keys = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, device=device), requires_grad=False)
            for _ in range(num_prompts)
        ])
        self._keys_initialized = False

    def initialize_keys_from_data(
        self,
        model: "FrozenBERTWithPrompt",
        data_stream,
        num_samples: int = 100,
    ) -> None:
        """Initialize keys from CLS embeddings of actual data samples.

        Uses k-means-style clustering to find diverse key vectors that
        span the input embedding space.
        """
        if self._keys_initialized:
            return

        # Collect CLS embeddings from data
        embeddings = []
        for i, batch in enumerate(data_stream):
            if len(embeddings) * batch["input_ids"].size(0) >= num_samples:
                break
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            emb = self.get_input_embedding(model, input_ids, attention_mask)
            embeddings.append(emb)

        all_embeddings = torch.cat(embeddings, dim=0)[:num_samples]

        # Use k-means++ initialization to get diverse keys
        keys = []
        # First key: random sample
        idx = torch.randint(0, all_embeddings.size(0), (1,)).item()
        keys.append(all_embeddings[idx])

        # Subsequent keys: sample proportional to squared distance from nearest key
        for _ in range(1, self.num_prompts):
            # Compute distance to nearest existing key
            key_stack = torch.stack(keys)
            dists = torch.cdist(all_embeddings, key_stack)  # [n, k]
            min_dists = dists.min(dim=1).values  # [n]
            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            keys.append(all_embeddings[idx])

        # Update key parameters
        for i, key in enumerate(keys):
            self.keys[i].data = F.normalize(key, dim=0)

        self._keys_initialized = True

    def get_input_embedding(
        self,
        model: FrozenBERTWithPrompt,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the [CLS] token embedding for input-based selection.

        Uses the backbone's CLS output (without prompt) for a more
        discriminative representation than raw word embeddings.

        Args:
            model: The frozen backbone model.
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].

        Returns:
            CLS embeddings of shape [batch, embed_dim].
        """
        with torch.no_grad():
            # Run through BERT encoder WITHOUT prompt to get CLS representation
            # This gives a more discriminative embedding than raw word embeddings
            input_embeds = model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            position_embeds = model.embeddings.position_embeddings(position_ids)
            token_type_ids = torch.zeros_like(input_ids)
            token_type_embeds = model.embeddings.token_type_embeddings(token_type_ids)

            embeddings = input_embeds + position_embeds + token_type_embeds
            embeddings = model.embeddings.LayerNorm(embeddings)
            embeddings = model.embeddings.dropout(embeddings)

            # Get extended attention mask
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask.float()) * -10000.0

            # Run through encoder
            encoder_outputs = model.model.bert.encoder(embeddings, attention_mask=extended_mask)

            # CLS token is the first token's hidden state
            cls_embedding = encoder_outputs[0][:, 0, :]
            return cls_embedding

    def select_prompt(
        self,
        query: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select prompt based on cosine similarity to keys.

        Uses per-sample similarity computation, then voting across batch
        to select the most frequently chosen prompt (L2P-style).

        Args:
            query: Query vector of shape [batch, embed_dim].

        Returns:
            Tuple of (selected_index, selected_prompt, pull_loss).
            pull_loss encourages selected key to match query (L2P-style).
        """
        # Stack keys: [num_prompts, embed_dim]
        keys = torch.stack([k for k in self.keys])

        # Normalize query per sample: [batch, embed_dim]
        query_norm = F.normalize(query, dim=-1)
        # Normalize keys: [num_prompts, embed_dim]
        keys_norm = F.normalize(keys, dim=-1)

        # Compute per-sample similarities: [batch, num_prompts]
        similarities = query_norm @ keys_norm.T

        # Get top-1 selection per sample: [batch]
        per_sample_idx = similarities.argmax(dim=1)

        # Count votes for each prompt
        num_prompts = len(self.prompts)
        vote_counts = torch.zeros(num_prompts, device=query.device)
        for idx in per_sample_idx:
            vote_counts[idx] += 1

        # Select prompt with most votes (deterministic for reproducibility)
        selected_idx = vote_counts.argmax().item()

        # No pull loss - keys are fixed, only prompts learn
        # This prevents winner-take-all collapse
        pull_loss = torch.tensor(0.0, device=query.device)

        return selected_idx, self.prompts[selected_idx], pull_loss

    def get_parameters(self) -> list[nn.Parameter]:
        """Return all trainable parameters (only prompts, not keys)."""
        # Keys are fixed for stable selection, only prompts learn
        return list(self.prompts)


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

    # Prompt pool with keys
    print(f"Initializing {num_prompts} prompts with keys...")
    pool = PromptPoolWithKeys(
        num_prompts=num_prompts,
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        device=config.device,
    )

    # Initialize keys from data (use k-means++ on CLS embeddings)
    print("Initializing keys from data...")
    if config.dataset == "amazon":
        init_stream = StreamingAmazonMultiDomain(
            steps_per_domain=config.steps_per_domain,
            batch_size=config.batch_size,
            tokenizer_name=config.backbone,
            domains=config.amazon_domains,
            balance_labels=config.balance_labels,
        )
    else:
        init_stream = StreamingSST2(
            flip_every_n=config.flip_interval,
            batch_size=config.batch_size,
            tokenizer_name=config.backbone,
        )
    pool.initialize_keys_from_data(model, init_stream, num_samples=200)

    # Optimizer for prompts and classifier (keys are fixed)
    optimizer = torch.optim.AdamW(
        pool.get_parameters() + list(model.model.classifier.parameters()),
        lr=config.lr
    )

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

            # Get input embedding for selection (uses CLS from backbone)
            input_embed = pool.get_input_embedding(
                model, batch_device["input_ids"], batch_device["attention_mask"]
            )

            # Select prompt based on similarity
            selected_idx, prompt, _ = pool.select_prompt(input_embed)
            selection_counts[selected_idx] += 1

            # Forward pass
            logits = model(batch_device["input_ids"], batch_device["attention_mask"], prompt)
            loss = F.cross_entropy(logits, batch_device["labels"])

            # Backward pass (only updates selected prompt, keys are fixed)
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
