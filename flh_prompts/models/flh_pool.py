"""FLH (Fixed-share with Learned History) Prompt Pool implementation."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLHPromptPool:
    """Fixed-share with Learned History prompt pool for continual learning.

    Maintains a pool of soft prompts with Boltzmann-weighted selection.
    New prompts can be "birthed" at any time, and weights are updated
    based on per-prompt losses using the FLH algorithm.

    Args:
        prompt_length: Number of tokens in each prompt.
        embed_dim: Embedding dimension (must match backbone).
        alpha: Temperature parameter for Boltzmann weight updates.
        device: Device to create prompts on.
        init_std: Standard deviation for prompt initialization.
    """

    def __init__(
        self,
        prompt_length: int = 20,
        embed_dim: int = 768,
        alpha: float = 0.1,
        device: str = "cuda",
        init_std: float = 0.02,
    ):
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.device = device
        self.init_std = init_std

        # State: list of prompts and their weights
        self.prompts: list[nn.Parameter] = []
        self.weights: list[float] = []

        # Birth counter for FLH mixing
        self._birth_count = 0

    def birth_prompt(self) -> nn.Parameter:
        """Create and add a new prompt to the pool.

        Uses FLH mixing rule: existing weights are scaled by (1 - 1/t),
        and the new prompt gets weight 1/t, where t is the birth count.

        Returns:
            The newly created prompt parameter.
        """
        # Initialize new prompt with small random values
        new_prompt = nn.Parameter(
            torch.randn(self.prompt_length, self.embed_dim, device=self.device) * self.init_std
        )

        self._birth_count += 1
        t = self._birth_count

        # FLH mixing: redistribute weights
        if len(self.weights) > 0:
            mix_factor = 1.0 - 1.0 / t
            self.weights = [w * mix_factor for w in self.weights]
            new_weight = 1.0 / t
        else:
            new_weight = 1.0

        self.prompts.append(new_prompt)
        self.weights.append(new_weight)

        # Ensure weights still sum to 1 (handle numerical issues)
        self._normalize_weights()

        return new_prompt

    def update_weights(self, losses: list[float]) -> None:
        """Update weights using Boltzmann distribution over losses.

        w_i *= exp(-alpha * loss_i), then normalize.

        Args:
            losses: List of losses, one per prompt in the pool.
        """
        if len(losses) != len(self.prompts):
            raise ValueError(f"Expected {len(self.prompts)} losses, got {len(losses)}")

        # Boltzmann update
        for i, loss in enumerate(losses):
            self.weights[i] *= math.exp(-self.alpha * loss)

        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
        elif len(self.weights) > 0:
            # If all weights are 0, reset to uniform
            self.weights = [1.0 / len(self.weights)] * len(self.weights)

    def get_prompt(
        self,
        mode: Literal["weighted_sum", "sample", "top_k"] = "weighted_sum",
    ) -> torch.Tensor:
        """Get a prompt from the pool using the specified selection mode.

        Args:
            mode:
                - "weighted_sum": Return weighted combination of all prompts.
                - "sample": Thompson-sample one prompt proportional to weights.
                - "top_k": Return the highest-weight prompt.

        Returns:
            Prompt tensor of shape [prompt_length, embed_dim].
        """
        if len(self.prompts) == 0:
            raise RuntimeError("No prompts in pool. Call birth_prompt() first.")

        if mode == "weighted_sum":
            # Weighted combination of all prompts
            result = torch.zeros(self.prompt_length, self.embed_dim, device=self.device)
            for prompt, weight in zip(self.prompts, self.weights):
                result += weight * prompt
            return result

        elif mode == "sample":
            # Thompson sampling proportional to weights
            idx = torch.multinomial(
                torch.tensor(self.weights, device=self.device),
                num_samples=1
            ).item()
            return self.prompts[idx]

        elif mode == "top_k":
            # Return highest-weight prompt
            idx = max(range(len(self.weights)), key=lambda i: self.weights[i])
            return self.prompts[idx]

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_all_losses(
        self,
        model: nn.Module,
        batch: dict,
    ) -> list[float]:
        """Compute loss for each prompt in the pool.

        Args:
            model: The frozen backbone model.
            batch: Batch dict with input_ids, attention_mask, labels.

        Returns:
            List of cross-entropy losses, one per prompt.
        """
        losses = []

        with torch.no_grad():
            for prompt in self.prompts:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    prompt,
                )
                loss = F.cross_entropy(logits, batch["labels"])
                losses.append(loss.item())

        return losses

    def get_entropy(self) -> float:
        """Compute entropy of the weight distribution.

        Higher entropy = more uniform weights.
        Lower entropy = weights concentrated on few prompts.

        Returns:
            Entropy value: -sum(w * log(w)).
        """
        entropy = 0.0
        for w in self.weights:
            if w > 1e-10:  # Avoid log(0)
                entropy -= w * math.log(w)
        return entropy

    def get_parameters(self) -> list[nn.Parameter]:
        """Return all prompt parameters for optimizer."""
        return self.prompts

    def num_prompts(self) -> int:
        """Return the number of prompts in the pool."""
        return len(self.prompts)

    def get_weight_distribution(self) -> list[float]:
        """Return copy of current weight distribution."""
        return self.weights.copy()

    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "prompts": [p.data.clone() for p in self.prompts],
            "weights": self.weights.copy(),
            "birth_count": self._birth_count,
            "prompt_length": self.prompt_length,
            "embed_dim": self.embed_dim,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.prompt_length = state["prompt_length"]
        self.embed_dim = state["embed_dim"]
        self.alpha = state["alpha"]
        self._birth_count = state["birth_count"]

        self.prompts = [
            nn.Parameter(p.to(self.device)) for p in state["prompts"]
        ]
        self.weights = state["weights"].copy()


def test_flh_pool():
    """Test the FLH prompt pool."""
    print("Testing FLHPromptPool...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = FLHPromptPool(prompt_length=20, embed_dim=768, alpha=0.1, device=device)

    # Test 1: Birth prompts
    print("\n1. Birthing 3 prompts...")
    for i in range(3):
        pool.birth_prompt()
        print(f"   After birth {i+1}: weights = {[f'{w:.4f}' for w in pool.weights]}")

    # Test 2: Update weights with synthetic losses
    print("\n2. Updating weights with losses [0.5, 1.0, 2.0]...")
    pool.update_weights([0.5, 1.0, 2.0])
    print(f"   New weights: {[f'{w:.4f}' for w in pool.weights]}")
    print(f"   (Lowest loss prompt should have highest weight)")

    # Verify lowest loss prompt has highest weight
    assert pool.weights[0] > pool.weights[1] > pool.weights[2], "Weights not ordered correctly!"
    print("   ✓ Weight ordering correct")

    # Test 3: Get prompts in different modes
    print("\n3. Testing get_prompt modes...")
    weighted = pool.get_prompt(mode="weighted_sum")
    print(f"   weighted_sum shape: {weighted.shape}")

    top_k = pool.get_prompt(mode="top_k")
    print(f"   top_k shape: {top_k.shape}")

    sampled = pool.get_prompt(mode="sample")
    print(f"   sample shape: {sampled.shape}")

    # Test 4: Entropy
    print("\n4. Testing entropy...")
    entropy = pool.get_entropy()
    print(f"   Entropy: {entropy:.4f}")

    # Reset to uniform and check entropy
    pool.weights = [1/3, 1/3, 1/3]
    uniform_entropy = pool.get_entropy()
    print(f"   Uniform entropy: {uniform_entropy:.4f}")
    print(f"   Max entropy for 3 prompts: {math.log(3):.4f}")

    # Test 5: State dict
    print("\n5. Testing state_dict...")
    state = pool.state_dict()
    new_pool = FLHPromptPool(device=device)
    new_pool.load_state_dict(state)
    print(f"   Restored {new_pool.num_prompts()} prompts")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_flh_pool()
