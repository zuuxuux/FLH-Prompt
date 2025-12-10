"""Prompt Pool implementations for online learning.

Algorithms:
- HedgePool: Pure multiplicative weights (baseline, can go extinct)
- FixedSharePool: Fixed Share with per-step mixing (prevents extinction)
- FLHPromptPool: Alias for FixedSharePool (legacy compatibility)
- TrueFLHPool: Follow the Leading History with sub-algorithm ensemble (adaptive regret)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class HedgePool:
    """Pure Multiplicative Weights / Hedge algorithm (baseline).

    No per-step mixing, so prompts can go extinct and never recover.
    Update rule: w_i ← w_i · exp(-η · loss_i), then normalize

    Regret bound: O(√(T log K)) against best SINGLE expert.
    Does NOT compete with switching sequences.

    Args:
        prompt_length: Number of tokens in each prompt.
        embed_dim: Embedding dimension (must match backbone).
        eta: Learning rate for Boltzmann weight updates.
        device: Device to create prompts on.
        init_std: Standard deviation for prompt initialization.
    """

    def __init__(
        self,
        prompt_length: int = 20,
        embed_dim: int = 768,
        eta: float = 0.1,
        device: str = "cuda",
        init_std: float = 0.02,
        # Legacy alias
        alpha: float = None,
    ):
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.eta = alpha if alpha is not None else eta  # Support legacy 'alpha' name
        self.device = device
        self.init_std = init_std

        # State: list of prompts and their weights
        self.prompts: list[nn.Parameter] = []
        self.weights: list[float] = []

        # Birth counter for sleeping expert prior
        self._birth_count = 0

    def birth_prompt(self) -> nn.Parameter:
        """Create and add a new prompt to the pool.

        Uses sleeping expert prior: existing weights are scaled by (1 - 1/t),
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
        """Update weights using multiplicative rule only (no mixing).

        w_i *= exp(-eta * loss_i), then normalize.

        Args:
            losses: List of losses, one per prompt in the pool.
        """
        if len(losses) != len(self.prompts):
            raise ValueError(f"Expected {len(self.prompts)} losses, got {len(losses)}")

        # Multiplicative update only - no mixing, so extinction is possible
        for i, loss in enumerate(losses):
            self.weights[i] *= math.exp(-self.eta * loss)

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
            "eta": self.eta,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.prompt_length = state["prompt_length"]
        self.embed_dim = state["embed_dim"]
        self.eta = state.get("eta", state.get("alpha", 0.1))  # Support legacy 'alpha'
        self._birth_count = state["birth_count"]

        self.prompts = [
            nn.Parameter(p.to(self.device)) for p in state["prompts"]
        ]
        self.weights = state["weights"].copy()


class FixedSharePool:
    """Fixed Share algorithm for competing with switching sequences.

    Two-phase update at each step:
    1. Multiplicative: w_i ← w_i · exp(-η · loss_i), normalize
    2. Mix toward uniform: w_i ← (1 - α) · w_i + α / K

    The mixing rate α ensures every prompt maintains at least α/K weight,
    enabling recovery when regimes revisit. This solves the extinction problem.

    Regret bound: O(√(T log K) + S log K) against best sequence with S switches.

    Args:
        prompt_length: Number of tokens in each prompt.
        embed_dim: Embedding dimension (must match backbone).
        eta: Learning rate for multiplicative updates.
        alpha: Mixing rate toward uniform (prevents extinction).
        device: Device to create prompts on.
        init_std: Standard deviation for prompt initialization.
    """

    def __init__(
        self,
        prompt_length: int = 20,
        embed_dim: int = 768,
        eta: float = 0.1,
        alpha: float = 0.01,  # KEY parameter: mixing rate
        device: str = "cuda",
        init_std: float = 0.02,
    ):
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.eta = eta
        self.alpha = alpha
        self.device = device
        self.init_std = init_std

        self.prompts: list[nn.Parameter] = []
        self.weights: list[float] = []
        self._birth_count = 0
        self._step_count = 0

    def birth_prompt(self) -> nn.Parameter:
        """Create new prompt with sleeping expert prior."""
        new_prompt = nn.Parameter(
            torch.randn(self.prompt_length, self.embed_dim, device=self.device) * self.init_std
        )

        self._birth_count += 1
        t = self._birth_count

        if len(self.weights) > 0:
            mix_factor = 1.0 - 1.0 / t
            self.weights = [w * mix_factor for w in self.weights]
            new_weight = 1.0 / t
        else:
            new_weight = 1.0

        self.prompts.append(new_prompt)
        self.weights.append(new_weight)
        self._normalize_weights()

        return new_prompt

    def update_weights(self, losses: list[float]) -> None:
        """Fixed Share update: multiplicative + mix toward uniform."""
        if len(losses) != len(self.prompts):
            raise ValueError(f"Expected {len(self.prompts)} losses, got {len(losses)}")

        K = len(self.prompts)
        self._step_count += 1

        # Phase 1: Multiplicative update
        for i, loss in enumerate(losses):
            self.weights[i] *= math.exp(-self.eta * loss)
        self._normalize_weights()

        # Phase 2: Mix toward uniform (PREVENTS EXTINCTION)
        for i in range(K):
            self.weights[i] = (1.0 - self.alpha) * self.weights[i] + self.alpha / K

    def _normalize_weights(self) -> None:
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
        elif len(self.weights) > 0:
            self.weights = [1.0 / len(self.weights)] * len(self.weights)

    def get_prompt(self, mode: Literal["weighted_sum", "sample", "top_k"] = "weighted_sum") -> torch.Tensor:
        if len(self.prompts) == 0:
            raise RuntimeError("No prompts in pool. Call birth_prompt() first.")

        if mode == "weighted_sum":
            result = torch.zeros(self.prompt_length, self.embed_dim, device=self.device)
            for prompt, weight in zip(self.prompts, self.weights):
                result += weight * prompt
            return result
        elif mode == "sample":
            idx = torch.multinomial(torch.tensor(self.weights, device=self.device), 1).item()
            return self.prompts[idx]
        elif mode == "top_k":
            idx = max(range(len(self.weights)), key=lambda i: self.weights[i])
            return self.prompts[idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_all_losses(
        self,
        model: nn.Module,
        batch: dict,
    ) -> list[float]:
        """Compute loss for each prompt in the pool."""
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
        entropy = 0.0
        for w in self.weights:
            if w > 1e-10:
                entropy -= w * math.log(w)
        return entropy

    def get_parameters(self) -> list[nn.Parameter]:
        return self.prompts

    def num_prompts(self) -> int:
        return len(self.prompts)

    def get_weight_distribution(self) -> list[float]:
        return self.weights.copy()

    def state_dict(self) -> dict:
        return {
            "prompts": [p.data.clone() for p in self.prompts],
            "weights": self.weights.copy(),
            "birth_count": self._birth_count,
            "step_count": self._step_count,
            "prompt_length": self.prompt_length,
            "embed_dim": self.embed_dim,
            "eta": self.eta,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        self.prompt_length = state["prompt_length"]
        self.embed_dim = state["embed_dim"]
        self.eta = state["eta"]
        self.alpha = state["alpha"]
        self._birth_count = state["birth_count"]
        self._step_count = state.get("step_count", 0)
        self.prompts = [nn.Parameter(p.to(self.device)) for p in state["prompts"]]
        self.weights = state["weights"].copy()


class TrueFLHPool:
    """True Follow the Leading History (FLH) algorithm.

    Maintains an ensemble of sub-algorithms (Hedge copies) started at different times.
    This achieves adaptive regret - low regret on ANY interval, not just [1, T].

    Structure:
    - Copies A_1, A_2, ..., each started at a different step with uniform weights
    - Meta-weights π_s (harmonic prior: 1/s normalized) combine copies
    - Final weights: w_i = Σ_s π_s · w_i^(s)

    Uses geometric spacing for O(log T) copies instead of T.

    Regret bound: O(√(τ log K)) on ANY interval of length τ.

    Args:
        prompt_length: Number of tokens in each prompt.
        embed_dim: Embedding dimension (must match backbone).
        eta: Learning rate for each sub-algorithm.
        device: Device to create prompts on.
        init_std: Standard deviation for prompt initialization.
        geometric_base: Factor for geometric spacing of sub-algorithm births.
    """

    def __init__(
        self,
        prompt_length: int = 20,
        embed_dim: int = 768,
        eta: float = 0.1,
        device: str = "cuda",
        init_std: float = 0.02,
        geometric_base: float = 2.0,
        # Legacy alias
        alpha: float = None,
    ):
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.eta = alpha if alpha is not None else eta
        self.device = device
        self.init_std = init_std
        self.geometric_base = geometric_base

        self.prompts: list[nn.Parameter] = []

        # Sub-algorithms: list of {"weights": [...], "birth_step": s}
        self.sub_algorithms: list[dict] = []
        self.meta_weights: list[float] = []

        self._birth_count = 0
        self._step_count = 0
        self._next_sub_birth_step = 2  # Start at 2 for clean geometric: 0, 2, 4, 8, 16...

    def birth_prompt(self) -> nn.Parameter:
        """Create new prompt and add to all sub-algorithms."""
        new_prompt = nn.Parameter(
            torch.randn(self.prompt_length, self.embed_dim, device=self.device) * self.init_std
        )

        self._birth_count += 1
        K_new = self._birth_count

        self.prompts.append(new_prompt)

        # Update all existing sub-algorithms with sleeping expert prior
        for sub in self.sub_algorithms:
            old_K = len(sub["weights"])
            if old_K > 0:
                mix_factor = 1.0 - 1.0 / K_new
                sub["weights"] = [w * mix_factor for w in sub["weights"]]
                sub["weights"].append(1.0 / K_new)
            else:
                sub["weights"].append(1.0)
            self._normalize_sub_weights(sub)

        # Create first sub-algorithm if this is the first prompt
        if len(self.sub_algorithms) == 0:
            self._birth_sub_algorithm()

        return new_prompt

    def _birth_sub_algorithm(self) -> None:
        """Create new sub-algorithm with uniform weights."""
        K = len(self.prompts)
        if K == 0:
            return

        self.sub_algorithms.append({
            "weights": [1.0 / K] * K,
            "birth_step": self._step_count,
        })
        self._update_meta_weights()

    def _update_meta_weights(self) -> None:
        """Compute meta-weights using harmonic prior based on birth step.

        Uses 1/(birth_step + 1) so earlier sub-algorithms get higher weight.
        This properly reflects when each copy started, not its position in the list.
        """
        n = len(self.sub_algorithms)
        if n == 0:
            self.meta_weights = []
            return

        # Use birth_step, not index - with geometric spacing birth steps are 0, 2, 4, 8...
        raw = [1.0 / max(1, sub["birth_step"] + 1) for sub in self.sub_algorithms]
        total = sum(raw)
        self.meta_weights = [w / total for w in raw]

    def _normalize_sub_weights(self, sub: dict) -> None:
        total = sum(sub["weights"])
        if total > 0:
            sub["weights"] = [w / total for w in sub["weights"]]
        elif len(sub["weights"]) > 0:
            sub["weights"] = [1.0 / len(sub["weights"])] * len(sub["weights"])

    def update_weights(self, losses: list[float]) -> None:
        """Update all sub-algorithms and optionally birth new one."""
        if len(losses) != len(self.prompts):
            raise ValueError(f"Expected {len(self.prompts)} losses, got {len(losses)}")

        self._step_count += 1

        # Update each sub-algorithm
        for sub in self.sub_algorithms:
            for i, loss in enumerate(losses):
                sub["weights"][i] *= math.exp(-self.eta * loss)
            self._normalize_sub_weights(sub)

        # Birth new sub-algorithm at geometric intervals
        if self._step_count >= self._next_sub_birth_step:
            self._birth_sub_algorithm()
            self._next_sub_birth_step = int(self._next_sub_birth_step * self.geometric_base)

    def get_combined_weights(self) -> list[float]:
        """Compute final weights as weighted combination of sub-algorithms."""
        K = len(self.prompts)
        if K == 0:
            return []

        combined = [0.0] * K
        for sub, meta_w in zip(self.sub_algorithms, self.meta_weights):
            for i, w in enumerate(sub["weights"]):
                combined[i] += meta_w * w

        return combined

    @property
    def weights(self) -> list[float]:
        """Compatibility property returning combined weights."""
        return self.get_combined_weights()

    def get_prompt(self, mode: Literal["weighted_sum", "sample", "top_k"] = "weighted_sum") -> torch.Tensor:
        if len(self.prompts) == 0:
            raise RuntimeError("No prompts in pool. Call birth_prompt() first.")

        weights = self.get_combined_weights()

        if mode == "weighted_sum":
            result = torch.zeros(self.prompt_length, self.embed_dim, device=self.device)
            for prompt, weight in zip(self.prompts, weights):
                result += weight * prompt
            return result
        elif mode == "sample":
            idx = torch.multinomial(torch.tensor(weights, device=self.device), 1).item()
            return self.prompts[idx]
        elif mode == "top_k":
            idx = max(range(len(weights)), key=lambda i: weights[i])
            return self.prompts[idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_all_losses(
        self,
        model: nn.Module,
        batch: dict,
    ) -> list[float]:
        """Compute loss for each prompt in the pool."""
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
        weights = self.get_combined_weights()
        entropy = 0.0
        for w in weights:
            if w > 1e-10:
                entropy -= w * math.log(w)
        return entropy

    def get_parameters(self) -> list[nn.Parameter]:
        return self.prompts

    def num_prompts(self) -> int:
        return len(self.prompts)

    def num_sub_algorithms(self) -> int:
        return len(self.sub_algorithms)

    def get_weight_distribution(self) -> list[float]:
        return self.get_combined_weights()

    def state_dict(self) -> dict:
        return {
            "prompts": [p.data.clone() for p in self.prompts],
            "sub_algorithms": [{"weights": s["weights"].copy(), "birth_step": s["birth_step"]}
                              for s in self.sub_algorithms],
            "meta_weights": self.meta_weights.copy(),
            "birth_count": self._birth_count,
            "step_count": self._step_count,
            "next_sub_birth_step": self._next_sub_birth_step,
            "prompt_length": self.prompt_length,
            "embed_dim": self.embed_dim,
            "eta": self.eta,
            "geometric_base": self.geometric_base,
        }

    def load_state_dict(self, state: dict) -> None:
        self.prompt_length = state["prompt_length"]
        self.embed_dim = state["embed_dim"]
        self.eta = state["eta"]
        self.geometric_base = state["geometric_base"]
        self._birth_count = state["birth_count"]
        self._step_count = state["step_count"]
        self._next_sub_birth_step = state["next_sub_birth_step"]
        self.prompts = [nn.Parameter(p.to(self.device)) for p in state["prompts"]]
        self.sub_algorithms = [{"weights": s["weights"].copy(), "birth_step": s["birth_step"]}
                              for s in state["sub_algorithms"]]
        self.meta_weights = state["meta_weights"].copy()


# Legacy alias for backwards compatibility
FLHPromptPool = FixedSharePool


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
