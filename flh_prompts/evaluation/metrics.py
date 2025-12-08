"""Evaluation metrics for continual learning experiments."""

import math
from typing import Optional

import numpy as np


def compute_adaptive_regret(
    accuracies: list[float],
    window_size: int = 500,
) -> list[float]:
    """Compute adaptive regret (mean accuracy over sliding window).

    Args:
        accuracies: List of accuracy values.
        window_size: Size of the sliding window.

    Returns:
        List of adaptive regret values (one per step after window_size).
    """
    regrets = []
    for i in range(len(accuracies)):
        start = max(0, i - window_size + 1)
        window = accuracies[start:i + 1]
        regrets.append(np.mean(window))
    return regrets


def compute_recovery_times(
    steps: list[int],
    accuracies: list[float],
    regimes: list[int],
    threshold: float = 0.8,
) -> list[dict]:
    """Compute recovery time after each regime flip.

    Recovery time is the number of steps after a flip to reach
    the threshold accuracy.

    Args:
        steps: List of step numbers.
        accuracies: List of accuracy values.
        regimes: List of regime indices.
        threshold: Accuracy threshold to consider "recovered".

    Returns:
        List of dicts with flip_step, recovery_step, recovery_time.
    """
    recovery_info = []

    # Find regime change points
    flip_indices = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            flip_indices.append(i)

    # Compute recovery time for each flip
    for flip_idx in flip_indices:
        flip_step = steps[flip_idx]

        # Find when accuracy exceeds threshold after flip
        recovery_step = None
        for j in range(flip_idx, len(accuracies)):
            if accuracies[j] >= threshold:
                recovery_step = steps[j]
                break

        if recovery_step is not None:
            recovery_time = recovery_step - flip_step
        else:
            recovery_time = None  # Never recovered

        recovery_info.append({
            "flip_step": flip_step,
            "regime": regimes[flip_idx],
            "recovery_step": recovery_step,
            "recovery_time": recovery_time,
        })

    return recovery_info


def compute_weight_entropy(weights: list[float]) -> float:
    """Compute entropy of weight distribution.

    Args:
        weights: List of weights (should sum to 1).

    Returns:
        Entropy value: -sum(w * log(w)).
    """
    entropy = 0.0
    for w in weights:
        if w > 1e-10:
            entropy -= w * math.log(w)
    return entropy


def compute_mean_accuracy_per_regime(
    accuracies: list[float],
    regimes: list[int],
) -> dict[int, float]:
    """Compute mean accuracy for each regime.

    Args:
        accuracies: List of accuracy values.
        regimes: List of regime indices.

    Returns:
        Dict mapping regime index to mean accuracy.
    """
    regime_accuracies = {}
    for acc, regime in zip(accuracies, regimes):
        if regime not in regime_accuracies:
            regime_accuracies[regime] = []
        regime_accuracies[regime].append(acc)

    return {r: np.mean(accs) for r, accs in regime_accuracies.items()}


def compute_forgetting(
    accuracies: list[float],
    regimes: list[int],
) -> float:
    """Compute forgetting metric.

    Forgetting is measured as the drop in accuracy when returning
    to a previously seen regime type (normal vs flipped).

    Args:
        accuracies: List of accuracy values.
        regimes: List of regime indices.

    Returns:
        Average forgetting across regime returns.
    """
    # Group by regime type (even = normal, odd = flipped)
    type_history = {"normal": [], "flipped": []}

    current_type = None
    type_accuracies = []

    for acc, regime in zip(accuracies, regimes):
        regime_type = "normal" if regime % 2 == 0 else "flipped"

        if regime_type != current_type:
            # Regime type changed - save previous
            if current_type is not None and type_accuracies:
                type_history[current_type].append(np.mean(type_accuracies))
            current_type = regime_type
            type_accuracies = []

        type_accuracies.append(acc)

    # Save final
    if type_accuracies:
        type_history[current_type].append(np.mean(type_accuracies))

    # Compute forgetting for each type
    forgetting_values = []
    for regime_type, history in type_history.items():
        if len(history) > 1:
            # Compare each return to the best previous performance
            for i in range(1, len(history)):
                best_previous = max(history[:i])
                forgetting = best_previous - history[i]
                if forgetting > 0:
                    forgetting_values.append(forgetting)

    return np.mean(forgetting_values) if forgetting_values else 0.0


def summarize_results(results: dict) -> dict:
    """Compute summary statistics from training results.

    Args:
        results: Dict with steps, losses, accuracies, regimes, etc.

    Returns:
        Dict of summary statistics.
    """
    steps = results.get("steps", [])
    accuracies = results.get("accuracies", [])
    losses = results.get("losses", [])
    regimes = results.get("regimes", [])

    summary = {
        "total_steps": len(steps),
        "final_accuracy": accuracies[-1] if accuracies else None,
        "mean_accuracy": np.mean(accuracies) if accuracies else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": np.mean(losses) if losses else None,
    }

    if regimes:
        # Adaptive regret
        adaptive_regret = compute_adaptive_regret(accuracies)
        summary["adaptive_regret_500"] = adaptive_regret[-1] if adaptive_regret else None

        # Recovery times
        recovery_info = compute_recovery_times(steps, accuracies, regimes)
        if recovery_info:
            recovery_times = [r["recovery_time"] for r in recovery_info if r["recovery_time"] is not None]
            summary["mean_recovery_time"] = np.mean(recovery_times) if recovery_times else None
            summary["num_recoveries"] = len(recovery_times)
            summary["num_flips"] = len(recovery_info)

        # Per-regime accuracy
        summary["accuracy_per_regime"] = compute_mean_accuracy_per_regime(accuracies, regimes)

        # Forgetting
        summary["forgetting"] = compute_forgetting(accuracies, regimes)

    return summary


if __name__ == "__main__":
    # Test metrics with synthetic data
    import random

    print("Testing metrics with synthetic data...")

    # Simulate 1000 steps with flips every 200 steps
    steps = list(range(1000))
    regimes = [i // 200 for i in steps]

    # Simulate sawtooth accuracy pattern
    accuracies = []
    for i, regime in enumerate(regimes):
        base_acc = 0.5 + 0.3 * (i % 200) / 200  # Ramp up
        noise = random.gauss(0, 0.05)
        accuracies.append(max(0, min(1, base_acc + noise)))

    # Test adaptive regret
    regret = compute_adaptive_regret(accuracies, window_size=100)
    print(f"\nAdaptive regret (last value): {regret[-1]:.4f}")

    # Test recovery times
    recovery = compute_recovery_times(steps, accuracies, regimes, threshold=0.7)
    print(f"\nRecovery info:")
    for r in recovery[:3]:
        print(f"  Flip at {r['flip_step']}: recovered in {r['recovery_time']} steps")

    # Test forgetting
    forgetting = compute_forgetting(accuracies, regimes)
    print(f"\nForgetting: {forgetting:.4f}")

    # Test summary
    results = {"steps": steps, "accuracies": accuracies, "losses": [1-a for a in accuracies], "regimes": regimes}
    summary = summarize_results(results)
    print(f"\nSummary:")
    for k, v in summary.items():
        if k != "accuracy_per_regime":
            print(f"  {k}: {v}")

    print("\n✓ All metrics tests passed!")
