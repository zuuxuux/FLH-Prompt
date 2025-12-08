"""Plotting utilities for FLH-Prompts experiments."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_over_time(
    steps: list[int],
    accuracies: list[float],
    regimes: list[int],
    output_path: str | Path,
    title: str = "Accuracy Over Time",
    smooth_window: int = 50,
):
    """Plot accuracy over time with regime change markers.

    Args:
        steps: List of step numbers.
        accuracies: List of accuracy values.
        regimes: List of regime indices.
        output_path: Path to save the plot.
        title: Plot title.
        smooth_window: Window size for smoothing.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Smooth accuracies
    if smooth_window > 1:
        smoothed = np.convolve(accuracies, np.ones(smooth_window)/smooth_window, mode='valid')
        steps_smoothed = steps[smooth_window-1:]
    else:
        smoothed = accuracies
        steps_smoothed = steps

    # Plot accuracy
    ax.plot(steps_smoothed, smoothed, label="Accuracy", color="blue", linewidth=1)

    # Mark regime changes with vertical lines
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            ax.axvline(x=steps[i], color="red", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_weight_entropy(
    steps: list[int],
    entropies: list[float],
    output_path: str | Path,
    title: str = "Weight Entropy Over Time",
):
    """Plot weight entropy over time.

    Args:
        steps: List of step numbers.
        entropies: List of entropy values.
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(steps, entropies, color="green", linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_comparison(
    results_dict: dict[str, dict],
    output_path: str | Path,
    title: str = "Method Comparison",
    smooth_window: int = 50,
):
    """Plot accuracy comparison across multiple methods.

    Args:
        results_dict: Dict mapping method name to results dict.
        output_path: Path to save the plot.
        title: Plot title.
        smooth_window: Window size for smoothing.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["blue", "orange", "green", "red", "purple"]

    for idx, (method_name, results) in enumerate(results_dict.items()):
        steps = results["steps"]
        accuracies = results["accuracies"]

        # Smooth
        if smooth_window > 1:
            smoothed = np.convolve(accuracies, np.ones(smooth_window)/smooth_window, mode='valid')
            steps_smoothed = steps[smooth_window-1:]
        else:
            smoothed = accuracies
            steps_smoothed = steps

        color = colors[idx % len(colors)]
        ax.plot(steps_smoothed, smoothed, label=method_name, color=color, linewidth=1.5)

    # Mark regime changes (use first result for reference)
    first_results = list(results_dict.values())[0]
    regimes = first_results.get("regimes", [])
    steps = first_results.get("steps", [])
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            ax.axvline(x=steps[i], color="gray", linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_weight_heatmap(
    steps: list[int],
    weight_history: list[list[float]],
    output_path: str | Path,
    title: str = "Weight Distribution Over Time",
):
    """Plot heatmap of prompt weights over time.

    Args:
        steps: List of step numbers.
        weight_history: List of weight distributions (one per step).
        output_path: Path to save the plot.
        title: Plot title.
    """
    # Convert to numpy array, padding shorter weight lists
    max_prompts = max(len(w) for w in weight_history)
    weight_matrix = np.zeros((len(weight_history), max_prompts))

    for i, weights in enumerate(weight_history):
        weight_matrix[i, :len(weights)] = weights

    fig, ax = plt.subplots(figsize=(14, 6))

    # Subsample if too many steps
    if len(steps) > 500:
        subsample = len(steps) // 500
        weight_matrix = weight_matrix[::subsample]
        steps = steps[::subsample]

    im = ax.imshow(
        weight_matrix.T,
        aspect="auto",
        cmap="YlOrRd",
        extent=[steps[0], steps[-1], -0.5, max_prompts - 0.5],
        origin="lower",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Prompt Index")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_recovery_bars(
    recovery_data: dict[str, list[dict]],
    output_path: str | Path,
    title: str = "Recovery Time Comparison",
):
    """Plot bar chart of recovery times per method.

    Args:
        recovery_data: Dict mapping method name to list of recovery info dicts.
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(recovery_data.keys())
    mean_recovery = []
    std_recovery = []

    for method in methods:
        recovery_times = [
            r["recovery_time"]
            for r in recovery_data[method]
            if r["recovery_time"] is not None
        ]
        if recovery_times:
            mean_recovery.append(np.mean(recovery_times))
            std_recovery.append(np.std(recovery_times))
        else:
            mean_recovery.append(0)
            std_recovery.append(0)

    x = np.arange(len(methods))
    bars = ax.bar(x, mean_recovery, yerr=std_recovery, capsize=5)

    ax.set_xlabel("Method")
    ax.set_ylabel("Recovery Time (steps)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison_from_wandb(
    run_ids: list[str],
    output_path: str | Path,
    project: str = "flh-prompts",
):
    """Fetch data from wandb runs and create comparison plot.

    Args:
        run_ids: List of wandb run IDs.
        output_path: Path to save the plot.
        project: Wandb project name.
    """
    import wandb

    api = wandb.Api()

    results_dict = {}
    for run_id in run_ids:
        run = api.run(f"{project}/{run_id}")
        history = run.history()

        results_dict[run.name] = {
            "steps": history["step"].tolist(),
            "accuracies": history["accuracy"].tolist(),
            "regimes": history["regime"].tolist() if "regime" in history else [],
        }

    plot_accuracy_comparison(results_dict, output_path)


if __name__ == "__main__":
    import random

    print("Testing plotting with synthetic data...")

    # Create output directory
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    steps = list(range(1000))
    regimes = [i // 200 for i in steps]

    # Simulate different methods
    methods_data = {}

    # FLH - fast recovery
    flh_acc = []
    for i, regime in enumerate(regimes):
        base = 0.5 + 0.4 * min(1, (i % 200) / 50)
        flh_acc.append(base + random.gauss(0, 0.05))
    methods_data["FLH"] = {"steps": steps, "accuracies": flh_acc, "regimes": regimes}

    # Single prompt - slow recovery
    single_acc = []
    for i, regime in enumerate(regimes):
        base = 0.5 + 0.3 * min(1, (i % 200) / 150)
        single_acc.append(base + random.gauss(0, 0.05))
    methods_data["Single"] = {"steps": steps, "accuracies": single_acc, "regimes": regimes}

    # Random - noisy
    random_acc = [0.5 + random.gauss(0, 0.1) for _ in steps]
    methods_data["Random"] = {"steps": steps, "accuracies": random_acc, "regimes": regimes}

    # Test plots
    print("Generating accuracy plot...")
    plot_accuracy_over_time(steps, flh_acc, regimes, output_dir / "accuracy.png")

    print("Generating comparison plot...")
    plot_accuracy_comparison(methods_data, output_dir / "comparison.png")

    print("Generating entropy plot...")
    entropies = [0.5 + 0.5 * (i / 1000) + random.gauss(0, 0.05) for i in steps]
    plot_weight_entropy(steps, entropies, output_dir / "entropy.png")

    print("Generating weight heatmap...")
    weight_history = []
    num_prompts = 3
    for i in steps:
        weights = [random.random() for _ in range(num_prompts)]
        weights = [w / sum(weights) for w in weights]
        weight_history.append(weights)
    plot_weight_heatmap(steps, weight_history, output_dir / "weights.png")

    print(f"\n✓ Test plots saved to {output_dir}/")
