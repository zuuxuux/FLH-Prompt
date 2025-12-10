"""CLI for FLH-Prompts."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from flh_prompts.training.trainer import TrainConfig, train_flh

app = typer.Typer(
    name="flh",
    help="FLH-Prompts: Fixed-share Learned History Prompt Pooling for Continual Learning",
    add_completion=False,
)
console = Console()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.command()
def train(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to config file",
    ),
    method: str = typer.Option(
        "flh",
        "--method", "-m",
        help="Training method: flh (Fixed Share), hedge, trueflh, single, random, similarity",
    ),
    dataset: str = typer.Option(
        "sst2",
        "--dataset",
        help="Dataset: sst2 (label-flip) or amazon (domain rotation)",
    ),
    train_mode: str = typer.Option(
        "weighted_only",
        "--train-mode", "-t",
        help="Training mode: weighted_only, all_prompts",
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps", "-s",
        help="Total training steps (overrides config)",
    ),
    wandb_project: str = typer.Option(
        "flh-prompts",
        "--wandb-project", "-w",
        help="Wandb project name",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name", "-n",
        help="Wandb run name",
    ),
    alpha: Optional[float] = typer.Option(
        None,
        "--alpha", "-a",
        help="FLH alpha (temperature) parameter",
    ),
    birth_interval: Optional[int] = typer.Option(
        None,
        "--birth-interval", "-b",
        help="Steps between prompt births",
    ),
    flip_interval: Optional[int] = typer.Option(
        None,
        "--flip-interval", "-f",
        help="Steps between label flips (SST2 only)",
    ),
    steps_per_domain: Optional[int] = typer.Option(
        None,
        "--steps-per-domain",
        help="Steps per domain before switching (Amazon only)",
    ),
    amazon_domains: Optional[str] = typer.Option(
        None,
        "--amazon-domains",
        help="Comma-separated list of Amazon domains (default: all 5)",
    ),
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to use (cuda/cpu)",
    ),
):
    """Train a prompt method on the rotating sentiment task."""
    # Load base config
    config_path = Path(config)
    if config_path.exists():
        cfg_dict = load_config(config)
        console.print(f"[green]Loaded config from {config}[/green]")
    else:
        cfg_dict = {}
        console.print(f"[yellow]Config {config} not found, using defaults[/yellow]")

    # Apply CLI overrides
    if steps is not None:
        cfg_dict["total_steps"] = steps
    if alpha is not None:
        cfg_dict["alpha"] = alpha
    if birth_interval is not None:
        cfg_dict["birth_interval"] = birth_interval
    if flip_interval is not None:
        cfg_dict["flip_interval"] = flip_interval
    if steps_per_domain is not None:
        cfg_dict["steps_per_domain"] = steps_per_domain
    if amazon_domains is not None:
        cfg_dict["amazon_domains"] = [d.strip() for d in amazon_domains.split(",")]
    if lr is not None:
        cfg_dict["lr"] = lr

    cfg_dict["dataset"] = dataset
    cfg_dict["train_mode"] = train_mode
    cfg_dict["wandb_project"] = wandb_project
    cfg_dict["device"] = device
    if run_name:
        cfg_dict["wandb_run_name"] = run_name

    # Create config object
    train_config = TrainConfig(**{k: v for k, v in cfg_dict.items() if hasattr(TrainConfig, k)})

    # Display config
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for key, value in train_config.__dict__.items():
        table.add_row(key, str(value))

    console.print(table)

    # Run training based on method
    if method == "flh":
        console.print("\n[bold blue]Starting Fixed Share training...[/bold blue]")
        results = train_flh(train_config, pool_type="fixedshare")
    elif method == "hedge":
        console.print("\n[bold blue]Starting Hedge (pure multiplicative) training...[/bold blue]")
        results = train_flh(train_config, pool_type="hedge")
    elif method == "trueflh":
        console.print("\n[bold blue]Starting True FLH (adaptive regret) training...[/bold blue]")
        results = train_flh(train_config, pool_type="trueflh")
    elif method == "single":
        console.print("\n[bold blue]Starting single prompt baseline...[/bold blue]")
        from flh_prompts.baselines.single_prompt import train_single_prompt
        results = train_single_prompt(train_config)
    elif method == "random":
        console.print("\n[bold blue]Starting fixed random baseline...[/bold blue]")
        from flh_prompts.baselines.fixed_random import train_fixed_random
        results = train_fixed_random(train_config)
    elif method == "similarity":
        console.print("\n[bold blue]Starting input similarity baseline...[/bold blue]")
        from flh_prompts.baselines.input_similarity import train_input_similarity
        results = train_input_similarity(train_config)
    else:
        console.print(f"[red]Unknown method: {method}[/red]")
        raise typer.Exit(1)

    # Summary
    final_acc = results["accuracies"][-1] if results["accuracies"] else 0
    console.print(f"\n[bold green]Training complete! Final accuracy: {final_acc:.4f}[/bold green]")


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(
        ...,
        help="Path to checkpoint file",
    ),
    output: str = typer.Option(
        "results/",
        "--output", "-o",
        help="Output directory for plots",
    ),
):
    """Evaluate a trained model and generate plots."""
    import torch

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    console.print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint_path, weights_only=False)

    results = state.get("results", {})
    config = state.get("config", {})

    console.print(f"[green]Loaded checkpoint from step {state.get('step', 'unknown')}[/green]")
    console.print(f"Final accuracy: {results['accuracies'][-1]:.4f}")

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    from flh_prompts.evaluation.plotting import (
        plot_accuracy_over_time,
        plot_weight_entropy,
    )

    plot_accuracy_over_time(
        results["steps"],
        results["accuracies"],
        results["regimes"],
        output_dir / "accuracy.png",
    )
    console.print(f"[green]Saved accuracy plot to {output_dir / 'accuracy.png'}[/green]")

    if "weight_entropies" in results:
        plot_weight_entropy(
            results["steps"],
            results["weight_entropies"],
            output_dir / "entropy.png",
        )
        console.print(f"[green]Saved entropy plot to {output_dir / 'entropy.png'}[/green]")


@app.command()
def compare(
    run_ids: list[str] = typer.Argument(
        ...,
        help="Wandb run IDs to compare",
    ),
    output: str = typer.Option(
        "comparison.png",
        "--output", "-o",
        help="Output plot path",
    ),
):
    """Compare multiple runs from wandb."""
    import wandb
    from flh_prompts.evaluation.plotting import plot_comparison_from_wandb

    console.print(f"Fetching data for {len(run_ids)} runs...")
    plot_comparison_from_wandb(run_ids, output)
    console.print(f"[green]Saved comparison plot to {output}[/green]")


@app.command()
def info():
    """Show system information and verify setup."""
    import torch
    import transformers

    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PyTorch", torch.__version__)
    table.add_row("Transformers", transformers.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    console.print(table)


if __name__ == "__main__":
    app()
