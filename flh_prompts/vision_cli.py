"""CLI for FLH-Prompts Vision experiments."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from flh_prompts.training.vision_trainer import (
    VisionTrainConfig,
    train_flh_vision,
    train_single_prompt_vision,
    train_random_pool_vision,
    train_similarity_vision,
)

app = typer.Typer(
    name="flh-vision",
    help="FLH-Prompts Vision: ViT + CIFAR-10 experiments",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    method: str = typer.Option(
        "flh",
        "--method", "-m",
        help="Training method: flh, single, random, similarity",
    ),
    dataset: str = typer.Option(
        "cifar10",
        "--dataset",
        help="Dataset: cifar10 (10-class with rotation) or cifar10_binary (2-class with flip)",
    ),
    steps: int = typer.Option(
        10000,
        "--steps", "-s",
        help="Total training steps",
    ),
    rotate_interval: int = typer.Option(
        1000,
        "--rotate-interval", "-r",
        help="Steps between label rotations/flips",
    ),
    birth_interval: int = typer.Option(
        500,
        "--birth-interval", "-b",
        help="Steps between prompt births (FLH only)",
    ),
    alpha: float = typer.Option(
        0.1,
        "--alpha", "-a",
        help="FLH alpha (temperature) parameter",
    ),
    num_prompts: int = typer.Option(
        10,
        "--num-prompts", "-n",
        help="Number of prompts (random/similarity only)",
    ),
    lr: float = typer.Option(
        0.001,
        "--lr",
        help="Learning rate",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size",
    ),
    wandb_project: str = typer.Option(
        "flh-prompts",
        "--wandb-project", "-w",
        help="Wandb project name",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Wandb run name",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to use (cuda/cpu)",
    ),
):
    """Train a prompt method on CIFAR-10 with rotating labels."""
    # Determine num_classes based on dataset
    num_classes = 2 if dataset == "cifar10_binary" else 10

    # Create config
    config = VisionTrainConfig(
        dataset=dataset,
        num_classes=num_classes,
        total_steps=steps,
        rotate_interval=rotate_interval,
        birth_interval=birth_interval,
        alpha=alpha,
        lr=lr,
        batch_size=batch_size,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
        device=device,
    )

    # Display config
    table = Table(title="Vision Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Method", method)
    for key, value in config.__dict__.items():
        table.add_row(key, str(value))
    if method in ["random", "similarity"]:
        table.add_row("num_prompts", str(num_prompts))

    console.print(table)

    # Run training based on method
    if method == "flh":
        console.print("\n[bold blue]Starting FLH Vision training...[/bold blue]")
        results = train_flh_vision(config)
    elif method == "single":
        console.print("\n[bold blue]Starting single prompt vision baseline...[/bold blue]")
        results = train_single_prompt_vision(config)
    elif method == "random":
        console.print("\n[bold blue]Starting random pool vision baseline...[/bold blue]")
        results = train_random_pool_vision(config, num_prompts=num_prompts)
    elif method == "similarity":
        console.print("\n[bold blue]Starting input similarity vision baseline...[/bold blue]")
        results = train_similarity_vision(config, num_prompts=num_prompts)
    else:
        console.print(f"[red]Unknown method: {method}[/red]")
        raise typer.Exit(1)

    # Summary
    final_acc = results["accuracies"][-1] if results["accuracies"] else 0
    console.print(f"\n[bold green]Training complete! Final accuracy: {final_acc:.4f}[/bold green]")


@app.command()
def info():
    """Show system information for vision experiments."""
    import torch
    import transformers

    table = Table(title="Vision System Information")
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
