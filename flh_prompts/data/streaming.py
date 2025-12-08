"""Streaming data pipeline with label flipping for continual learning experiments."""

from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class StreamingSST2:
    """Infinite streaming data loader for SST-2 with periodic label flipping.

    This loader cycles through SST-2 indefinitely and flips labels (0↔1) at
    specified intervals to simulate distribution shifts.

    Args:
        flip_every_n: Number of steps between label flips (regime changes).
        batch_size: Batch size for data loading.
        tokenizer_name: Name of the tokenizer to use.
        max_length: Maximum sequence length for tokenization.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        flip_every_n: int = 1000,
        batch_size: int = 32,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        seed: int = 42,
    ):
        self.flip_every_n = flip_every_n
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load and prepare dataset
        dataset = load_dataset("stanfordnlp/sst2", split="train")
        self.dataset = dataset.shuffle(seed=seed)

        # State tracking
        self.global_step = 0
        self.current_regime = 0  # Even = normal, Odd = flipped

    @property
    def should_flip(self) -> bool:
        """Whether labels should be flipped in the current regime."""
        return self.current_regime % 2 == 1

    def _tokenize_batch(self, examples: dict) -> dict:
        """Tokenize a batch of examples."""
        return self.tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _maybe_flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Flip labels if in an odd regime."""
        if self.should_flip:
            return 1 - labels
        return labels

    def __iter__(self) -> Iterator[dict]:
        """Yield batches infinitely, tracking steps and flipping labels."""
        while True:
            # Create a dataloader that will be cycled through
            indices = list(range(len(self.dataset)))

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    # Skip incomplete batches at end of epoch
                    continue

                # Get batch of examples
                batch_examples = self.dataset.select(batch_indices)

                # Extract sentences and labels as lists
                sentences = [batch_examples[j]["sentence"] for j in range(len(batch_examples))]
                labels_list = [batch_examples[j]["label"] for j in range(len(batch_examples))]

                # Tokenize
                tokenized = self._tokenize_batch({"sentence": sentences})

                # Get and potentially flip labels
                labels = torch.tensor(labels_list, dtype=torch.long)
                labels = self._maybe_flip_labels(labels)

                # Update regime if needed (before yielding so logging is accurate)
                if self.global_step > 0 and self.global_step % self.flip_every_n == 0:
                    self.current_regime += 1

                yield {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                    "step": self.global_step,
                    "regime": self.current_regime,
                    "flipped": self.should_flip,
                }

                self.global_step += 1

    def reset(self):
        """Reset the streamer to initial state."""
        self.global_step = 0
        self.current_regime = 0


def test_streaming():
    """Test the streaming data loader with flip verification."""
    print("Loading StreamingSST2...")
    streamer = StreamingSST2(flip_every_n=1000, batch_size=32)

    print("\nStreaming 5000 samples, checking flips at steps 999, 1000, 1001...")

    for batch in streamer:
        step = batch["step"]

        # Print info at critical steps
        if step in [999, 1000, 1001]:
            print(f"\nStep {step}:")
            print(f"  Regime: {batch['regime']}")
            print(f"  Flipped: {batch['flipped']}")
            print(f"  First 5 labels: {batch['labels'][:5].tolist()}")

        if step >= 5000:
            break

    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_streaming()
