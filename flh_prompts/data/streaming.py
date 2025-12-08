"""Streaming data pipeline with label flipping for continual learning experiments."""

from typing import Iterator, Literal

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# Multi-domain sentiment datasets from different sources
# Each entry: (dataset_name, config, text_field, label_field, is_binary_sentiment, pos_labels)
# pos_labels: if is_binary_sentiment=False, these label values map to positive (1)
DOMAIN_CONFIGS = {
    "imdb": {
        "dataset": "stanfordnlp/imdb",
        "config": None,
        "text_field": "text",
        "label_field": "label",
        "is_binary": True,  # 0=neg, 1=pos already
    },
    "yelp": {
        "dataset": "Yelp/yelp_review_full",
        "config": None,
        "text_field": "text",
        "label_field": "label",
        "is_binary": False,  # 0-4 labels (1-5 stars)
        "pos_labels": [3, 4],  # 4-5 stars = positive
        "neg_labels": [0, 1],  # 1-2 stars = negative
    },
    "apps": {
        "dataset": "app_reviews",
        "config": None,
        "text_field": "review",
        "label_field": "star",
        "is_binary": False,  # 1-5 stars
        "pos_labels": [4, 5],  # 4-5 stars = positive
        "neg_labels": [1, 2],  # 1-2 stars = negative
    },
    "rotten_tomatoes": {
        "dataset": "cornell-movie-review-data/rotten_tomatoes",
        "config": None,
        "text_field": "text",
        "label_field": "label",
        "is_binary": True,  # 0=neg, 1=pos already
    },
    "sst2": {
        "dataset": "stanfordnlp/sst2",
        "config": None,
        "text_field": "sentence",
        "label_field": "label",
        "is_binary": True,  # 0=neg, 1=pos already
    },
}

# Default domains to use (diverse vocabulary)
SENTIMENT_DOMAINS = ["imdb", "yelp", "apps", "rotten_tomatoes", "sst2"]

# Human-readable names for logging
DOMAIN_DISPLAY_NAMES = {
    "imdb": "Movies (IMDB)",
    "yelp": "Business (Yelp)",
    "apps": "Apps & Tech",
    "rotten_tomatoes": "Movies (RT)",
    "sst2": "Sentences (SST-2)",
}

# Export both new and old names for backwards compatibility
AMAZON_DOMAINS = SENTIMENT_DOMAINS  # Alias for backwards compatibility
AMAZON_DOMAIN_NAMES = DOMAIN_DISPLAY_NAMES  # Alias for backwards compatibility


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


class StreamingAmazonMultiDomain:
    """Infinite streaming data loader with domain rotation for continual learning.

    This loader cycles through diverse sentiment analysis domains and yields
    batches from the current domain only. Unlike label-flip, the *input
    distribution* changes while the task (binary sentiment) stays constant.

    Available domains:
        - imdb: Movie reviews from IMDB
        - yelp: Business reviews from Yelp
        - apps: App store reviews
        - rotten_tomatoes: Movie reviews from Rotten Tomatoes
        - sst2: Stanford Sentiment Treebank sentences

    Sentiment labels (all normalized to binary):
        - Negative (0): 1-2 star reviews or negative labels
        - Positive (1): 4-5 star reviews or positive labels
        - Discarded: 3 star/neutral reviews

    Args:
        steps_per_domain: Number of steps to spend in each domain before switching.
        batch_size: Batch size for data loading.
        tokenizer_name: Name of the tokenizer to use.
        max_length: Maximum sequence length for tokenization.
        domains: List of domain names to cycle through. Defaults to all 5.
        seed: Random seed for shuffling.
        balance_labels: Whether to balance positive/negative samples within domains.
    """

    def __init__(
        self,
        steps_per_domain: int = 1000,
        batch_size: int = 32,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        domains: list[str] | None = None,
        seed: int = 42,
        balance_labels: bool = True,
    ):
        self.steps_per_domain = steps_per_domain
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.balance_labels = balance_labels

        # Validate and set domains
        self.domains = domains or SENTIMENT_DOMAINS
        for domain in self.domains:
            if domain not in DOMAIN_CONFIGS:
                raise ValueError(
                    f"Unknown domain: {domain}. Valid domains: {list(DOMAIN_CONFIGS.keys())}"
                )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load and prepare all domain datasets
        print(f"Loading sentiment data for {len(self.domains)} domains...")
        self.domain_datasets = {}
        self.domain_text_fields = {}
        for domain in self.domains:
            self._load_domain(domain)

        # State tracking
        self.global_step = 0
        self.current_domain_idx = 0
        self.current_regime = 0  # Increments each time domain changes

    def _load_domain(self, domain: str) -> None:
        """Load and prepare a single domain's dataset."""
        config = DOMAIN_CONFIGS[domain]
        display_name = DOMAIN_DISPLAY_NAMES.get(domain, domain)
        print(f"  Loading {display_name}...")

        # Load the dataset
        dataset = load_dataset(
            config["dataset"],
            config["config"],
            split="train",
        )

        # Store the text field name for this domain
        self.domain_text_fields[domain] = config["text_field"]

        # Convert to binary sentiment if needed
        if config["is_binary"]:
            # Already binary, just rename label field if needed
            if config["label_field"] != "label":
                dataset = dataset.rename_column(config["label_field"], "label")
        else:
            # Need to convert to binary
            pos_labels = set(config["pos_labels"])
            neg_labels = set(config["neg_labels"])

            def convert_to_binary(example):
                rating = example[config["label_field"]]
                if rating in pos_labels:
                    return {"label": 1, "keep": True}
                elif rating in neg_labels:
                    return {"label": 0, "keep": True}
                else:
                    return {"label": -1, "keep": False}

            dataset = dataset.map(convert_to_binary)
            dataset = dataset.filter(lambda x: x["keep"])

        # Shuffle dataset
        dataset = dataset.shuffle(seed=self.seed)

        if self.balance_labels:
            # Balance positive and negative samples
            pos_samples = dataset.filter(lambda x: x["label"] == 1)
            neg_samples = dataset.filter(lambda x: x["label"] == 0)

            # Use the smaller class size for both
            min_size = min(len(pos_samples), len(neg_samples))
            pos_samples = pos_samples.select(range(min_size))
            neg_samples = neg_samples.select(range(min_size))

            # Interleave for balanced batches
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([pos_samples, neg_samples])
            dataset = dataset.shuffle(seed=self.seed)

            print(f"    {display_name}: {min_size} pos + {min_size} neg = {len(dataset)} samples")
        else:
            print(f"    {display_name}: {len(dataset)} samples")

        self.domain_datasets[domain] = dataset

    @property
    def current_domain(self) -> str:
        """Get the current domain name."""
        return self.domains[self.current_domain_idx]

    @property
    def current_domain_display(self) -> str:
        """Get the human-readable current domain name."""
        return DOMAIN_DISPLAY_NAMES.get(self.current_domain, self.current_domain)

    def _tokenize_batch(self, texts: list[str]) -> dict:
        """Tokenize a batch of review texts."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __iter__(self) -> Iterator[dict]:
        """Yield batches infinitely, rotating through domains."""
        while True:
            domain = self.current_domain
            dataset = self.domain_datasets[domain]
            text_field = self.domain_text_fields[domain]
            indices = list(range(len(dataset)))

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    # Skip incomplete batches at end of epoch
                    continue

                # Get batch of examples
                batch_examples = dataset.select(batch_indices)

                # Extract texts and labels using domain-specific text field
                texts = [
                    batch_examples[j][text_field] or ""
                    for j in range(len(batch_examples))
                ]
                labels_list = [
                    batch_examples[j]["label"] for j in range(len(batch_examples))
                ]

                # Tokenize
                tokenized = self._tokenize_batch(texts)

                # Convert labels to tensor
                labels = torch.tensor(labels_list, dtype=torch.long)

                # Check if we should switch domains BEFORE yielding
                if (
                    self.global_step > 0
                    and self.global_step % self.steps_per_domain == 0
                ):
                    self.current_domain_idx = (self.current_domain_idx + 1) % len(
                        self.domains
                    )
                    self.current_regime += 1

                yield {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                    "step": self.global_step,
                    "regime": self.current_regime,
                    "domain": self.current_domain,
                    "domain_idx": self.current_domain_idx,
                    "domain_display": self.current_domain_display,
                }

                self.global_step += 1

    def reset(self):
        """Reset the streamer to initial state."""
        self.global_step = 0
        self.current_domain_idx = 0
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


def test_amazon_streaming():
    """Test the multi-domain streaming data loader."""
    print("Loading StreamingAmazonMultiDomain with 2 domains for testing...")

    # Use only 2 domains for faster testing (imdb for movies, yelp for business)
    streamer = StreamingAmazonMultiDomain(
        steps_per_domain=100,
        batch_size=32,
        domains=["imdb", "yelp"],
    )

    print(f"\nDomains: {streamer.domains}")
    print(f"Steps per domain: {streamer.steps_per_domain}")
    print("\nStreaming 250 steps, checking domain switches...")

    domain_counts = {}
    for batch in streamer:
        step = batch["step"]
        domain = batch["domain_display"]

        # Count samples per domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Print info at critical steps (around domain switches)
        if step in [0, 99, 100, 101, 199, 200, 201]:
            print(f"\nStep {step}:")
            print(f"  Domain: {domain}")
            print(f"  Domain idx: {batch['domain_idx']}")
            print(f"  Regime: {batch['regime']}")
            print(f"  Labels distribution: {batch['labels'].tolist()[:5]}...")

        if step >= 250:
            break

    print("\n\nDomain sample counts:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} batches")

    print("\n✓ Amazon streaming test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "amazon":
        test_amazon_streaming()
    else:
        test_streaming()
