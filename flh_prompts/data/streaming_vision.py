"""Streaming vision data pipeline with label rotation for continual learning experiments."""

from typing import Iterator

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class StreamingCIFAR10:
    """Infinite streaming data loader for CIFAR-10 with periodic label rotation.

    This loader cycles through CIFAR-10 indefinitely and rotates labels at
    specified intervals to simulate distribution shifts. Label rotation
    shuffles which class IDs map to which actual classes.

    For example, with rotation_type="shift":
        Regime 0: class 0 = airplane, class 1 = automobile, ...
        Regime 1: class 0 = automobile, class 1 = bird, ...
        Regime 2: class 0 = bird, class 1 = cat, ...

    This forces the model to learn that the same visual input can have
    different labels in different regimes - testing adaptation to
    changing label semantics.

    Args:
        rotate_every_n: Number of steps between label rotations (regime changes).
        batch_size: Batch size for data loading.
        image_size: Size to resize images to (default 224 for ViT).
        num_classes: Number of CIFAR-10 classes (default 10).
        rotation_type: How to rotate labels - "shift" or "swap_pairs".
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        rotate_every_n: int = 1000,
        batch_size: int = 32,
        image_size: int = 224,
        num_classes: int = 10,
        rotation_type: str = "shift",
        seed: int = 42,
    ):
        self.rotate_every_n = rotate_every_n
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.rotation_type = rotation_type
        self.seed = seed

        # Set random seed
        torch.manual_seed(seed)

        # Image transforms: resize to 224x224 for ViT and normalize
        # ViT expects ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load CIFAR-10 training set
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=self.transform,
        )

        # State tracking
        self.global_step = 0
        self.current_regime = 0  # Increments each rotation

        # Create label permutation for current regime
        self._update_label_permutation()

    def _update_label_permutation(self) -> None:
        """Update the label permutation based on current regime and rotation type."""
        if self.rotation_type == "shift":
            # Circular shift: each regime shifts labels by 1
            shift = self.current_regime % self.num_classes
            self.label_permutation = [(i + shift) % self.num_classes for i in range(self.num_classes)]
        elif self.rotation_type == "swap_pairs":
            # Swap adjacent pairs each regime
            perm = list(range(self.num_classes))
            if self.current_regime % 2 == 1:
                # Swap pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
                for i in range(0, self.num_classes - 1, 2):
                    perm[i], perm[i + 1] = perm[i + 1], perm[i]
            self.label_permutation = perm
        else:
            # No permutation
            self.label_permutation = list(range(self.num_classes))

    def _permute_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply the current label permutation."""
        # Map each label through the permutation
        permuted = torch.zeros_like(labels)
        for i, new_label in enumerate(self.label_permutation):
            permuted[labels == i] = new_label
        return permuted

    def __iter__(self) -> Iterator[dict]:
        """Yield batches infinitely, tracking steps and rotating labels."""
        # Create a shuffled index
        generator = torch.Generator().manual_seed(self.seed)

        while True:
            # Shuffle indices for each epoch
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    # Skip incomplete batches at end of epoch
                    continue

                # Collect batch
                images = []
                labels = []
                for idx in batch_indices:
                    img, label = self.dataset[idx]
                    images.append(img)
                    labels.append(label)

                # Stack into tensors
                pixel_values = torch.stack(images)  # [batch, 3, 224, 224]
                labels_tensor = torch.tensor(labels, dtype=torch.long)

                # Apply label permutation
                labels_tensor = self._permute_labels(labels_tensor)

                # Check if we should rotate BEFORE yielding
                if self.global_step > 0 and self.global_step % self.rotate_every_n == 0:
                    self.current_regime += 1
                    self._update_label_permutation()

                yield {
                    "pixel_values": pixel_values,
                    "labels": labels_tensor,
                    "step": self.global_step,
                    "regime": self.current_regime,
                    "label_permutation": self.label_permutation.copy(),
                }

                self.global_step += 1

    def reset(self):
        """Reset the streamer to initial state."""
        self.global_step = 0
        self.current_regime = 0
        self._update_label_permutation()


class StreamingCIFAR10Binary:
    """Streaming CIFAR-10 with binary classification and label flipping.

    Similar to StreamingSST2 but for images. Groups CIFAR-10 classes into
    two super-classes and flips labels at intervals.

    Super-classes:
        - Vehicles (0): airplane, automobile, ship, truck
        - Animals (1): bird, cat, deer, dog, frog, horse

    Args:
        flip_every_n: Number of steps between label flips.
        batch_size: Batch size for data loading.
        image_size: Size to resize images to (default 224 for ViT).
        seed: Random seed for shuffling.
    """

    # Class indices for each super-class
    VEHICLE_CLASSES = {0, 1, 8, 9}  # airplane, automobile, ship, truck
    ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse

    def __init__(
        self,
        flip_every_n: int = 1000,
        batch_size: int = 32,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.flip_every_n = flip_every_n
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed

        torch.manual_seed(seed)

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load CIFAR-10
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=self.transform,
        )

        # State tracking
        self.global_step = 0
        self.current_regime = 0

    @property
    def should_flip(self) -> bool:
        """Whether labels should be flipped in the current regime."""
        return self.current_regime % 2 == 1

    def _to_binary_label(self, label: int) -> int:
        """Convert CIFAR-10 class to binary (vehicles=0, animals=1)."""
        if label in self.VEHICLE_CLASSES:
            return 0
        return 1

    def _maybe_flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Flip labels if in an odd regime."""
        if self.should_flip:
            return 1 - labels
        return labels

    def __iter__(self) -> Iterator[dict]:
        """Yield batches infinitely, tracking steps and flipping labels."""
        generator = torch.Generator().manual_seed(self.seed)

        while True:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue

                images = []
                labels = []
                for idx in batch_indices:
                    img, label = self.dataset[idx]
                    images.append(img)
                    labels.append(self._to_binary_label(label))

                pixel_values = torch.stack(images)
                labels_tensor = torch.tensor(labels, dtype=torch.long)

                # Apply flip
                labels_tensor = self._maybe_flip_labels(labels_tensor)

                # Update regime if needed
                if self.global_step > 0 and self.global_step % self.flip_every_n == 0:
                    self.current_regime += 1

                yield {
                    "pixel_values": pixel_values,
                    "labels": labels_tensor,
                    "step": self.global_step,
                    "regime": self.current_regime,
                    "flipped": self.should_flip,
                }

                self.global_step += 1

    def reset(self):
        """Reset the streamer to initial state."""
        self.global_step = 0
        self.current_regime = 0


def test_cifar10_streaming():
    """Test the CIFAR-10 streaming data loader."""
    print("Loading StreamingCIFAR10...")
    streamer = StreamingCIFAR10(rotate_every_n=100, batch_size=32, rotation_type="shift")

    print(f"\nRotation type: {streamer.rotation_type}")
    print(f"Rotate every: {streamer.rotate_every_n} steps")
    print("\nStreaming 250 steps, checking rotations...")

    for batch in streamer:
        step = batch["step"]

        # Print info at critical steps
        if step in [0, 99, 100, 101, 199, 200, 201]:
            print(f"\nStep {step}:")
            print(f"  Regime: {batch['regime']}")
            print(f"  Label permutation: {batch['label_permutation']}")
            print(f"  First 5 labels: {batch['labels'][:5].tolist()}")
            print(f"  Pixel values shape: {batch['pixel_values'].shape}")

        if step >= 250:
            break

    print("\n✓ CIFAR-10 streaming test complete!")


def test_cifar10_binary():
    """Test the binary CIFAR-10 streaming data loader."""
    print("\nLoading StreamingCIFAR10Binary...")
    streamer = StreamingCIFAR10Binary(flip_every_n=100, batch_size=32)

    print(f"Flip every: {streamer.flip_every_n} steps")
    print("\nStreaming 250 steps, checking flips...")

    label_counts = {0: 0, 1: 0}
    for batch in streamer:
        step = batch["step"]

        # Count labels
        for label in batch["labels"].tolist():
            label_counts[label] += 1

        # Print info at critical steps
        if step in [0, 99, 100, 101, 199, 200, 201]:
            print(f"\nStep {step}:")
            print(f"  Regime: {batch['regime']}")
            print(f"  Flipped: {batch['flipped']}")
            print(f"  First 5 labels: {batch['labels'][:5].tolist()}")

        if step >= 250:
            break

    print(f"\nLabel distribution: {label_counts}")
    print("\n✓ Binary CIFAR-10 streaming test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "binary":
        test_cifar10_binary()
    else:
        test_cifar10_streaming()
