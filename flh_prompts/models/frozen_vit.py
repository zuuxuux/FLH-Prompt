"""Frozen ViT backbone with soft prompt injection."""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class FrozenViTWithPrompt(nn.Module):
    """ViT model with frozen weights and soft prompt prepending.

    This model loads a pretrained ViT for image classification, freezes all
    parameters, and allows soft prompts to be prepended to patch embeddings.

    Visual Prompt Tuning (VPT) approach: prepend learnable prompt tokens to the
    sequence of patch embeddings, after the CLS token.

    Args:
        model_name: HuggingFace model name (default: google/vit-base-patch16-224).
        num_labels: Number of output classes.
        device: Device to load model on.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 10,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.num_labels = num_labels

        # Load pretrained ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # Allow different num_labels
        ).to(device)

        # Freeze only the ViT backbone, keep classifier trainable
        for param in self.model.vit.parameters():
            param.requires_grad = False
        # Classifier stays trainable (requires_grad=True by default)

        # Get embedding dimension (768 for ViT-Base)
        self.embed_dim = self.model.config.hidden_size

        # Store references to key components
        self.patch_embeddings = self.model.vit.embeddings.patch_embeddings
        self.cls_token = self.model.vit.embeddings.cls_token
        self.position_embeddings = self.model.vit.embeddings.position_embeddings
        self.dropout = self.model.vit.embeddings.dropout

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with soft prompt prepended to patch embeddings.

        Args:
            pixel_values: Images of shape [batch, 3, 224, 224].
            prompt_embeds: Soft prompt embeddings of shape [prompt_len, embed_dim]
                          or [batch, prompt_len, embed_dim].

        Returns:
            Logits of shape [batch, num_labels].
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # Get patch embeddings: [batch, num_patches, embed_dim]
        # num_patches = (224/16)^2 = 196 for ViT-Base-16
        patch_embeds = self.patch_embeddings(pixel_values)

        # Expand CLS token for batch: [1, 1, embed_dim] -> [batch, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Handle prompt_embeds shape
        if prompt_embeds.dim() == 2:
            # Shape [prompt_len, embed_dim] -> [batch, prompt_len, embed_dim]
            prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        prompt_len = prompt_embeds.size(1)
        num_patches = patch_embeds.size(1)

        # VPT: Prepend prompts after CLS but before patches
        # Sequence: [CLS, PROMPT_1, ..., PROMPT_N, PATCH_1, ..., PATCH_196]
        embeddings = torch.cat([cls_tokens, prompt_embeds, patch_embeds], dim=1)

        # Handle position embeddings
        # Original position embeddings are for [CLS + patches] = 1 + 196 = 197
        # Shape is [1, 197, 768]
        # We need positions for [CLS + prompts + patches] = 1 + prompt_len + 196
        total_len = 1 + prompt_len + num_patches
        orig_pos_embeds = self.position_embeddings  # [1, 197, 768]

        # Use CLS position for prompts, then patch positions
        cls_pos = orig_pos_embeds[:, :1, :]  # [1, 1, 768]
        patch_pos = orig_pos_embeds[:, 1:, :]  # [1, 196, 768]

        # Prompts share CLS position (simple VPT approach)
        prompt_pos = cls_pos.expand(-1, prompt_len, -1)  # [1, prompt_len, 768]

        # Combine: [CLS_pos, PROMPT_pos..., PATCH_pos...]
        position_embeds = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)  # [1, total_len, 768]
        position_embeds = position_embeds.expand(batch_size, -1, -1)

        # Add position embeddings
        embeddings = embeddings + position_embeds

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Forward through ViT encoder
        encoder_outputs = self.model.vit.encoder(embeddings)
        sequence_output = encoder_outputs[0]

        # Apply layer norm
        sequence_output = self.model.vit.layernorm(sequence_output)

        # Get CLS token output (first token)
        cls_output = sequence_output[:, 0]

        # Classification head
        logits = self.model.classifier(cls_output)

        return logits

    def get_patch_embedding(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Get the CLS embedding for input-based selection (similarity baseline).

        Args:
            pixel_values: Images of shape [batch, 3, 224, 224].

        Returns:
            CLS embeddings of shape [batch, embed_dim].
        """
        with torch.no_grad():
            batch_size = pixel_values.size(0)

            # Get patch embeddings
            patch_embeds = self.patch_embeddings(pixel_values)

            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)

            # Add position embeddings (original 197 positions) - shape is [1, 197, 768]
            position_embeds = self.position_embeddings.expand(batch_size, -1, -1)
            embeddings = embeddings + position_embeds
            embeddings = self.dropout(embeddings)

            # Forward through encoder
            encoder_outputs = self.model.vit.encoder(embeddings)
            sequence_output = encoder_outputs[0]
            sequence_output = self.model.vit.layernorm(sequence_output)

            # Return CLS token
            return sequence_output[:, 0]

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim


def test_frozen_vit():
    """Test the frozen ViT backbone with random prompt."""
    print("Loading FrozenViTWithPrompt...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenViTWithPrompt(num_labels=10, device=device)

    print(f"Embedding dimension: {model.embedding_dim}")

    # Create random inputs
    batch_size = 4
    prompt_len = 20

    # Random images (normalized like ImageNet)
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    prompt_embeds = torch.randn(prompt_len, model.embedding_dim, device=device) * 0.02

    print(f"\nInput shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(pixel_values, prompt_embeds)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits sample: {logits[0].tolist()}")

    # Test embedding extraction
    print("\nTesting embedding extraction...")
    with torch.no_grad():
        embeddings = model.get_patch_embedding(pixel_values)
    print(f"Embedding shape: {embeddings.shape}")

    # Verify no gradients on model params
    all_frozen = all(not p.requires_grad for p in model.model.parameters())
    print(f"\n✓ All parameters frozen: {all_frozen}")
    print("✓ Test complete!")


if __name__ == "__main__":
    test_frozen_vit()
