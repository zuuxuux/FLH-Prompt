"""Frozen BERT backbone with soft prompt injection."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class FrozenBERTWithPrompt(nn.Module):
    """BERT model with frozen weights and soft prompt prepending.

    This model loads a pretrained BERT for sequence classification, freezes all
    parameters, and allows soft prompts to be prepended to input embeddings.

    Args:
        model_name: HuggingFace model name (default: bert-base-uncased).
        num_labels: Number of output classes.
        device: Device to load model on.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        # Load model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ).to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.embed_dim = self.model.config.hidden_size

        # Get the embedding layer for input embedding lookup
        self.embeddings = self.model.bert.embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with soft prompt prepended to inputs.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            prompt_embeds: Soft prompt embeddings of shape [prompt_len, embed_dim]
                          or [batch, prompt_len, embed_dim].

        Returns:
            Logits of shape [batch, num_labels].
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Get input embeddings from token IDs
        input_embeds = self.embeddings.word_embeddings(input_ids)

        # Handle prompt_embeds shape
        if prompt_embeds.dim() == 2:
            # Shape [prompt_len, embed_dim] -> [batch, prompt_len, embed_dim]
            prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        prompt_len = prompt_embeds.size(1)

        # Prepend prompt embeddings to input embeddings
        # [batch, prompt_len + seq_len, embed_dim]
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # Extend attention mask to cover prompt tokens
        # Prompt tokens should always be attended to (mask = 1)
        prompt_mask = torch.ones(batch_size, prompt_len, device=device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Create position IDs for the combined sequence
        seq_length = combined_embeds.size(1)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

        # Add position embeddings and other embedding components
        position_embeds = self.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
        token_type_embeds = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = combined_embeds + position_embeds + token_type_embeds
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)

        # Forward through BERT encoder
        encoder_outputs = self.model.bert.encoder(
            embeddings,
            attention_mask=self._get_extended_attention_mask(extended_attention_mask),
        )

        # Get the [CLS] token representation (first token after prompt)
        # Actually, after prepending prompt, CLS is still at position 0 of the original input
        # But we prepended prompt, so CLS is at position prompt_len
        # However, BERT's pooler expects the first token, so we use position 0 of our combined sequence
        sequence_output = encoder_outputs[0]

        # Use the first token (which is the first prompt token now) or we could use CLS position
        # For classification, we'll use the pooler on the sequence output
        pooled_output = self.model.bert.pooler(sequence_output)

        # Classification head
        output = self.model.dropout(pooled_output)
        logits = self.model.classifier(output)

        return logits

    def _get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert attention mask to the format expected by BERT encoder."""
        # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        extended_attention_mask = attention_mask[:, None, None, :]
        # Convert to attention scores format (0 -> -inf, 1 -> 0)
        extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
        return extended_attention_mask

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim


def test_frozen_backbone():
    """Test the frozen backbone with random prompt."""
    print("Loading FrozenBERTWithPrompt...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenBERTWithPrompt(device=device)

    print(f"Embedding dimension: {model.embedding_dim}")

    # Create random inputs
    batch_size = 4
    seq_len = 32
    prompt_len = 20

    input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    prompt_embeds = torch.randn(prompt_len, model.embedding_dim, device=device) * 0.02

    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask, prompt_embeds)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits sample: {logits[0].tolist()}")

    # Verify no gradients on model params
    all_frozen = all(not p.requires_grad for p in model.model.parameters())
    print(f"\n✓ All parameters frozen: {all_frozen}")
    print("✓ Test complete!")


if __name__ == "__main__":
    test_frozen_backbone()
