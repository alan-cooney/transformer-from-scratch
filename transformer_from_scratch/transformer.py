"""Transformer."""
from torch import nn
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.components.embed_unembed import Embed, Unembed
from transformer_from_scratch.components.layer import Layer
from transformer_from_scratch.components.positional_encoding import (
    SinusoidalPositionalEncoding,
)
from transformer_from_scratch.types import (
    BatchLogits,
    BatchResidualStream,
    BatchTokenIndices,
)


class Transformer(nn.Module):
    """Encoder Only Transformer."""

    def __init__(self, config: TransformerConfig = TransformerConfig()) -> None:
        """Initialise the Transformer."""
        super().__init__()

        # Embedding and unembedding
        self.embed = Embed(config)
        self.unembed = Unembed(config)

        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(config)

        # Layers
        self.layers = nn.ModuleList([])
        for _layer_idx in range(config.n_layers):
            self.layers.append(Layer(config))

        # Expose config as public attribute
        self.config = config

    def forward(self, tokens: BatchTokenIndices) -> BatchLogits:
        """Forward pass.

        Args:
            tokens (BatchTokenIndices): Input tokens (indices rather than one-hot)

        Returns:
            BatchLogits: Logits representing log probabilities for the tokens
        """
        # Embed + positional encoding
        residual_stream: BatchResidualStream = self.embed(tokens)
        residual_stream = self.positional_encoding(residual_stream)

        # Loop through layers
        for layer in self.layers:
            residual_stream = layer(residual_stream)

        # Unembed and return
        return self.unembed(residual_stream)
