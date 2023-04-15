"""Transformer."""
from torch import nn

from transformer_from_scratch.embed_unembed import Embed, Unembed
from transformer_from_scratch.layer import Layer
from transformer_from_scratch.positional_encoding import PositionalEncoding
from transformer_from_scratch.types import (
    BatchLogitsTT,
    BatchResidualStreamTT,
    BatchTokenIndicesTT,
)


class Transformer(nn.Module):
    """Transformer.

    Note that unlike the original paper, this uses an encoder-only architecture
    instead of encoder-decoder. This is because the original paper was focused
    on language translation whereas this model will be trained on different
    tasks.

    Default params are set based on GPT-2 small.
    """

    def __init__(
        self,
        d_head: int = 64,
        d_hidden: int = 2048,
        d_model: int = 768,
        d_vocab: int = 50432,  # Default of GptNeoX Vocab Size
        max_tokens: int = 1024,
        n_layers: int = 12,
    ) -> None:
        """Initialise the transformer.

        Args:
            d_head: Attention head dimension
            d_hidden: MLP dimension
            d_model: Model (residual stream) dimension
            d_vocab: Vocab size
            max_tokens: Maximum tokens per prompt
            n_layers: Number of layers (attention + mlp = 1 layer)
        """
        super().__init__()

        # Embedding and unembedding
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_tokens=max_tokens,
        )

        # Layers
        self.layers = nn.ModuleList([])
        for _layer_idx in range(n_layers):
            self.layers.append(Layer(d_model=d_model, d_head=d_head, d_hidden=d_hidden))

    def forward(self, tokens: BatchTokenIndicesTT) -> BatchLogitsTT:
        """Forward."""
        # Embed + positional encoding
        residual_stream: BatchResidualStreamTT = self.embed(tokens)
        residual_stream = self.positional_encoding(residual_stream)

        # Loop through layers
        for layer in self.layers:
            residual_stream = layer(residual_stream)

        # Unembed and return
        return self.unembed(residual_stream)
