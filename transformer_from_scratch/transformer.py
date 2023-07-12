"""Transformer."""
from torch import nn

from transformer_from_scratch.components.embed_unembed import Embed, Unembed
from transformer_from_scratch.components.layer import Layer
from transformer_from_scratch.components.positional_encoding import (
    SinusoidalPositionalEncoding,
)
from transformer_from_scratch.types import (
    BatchLogitsTT,
    BatchResidualStreamTT,
    BatchTokenIndicesTT,
)


class Transformer(nn.Module):
    """Encoder Only Transformer."""

    def __init__(
        self,
        d_head: int = 64,
        d_hidden: int = 2048,
        d_model: int = 768,
        d_vocab: int = 50432,
        max_tokens: int = 1024,
        n_layers: int = 12,
    ) -> None:
        """Initialise the Transformer.

        Default args are taken from the Attention is All You Need paper, with the exception of
        d_vocab which defaults to the GPT-NeoX vocab size.

        Args:
            d_head (int, optional): Number of head features per token.
            d_hidden (int, optional): Number of hidden layer features per token.
            d_model (int): Number of residual stream features per token.
            d_vocab: Number of tokens in the vocabulary.
            max_tokens: Maximum number of tokens in a sequence.
            n_layers: Number of layers (attention + mlp = 1 layer) in the architecture.
        """
        super().__init__()

        # Embedding and unembedding
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)

        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_tokens=max_tokens,
        )

        # Layers
        self.layers = nn.ModuleList([])
        for _layer_idx in range(n_layers):
            self.layers.append(
                Layer(
                    d_model=d_model,
                    d_head=d_head,
                    d_hidden=d_hidden,
                    max_tokens=max_tokens,
                )
            )

        # Expose config as public attributes
        self.d_head = d_head
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.max_tokens = max_tokens
        self.n_layers = n_layers

    def forward(self, tokens: BatchTokenIndicesTT) -> BatchLogitsTT:
        """Forward pass.

        Args:
            tokens (BatchTokenIndicesTT): Input tokens (indices rather than one-hot)

        Returns:
            BatchLogitsTT: Logits representing log probabilities for the tokens
        """
        # Embed + positional encoding
        residual_stream: BatchResidualStreamTT = self.embed(tokens)
        residual_stream = self.positional_encoding(residual_stream)

        # Loop through layers
        for layer in self.layers:
            residual_stream = layer(residual_stream)

        # Unembed and return
        return self.unembed(residual_stream)
