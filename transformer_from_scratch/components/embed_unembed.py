"""Embedding and Unembedding Layers."""
import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.types import (
    BatchLogits,
    BatchResidualStream,
    BatchTokenIndices,
    D,
)

EmbedUnembedWeights = Float[Tensor, f"{D.VOCAB} {D.RESIDUAL_FEATURE}"]


class Embed(nn.Module):
    """Embedding Module.

    Embedding involves taking each token, and embedding it from the high-dimensional vocabulary
    space (which can be of size e.g. 50,000) into the lower-dimensional residual stream space (which
    is of size `d_model`). This module embeds a batch of prompts (each containing multiple tokens).

    As an implementation detail, note that the tokens are provided as indices within the vocabulary
    (i.e. they are of shape BATCH x TOKEN where each value is an integer from 0 - vocabulary size),
    rather than as one-hot-encoded tensors (of shape BATCH x TOKEN x VOCABULARY). This means that
    whilst mathematically the embedding can be thought of as  matrix multiplication, in practice it
    we index into the weights matrix using the token indices.

    Note that in the original paper the weights for the unembed are shared with the weights for the
    embed. However, it turns out that the embed and unembed weights are an efficient place for
    the model to do bigram statistics (probability of a next token given just the current token,
    e.g. to learn that "Obama" commonly follows "Barack"). This was found in A Mathematical
    Framework for Transformer Circuits
    (https://transformer-circuits.pub/2021/framework/index.html). We therefore use separate
    embed and unembed weights.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the Embed layer."""
        super().__init__()

        self.d_model: int = config.d_model

        self.embed_weights: EmbedUnembedWeights = nn.Parameter(
            torch.empty(config.d_vocab, config.d_model),
        )

        # Initialise the weights
        # We use Xavier initialisation here as an appropriate choice where there is either a
        # symmetrical activation function (e.g. tanh) or no activation function (as here).
        nn.init.xavier_uniform_(self.embed_weights)

    def forward(self, tokens: BatchTokenIndices) -> BatchResidualStream:
        """Forward Pass through the Embedding Layer.

        The original paper multiples the embedding by sqrt(d_model) during the forward pass,
        presumably as a fix for having the same weights in the embedding and unembedding layers.
        However, as we're not following that approach we have omitted this implementation detail.

        Args:
            tokens (Tokens): Input tokens (indices rather than one-hot)

        Returns:
            ResidualStream: Continuous representations of the tokens
        """
        # Index into weights (with the tokens)
        return self.embed_weights[tokens, :]


class Unembed(nn.Module):
    """Unembedding Module.

    Unembedding involves taking each token at the end of the residual stream, and "unembedding" it
    from the low-dimensional residual stream space (which is of size `d_model`) into the higher
    dimensional vocabulary space (which can be of size e.g. 50,000). The resulting logits represent
    the log probability of each token in the vocabulary (for each position in each prompt in the
    batch).

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the Unembed Layer."""
        super().__init__()

        self.unembed_weights: EmbedUnembedWeights = nn.Parameter(
            torch.empty(config.d_model, config.d_vocab),
        )

        nn.init.xavier_uniform_(self.unembed_weights)

    def forward(self, residual_stream: BatchResidualStream) -> BatchLogits:
        """Forward Pass through the Unembedding Layer.

        Args:
            residual_stream (ResidualStream): Residual stream

        Returns:
            Logits: Logits representing log probabilities for the tokens
        """
        return einsum(
            "batch pos model, model vocab -> batch pos vocab",
            residual_stream,
            self.unembed_weights,
        )
