import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from alan_transformer.types import (
    D,
    BatchLogitsTT,
    BatchResidualStreamTT,
    BatchTokenIndicesTT,
)


EmbedUnembedWeightsTT = Float[Tensor, f"{D.VOCAB} {D.RESIDUAL_FEATURE}"]


class Embed(nn.Module):
    """Embedding Layer.

    Note that in the original paper the weights for the unembed are shared with the weights for the
    embed. However, it turns out that the embed and unembed weights are an efficient place for
    the model to do bigram statistics (probability of a next token given just the current token,
    e.g. to learn that "Obama" commonly follows "Barack"). This was found in A Mathematical
    Framework for Transformer Circuits
    (https://transformer-circuits.pub/2021/framework/index.html). We therefore use separate
    embed and unembed weights.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_vocab: int, d_model: int) -> None:
        """Initialize the Embed layer.

        Args:
            d_vocab (int): Number of tokens in the vocabulary
            d_model (int): Dimensionality of the residual stream
        """
        super().__init__()

        self.d_model: int = d_model

        self.embed_weights: EmbedUnembedWeightsTT = nn.Parameter(
            torch.empty(d_vocab, d_model),
        )

        # Initialise the weights
        # We use Xavier initialisation here as an appropriate choice where there is either a
        # symmetrical activation function (e.g. tanh) or no activation function (as here).
        nn.init.xavier_uniform_(self.embed_weights)

    def forward(self, tokens: BatchTokenIndicesTT) -> BatchResidualStreamTT:
        """Forward Pass through the Embedding Layer.

        The original paper multiples the embedding by sqrt(d_model) during the forward pass,
        presumably as a fix for having the same weights in the embedding and unembedding layers.
        However, as we're not following that approach we have omitted this implementation detail.

        Args:
            tokens (TokensTT): Input tokens (indices rather than one-hot)

        Returns:
            ResidualStreamTT: Continuous representations of the tokens
        """
        # Index into weights (with the tokens)
        return self.embed_weights[tokens, :]


class Unembed(nn.Module):
    """Unembedding layer.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_vocab: int, d_model: int) -> None:
        """Initialize the Unembed Layer.

        Args:
            d_vocab (int): Number of tokens in the vocabulary
            d_model (int): Dimensionality of the residual stream
        """
        super().__init__()

        self.unembed_weights: EmbedUnembedWeightsTT = nn.Parameter(
            torch.empty(d_model, d_vocab),
        )

        nn.init.xavier_uniform_(self.unembed_weights)

    def forward(self, residual_stream: BatchResidualStreamTT) -> BatchLogitsTT:
        """Forward Pass through the Unembedding Layer.

        Args:
            residual_stream (ResidualStreamTT): Residual stream

        Returns:
            LogitsTT: Logits representing log probabilities for the tokens
        """
        return einsum(
            "batch pos model, model vocab -> batch pos vocab",
            residual_stream,
            self.unembed_weights,
        )
