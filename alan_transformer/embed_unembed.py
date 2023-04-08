import math

import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from alan_transformer.types import LogitsTT, ResidualStreamTT, TokensTT


class Embed(nn.Module):
    """Embedding Layer.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_vocab: int, d_model: int) -> None:
        """Initialize the Embed layer.

        Args:
            d_vocab: The size of the input vocabulary
            d_model: The model/residual stream dimension size
        """
        super().__init__()

        self.d_model: int = d_model

        self.embed_weights: Float[Tensor, "vocab d_model"] = nn.Parameter(
            torch.empty(d_vocab, d_model),
        )

        self.embed_bias: Float[Tensor, "d_model"] = nn.Parameter(torch.empty(d_model))

    def forward(self, tokens: TokensTT) -> ResidualStreamTT:
        """Forward Pass through the Embedding Layer.

        Args:
            tokens (TokensTT): Input tokens (indices rather than one-hot)

        Returns:
            ResidualStreamTT: Continuous representations of the tokens
        """
        # Index into weights (with the tokens)
        embed_pre_bias: ResidualStreamTT = self.embed_weights[tokens, :]

        # Note the paper multiplies the embedding weights by sqrt(d_model),
        # which is equivalent to doing this here
        return embed_pre_bias * math.sqrt(self.d_model) + self.embed_bias


class Unembed(nn.Module):
    """Unembedding layer.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p5)

    Note that in the paper the weights for the unembed are shared with the
    weights for the embed. However, this is not very principled as these learned
    weights can do bigrams if they are different, so instead the unembed layer
    uses separate weights here.
    """

    def __init__(self, d_vocab: int, d_model: int) -> None:
        """Initialize the Unembed Layer.

        Args:
            d_vocab: The size of the input vocabulary
            d_model: The model/residual stream dimension size
        """
        super().__init__()

        self.unembed_weights: Float[Tensor, "d_model vocab"] = nn.Parameter(
            torch.empty(d_model, d_vocab),
        )

        self.unembed_bias: Float[Tensor, " vocab"] = nn.Parameter(torch.empty(d_vocab))

    def forward(self, residual_stream: ResidualStreamTT) -> LogitsTT:
        """Forward Pass through the Unembedding Layer.

        Args:
            residual_stream (ResidualStreamTT): Residual stream

        Returns:
            LogitsTT: Logits representing probabilities for the tokens
        """
        unembed_pre_bias = einsum(
            "batch pos model, model vocab -> batch pos vocab",
            residual_stream,
            self.unembed_weights,
        )

        return unembed_pre_bias + self.unembed_bias
