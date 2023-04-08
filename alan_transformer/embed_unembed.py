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
            torch.rand(d_vocab, d_model),
        )

        # Initialise the weights
        initrange = 1.0 / math.sqrt(d_vocab)
        self.embed_weights.data.uniform_(-initrange, initrange)

    def forward(self, tokens: TokensTT) -> ResidualStreamTT:
        """Forward Pass through the Embedding Layer.

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

        nn.init.kaiming_uniform_(self.unembed_weights, a=math.sqrt(5))

    def forward(self, residual_stream: ResidualStreamTT) -> LogitsTT:
        """Forward Pass through the Unembedding Layer.

        Args:
            residual_stream (ResidualStreamTT): Residual stream

        Returns:
            LogitsTT: Logits representing probabilities for the tokens
        """
        return einsum(
            "batch pos model, model vocab -> batch pos vocab",
            residual_stream,
            self.unembed_weights,
        )
