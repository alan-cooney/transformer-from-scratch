"""Feed Forward Module."""
import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from transformer_from_scratch.types import BatchResidualStreamTT
from transformer_from_scratch.types import TensorShapeLabels as D

BatchHiddenTT = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION} {D.HIDDEN_FEATURE}",
]
InnerWeightsTT = Float[Tensor, f"{D.RESIDUAL_FEATURE} {D.HIDDEN_FEATURE}"]
InnerBiasTT = Float[Tensor, f"{D.HIDDEN_FEATURE}"]
OuterWeightsTT = Float[Tensor, f"{D.HIDDEN_FEATURE} {D.RESIDUAL_FEATURE}"]
OuterBiasTT = Float[Tensor, f"{D.RESIDUAL_FEATURE}"]


class FeedForward(nn.Module):
    """Feed Forward Module

    The feed forward module takes an input of the residual stream and applies a standard two-layer
    feed forward network. The resulting output will then be added back onto the residual stream by
    the transformer.

    FFN(x) = max(0, xW1 + b1)W2 + b2

    https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """Feed Forward Sub-Layer Initialisation

        Args:
            d_model (int): Number of residual stream features per token.
            d_hidden (int): Number of hidden layer features per token.
        """
        super().__init__()

        self.weight_inner: InnerWeightsTT = nn.Parameter(
            torch.empty(d_model, d_hidden),
        )

        self.bias_inner: InnerBiasTT = nn.Parameter(torch.zeros(d_hidden))

        self.weight_outer: OuterWeightsTT = nn.Parameter(
            torch.empty(d_hidden, d_model),
        )

        self.bias_outer: OuterBiasTT = nn.Parameter(torch.zeros(d_model))

        # Initialise the weights
        # We use Kaiming Initialization for the inner weights, as we have a non-symmetric activation
        # function (ReLU)
        nn.init.kaiming_normal_(self.weight_inner)

        # We use Xavier Initialization for the outer weights, as we have no activation function
        nn.init.xavier_normal_(self.weight_outer)

    def forward(self, residual_stream: BatchResidualStreamTT) -> BatchResidualStreamTT:
        """Forward Pass through the Feed Forward Sub-Layer.

        Args:
            residual_stream (ResidualStreamTT): Feed Forward input

        Returns:
            ResidualStreamTT: Feed Forward output
        """
        # Inner = relu(x W1 + b1)
        inner_pre_bias: BatchHiddenTT = einsum(
            "batch pos d_model, d_model d_hidden -> batch pos d_hidden",
            residual_stream,
            self.weight_inner,
        )
        inner = inner_pre_bias + self.bias_inner
        inner_relu: BatchHiddenTT = torch.relu(inner)

        # Outer = inner @ W2 + b2
        outer_pre_bias: BatchResidualStreamTT = einsum(
            "batch pos d_hidden, d_hidden d_model -> batch pos d_model",
            inner_relu,
            self.weight_outer,
        )
        return outer_pre_bias + self.bias_outer
