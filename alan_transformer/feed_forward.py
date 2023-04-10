import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from alan_transformer.types import ResidualStreamTT

HiddenTT = Float[Tensor, "batch pos d_hidden"]


class FeedForward(nn.Module):
    """Feed Forward Sub-Layer

    FFN(x) = max(0, xW1 + b1)W2 + b2

    https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """Feed Forward Sub-Layer Initialisation

        Args:
            d_model (int): Dimensionality of the residual stream
            d_hidden (int): Dimensionality of the hidden layer
        """
        super().__init__()

        self.weight_inner: Float[Tensor, "d_model d_hidden"] = nn.Parameter(
            torch.empty(d_model, d_hidden),
        )

        self.bias_inner: Float[Tensor, "d_hidden"] = nn.Parameter(torch.zeros(d_hidden))

        self.weight_outer: Float[Tensor, "d_hidden d_model"] = nn.Parameter(
            torch.empty(d_hidden, d_model),
        )

        self.bias_outer: Float[Tensor, "d_model"] = nn.Parameter(torch.zeros(d_model))

        # Initialise the weights
        # We use Kaiming Initialization for the inner weights, as we have a non-symmetric activation
        # function (ReLU)
        nn.init.kaiming_normal_(self.weight_inner)

        # We use Xavier Initialization for the outer weights, as we have no activation function
        nn.init.xavier_normal_(self.weight_outer)

    def forward(self, residual_stream: ResidualStreamTT) -> ResidualStreamTT:
        """Forward Pass through the Feed Forward Sub-Layer.

        Args:
            residual_stream (ResidualStreamTT): Feed Forward input

        Returns:
            ResidualStreamTT: Feed Forward output
        """
        # Inner = relu(x W1 + b1)
        inner_pre_bias: HiddenTT = einsum(
            "batch pos d_model, d_model d_hidden -> batch pos d_hidden",
            residual_stream,
            self.weight_inner,
        )
        inner = inner_pre_bias + self.bias_inner
        inner_relu: HiddenTT = torch.relu(inner)

        # Outer = inner @ W2 + b2
        outer_pre_bias: ResidualStreamTT = einsum(
            "batch pos d_hidden, d_hidden d_model -> batch pos d_model",
            inner_relu,
            self.weight_outer,
        )
        return outer_pre_bias + self.bias_outer
