"""Feed Forward Module."""
import torch
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from transformer_from_scratch.components.config import TransformerConfig
from transformer_from_scratch.types import BatchResidualStream
from transformer_from_scratch.types import TensorShapeLabels as D

BatchHidden = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION} {D.MLP_FEATURE}",
]
InnerWeights = Float[Tensor, f"{D.RESIDUAL_FEATURE} {D.MLP_FEATURE}"]
InnerBias = Float[Tensor, f"{D.MLP_FEATURE}"]
OuterWeights = Float[Tensor, f"{D.MLP_FEATURE} {D.RESIDUAL_FEATURE}"]
OuterBias = Float[Tensor, f"{D.RESIDUAL_FEATURE}"]


class MLP(nn.Module):
    """MLP

    The MLP module takes an input of the residual stream and applies a standard two-layer
    feed forward network. The resulting output will then be added back onto the residual stream by
    the transformer.

    MLP(x) = max(0, xW1 + b1)W2 + b2

    https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, config: TransformerConfig) -> None:
        """MLP Sub-Layer Initialisation."""
        super().__init__()

        self.weight_inner: InnerWeights = nn.Parameter(
            torch.empty(config.d_model, config.d_mlp),
        )

        self.bias_inner: InnerBias = nn.Parameter(torch.zeros(config.d_mlp))

        self.weight_outer: OuterWeights = nn.Parameter(
            torch.empty(config.d_mlp, config.d_model),
        )

        self.bias_outer: OuterBias = nn.Parameter(torch.zeros(config.d_model))

        # Initialise the weights
        # We use Kaiming Initialization for the inner weights, as we have a non-symmetric activation
        # function (ReLU)
        nn.init.kaiming_normal_(self.weight_inner)

        # We use Xavier Initialization for the outer weights, as we have no activation function
        nn.init.xavier_normal_(self.weight_outer)

    def forward(self, residual_stream: BatchResidualStream) -> BatchResidualStream:
        """Forward Pass through the MLP Sub-Layer.

        Args:
            residual_stream (ResidualStream): MLP input

        Returns:
            ResidualStream: MLP output
        """
        # Inner = relu(x W1 + b1)
        inner_pre_bias: BatchHidden = einsum(
            "batch pos d_model, d_model d_hidden -> batch pos d_hidden",
            residual_stream,
            self.weight_inner,
        )
        inner = inner_pre_bias + self.bias_inner
        inner_relu: BatchHidden = torch.relu(inner)

        # Outer = inner @ W2 + b2
        outer_pre_bias: BatchResidualStream = einsum(
            "batch pos d_hidden, d_hidden d_model -> batch pos d_model",
            inner_relu,
            self.weight_outer,
        )
        return outer_pre_bias + self.bias_outer
