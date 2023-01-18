import torch
from fancy_einsum import einsum
from torch import nn
from torchtyping import TensorType as TT

from alan_transformer.types import ResidualStreamTT

HiddenTT = TT["batch", "pos", "d_hidden"]


class FeedForward(nn.Module):
    """Feed Forward Sub-Layer

    FFN(x) = max(0, xW1 + b1)W2 + b2

    https://arxiv.org/pdf/1706.03762.pdf (p5)
    """

    def __init__(self, d_model: int = 768, d_hidden: int = 2048) -> None:
        super().__init__()

        self.weight_inner: TT["d_model", "d_hidden"] = nn.Parameter(
            torch.empty(d_model, d_hidden))

        self.bias_inner: TT["d_hidden"] = nn.Parameter(
            torch.empty(d_hidden))

        self.weight_outer: TT["d_hidden", "d_model"] = nn.Parameter(
            torch.empty(d_hidden, d_model))

        self.bias_outer: TT["d_model"] = nn.Parameter(
            torch.empty(d_model))

    def forward(self, residual_stream: ResidualStreamTT) -> ResidualStreamTT:
        """Forward pass"""
        # Inner = relu(x W1 + b1)
        inner_pre_bias: HiddenTT = einsum(
            "batch pos d_model, d_model d_hidden -> batch pos d_hidden", residual_stream, self.weight_inner)
        inner = inner_pre_bias + self.bias_inner
        inner_relu: HiddenTT = torch.relu(inner)

        # Outer = inner @ W2 + b2
        outer_pre_bias: ResidualStreamTT = einsum(
            "batch pos d_hidden, d_hidden d_model -> batch pos d_model", inner_relu, self.weight_outer)
        return outer_pre_bias + self.bias_outer
