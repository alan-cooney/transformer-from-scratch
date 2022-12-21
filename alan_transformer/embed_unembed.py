import torch
from fancy_einsum import einsum
from torch import nn
from torchtyping import TensorType as TT

TokenizedType = TT["batch", "pos", "d_vocab"]
ResidualStreamType = TT["batch", "pos", "d_model"]


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()

        self.embed_weights: TT["d_vocab", "d_model"] = nn.Parameter(
            torch.empty(d_vocab, d_model))

        self.embed_bias: TT["d_model"] = nn.Parameter(torch.empty(d_model))

    def forward(self, tokens: TokenizedType) -> ResidualStreamType:
        """Forward pass"""
        embed_pre_bias = einsum(
            "batch pos d_vocab, d_vocab d_model -> batch pos d_model", tokens, self.embed_weights)

        return embed_pre_bias + self.embed_bias


class Unembed(nn.Module):
    """Unembed layer

    Note that in the paper the weights for the unembed are shared with the
    weights for the embed (except on embedding the are divided by
    sqrt(d_model)). However, this is not very principled as these learned
    weights can do bigrams if they are different, so instead the unembed layer
    uses separate weights here.
    """

    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()

        self.unembed_weights: TT["d_model", "d_vocab"] = nn.Parameter(
            torch.empty(d_model, d_vocab))

        self.embed_bias: TT["d_vocab"] = nn.Parameter(torch.empty(d_vocab))

    def forward(self, residual_stream: ResidualStreamType) -> TokenizedType:
        """Forward pass"""
        unembed_pre_bias = einsum(
            "batch pos d_model, d_model d_vocab -> batch pos d_vocab", residual_stream, self.unembed_weights)

        return unembed_pre_bias + self.embed_bias
