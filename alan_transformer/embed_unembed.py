from torch import nn
from torchtyping import TensorType as TT
import torch
from fancy_einsum import einsum

TokenizedType = TT["batch", "pos", "d_vocab"]
ResidualStreamType = TT["batch", "pos", "d_model"]


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()

        self.embed_weights: TT["d_vocab", "d_model"] = nn.Parameter(
            torch.rand(d_vocab, d_model))

        self.embed_bias: TT["d_model"] = nn.Parameter(torch.rand(d_model))

    def forward(self, tokens: TokenizedType) -> ResidualStreamType:
        """Forward pass"""
        embed_pre_bias = einsum(
            "batch pos d_vocab, d_vocab d_model -> batch pos d_model", tokens, self.embed_weights)

        return embed_pre_bias + self.embed_bias


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()

        self.unembed_weights: TT["d_model", "d_vocab"] = nn.Parameter(
            torch.rand(d_model, d_vocab))

        self.embed_bias: TT["d_vocab"] = nn.Parameter(torch.rand(d_vocab))

    def forward(self, residual_stream: ResidualStreamType) -> TokenizedType:
        """Forward pass"""
        unembed_pre_bias = einsum(
            "batch pos d_model, d_model d_vocab -> batch pos d_vocab", residual_stream, self.unembed_weights)

        return unembed_pre_bias + self.embed_bias
