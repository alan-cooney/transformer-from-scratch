"""Shared types"""
from jaxtyping import Float, Int
from torch import Tensor

TokensTT = Int[Tensor, "batch pos"]

ResidualStreamTT = Float[Tensor, "batch pos d_model"]

LogitsTT = Float[Tensor, "batch pos d_vocab"]
