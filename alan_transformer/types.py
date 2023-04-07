"""Shared types"""
from jaxtyping import Float
from torch import Tensor

TokensTT = Float[Tensor, "batch pos"]

ResidualStreamTT = Float[Tensor, "batch pos d_model"]

LogitsTT = Float[Tensor, "batch pos d_vocab"]
