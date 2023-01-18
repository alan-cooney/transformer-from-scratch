"""Shared types"""
from enum import Enum
from torchtyping import TensorType as TT


TokensTT = TT["batch", "pos"]

ResidualStreamTT = TT["batch", "pos", "d_model"]

LogitsTT = TT["batch", "pos", "d_vocab"]
