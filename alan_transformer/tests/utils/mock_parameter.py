import torch
from torch import nn


class MockParameter(nn.Parameter):
    """Helper to set mocked parameters to ones"""

    def __init__(self, data: torch.Tensor):
        self.data = torch.ones_like(self.data)
