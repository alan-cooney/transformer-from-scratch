"""PyTest global fixtures."""
import pytest
import torch

torch.set_default_device('cpu')

@pytest.fixture(autouse=True)
def set_default_device():
    """Default all tests to run on CPU."""
    torch.set_default_device('cpu')
