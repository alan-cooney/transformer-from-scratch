import torch
from transformer_from_scratch.train import get_default_device
from transformer_from_scratch.types import BatchLogitsTT, BatchTokenIndicesTT


class SimpleModel(torch.nn.Module):
    def __init__(self, output_indices: int, vocab_size: int) -> None:
        super().__init__()
        self.output_indices = output_indices
        self.vocab_size = vocab_size
    
    def forward(self, x: BatchTokenIndicesTT) -> BatchLogitsTT:
        output_indices = torch.fill(x, self.output_indices)
        return torch.nn.functional.one_hot(output_indices, num_classes=self.vocab_size).float()


class TestDevice:
    """Default device tests."""
    
    def test_get_default_device(self):
        """Test that a valid PyTorch device is returned."""
        device = get_default_device()
        assert isinstance(device, torch.device), "Returned device is not a torch.device instance"

# class TestEvaluate:
#     def test_fully_accurate_model(self):
        