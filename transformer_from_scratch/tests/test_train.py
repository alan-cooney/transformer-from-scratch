from typing import Dict
import torch
from torch.utils.data import Dataset

from transformer_from_scratch.train import evaluate, get_default_device
from transformer_from_scratch.types import (
    BatchLogitsTT,
    BatchTokenIndicesTT,
    TokenIndicesTT,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, output_indices: int, vocab_size: int) -> None:
        super().__init__()
        self.output_indices = output_indices
        self.vocab_size = vocab_size

    def forward(self, inputs: BatchTokenIndicesTT) -> BatchLogitsTT:
        """Forward pass

        Args:
            x (BatchTokenIndicesTT): Inputs

        Returns:
            BatchLogitsTT: 100% log probabilities of the specified output token index, for all positions in all batches.
        """
        output_indices = torch.fill(inputs, self.output_indices)
        return torch.nn.functional.one_hot(
            output_indices, num_classes=self.vocab_size
        ).float()


class MyDataset(Dataset):
    """Test Dataset."""

    def __init__(self, inputs: BatchTokenIndicesTT):
        super().__init__()
        self.inputs: BatchTokenIndicesTT = inputs

    def __getitem__(self, index) -> Dict[str, TokenIndicesTT]:
        return {"input_ids": self.inputs[index]}

    def __len__(self):
        return len(self.inputs)


class TestDevice:
    """Default device tests."""

    def test_get_default_device(self):
        """Test that a valid PyTorch device is returned."""
        device = get_default_device()
        assert isinstance(
            device, torch.device
        ), "Returned device is not a torch.device instance"


class TestEvaluate:
    """Evaluate tests."""

    def test_fully_accurate_model(self):
        """Test that a model that always outputs the correct token index, gets 100% accuracy."""
        model = SimpleModel(output_indices=3, vocab_size=10)
        inputs: BatchTokenIndicesTT = torch.tensor(
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        )
        test_dataset = MyDataset(inputs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        accuracy = evaluate(model, test_dataloader)
        assert accuracy == 1.0

    def test_fully_inaccurate_model(self):
        """Test that a model that always outputs the incorrect token index, gets 100% accuracy."""
        model = SimpleModel(output_indices=2, vocab_size=10)
        inputs: BatchTokenIndicesTT = torch.tensor(
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        )
        test_dataset = MyDataset(inputs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        accuracy = evaluate(model, test_dataloader)
        assert accuracy == 0.0
