from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from transformer_from_scratch.train import evaluate, get_default_device, train_loop
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.types import (
    BatchLogits,
    BatchTokenIndices,
    TokenIndices,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self, output_indices: int, vocab_size: int) -> None:
        super().__init__()
        self.output_indices = output_indices
        self.vocab_size = vocab_size

    def forward(self, inputs: BatchTokenIndices) -> BatchLogits:
        """Forward pass

        Args:
            x (BatchTokenIndices): Inputs

        Returns:
            BatchLogits: 100% log probabilities of the specified output token index, for all positions in all batches.
        """
        output_indices = torch.fill(inputs, self.output_indices)
        return torch.nn.functional.one_hot(
            output_indices, num_classes=self.vocab_size
        ).float()


class MyDataset(Dataset):
    """Test Dataset."""

    def __init__(self, inputs: BatchTokenIndices):
        super().__init__()
        self.inputs: BatchTokenIndices = inputs

    def __getitem__(self, index) -> Dict[str, TokenIndices]:
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
        inputs: BatchTokenIndices = torch.tensor(
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        )
        test_dataset = MyDataset(inputs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        accuracy = evaluate(model, test_dataloader)
        assert accuracy == 1.0

    def test_fully_inaccurate_model(self):
        """Test that a model that always outputs the incorrect token index, gets 100% accuracy."""
        model = SimpleModel(output_indices=2, vocab_size=10)
        inputs: BatchTokenIndices = torch.tensor(
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        )
        test_dataset = MyDataset(inputs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        accuracy = evaluate(model, test_dataloader)
        assert accuracy == 0.0


class TestTrainLoop:
    """Train loop tests."""

    def test_train_loop_executes(self, tmpdir):
        """Test that the train loop runs without error."""
        model = Transformer()
        inputs: BatchTokenIndices = torch.tensor(
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        )
        dataset = MyDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        checkpoint_dir = Path(tmpdir)
        train_loop(
            model,
            dataloader,
            dataloader,
            epochs=1,
            checkpoint_dir=checkpoint_dir,
            device=torch.device("cpu"),
        )

    def test_model_parameters_change(self, tmpdir):
        """Test that model parameters change after training."""
        model = Transformer()
        inputs: BatchTokenIndices = torch.randint(low=0, high=10, size=(4, 3))
        dataset = MyDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        checkpoint_dir = Path(tmpdir)

        # Get initial model parameters
        initial_params = [param.clone() for param in model.parameters()]

        # Train
        train_loop(
            model,
            dataloader,
            dataloader,
            epochs=1,
            checkpoint_dir=checkpoint_dir,
            device=torch.device("cpu"),
        )

        # Check if parameters have changed
        for param, initial in zip(model.parameters(), initial_params):
            assert not torch.equal(
                param, initial
            ), "Model parameter did not change during training"
            assert torch.allclose(
                param, initial, atol=0.1
            ), "Model parameter changed too much during training"
