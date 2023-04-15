"""Feed Forward Tests."""
import random
from typing import Tuple, Type

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from transformer_from_scratch.feed_forward import FeedForward

from ..types import BatchLogitsTT, ResidualStreamTT


class RegressionTaskDataset(Dataset):
    """Regression task dataset.

    Extend this to add datasets that the model should be able to learn.
    """

    def __init__(self, num_samples: int, sequence_length: int, d_model: int):
        """Initialize the dataset.

        Args:
            num_samples (int): Number of samples
            sequence_length (int): Sequence length
            d_model (int): Dimensionality of the residual stream
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.d_model = d_model

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[ResidualStreamTT, ResidualStreamTT]:
        raise NotImplementedError


class IdentityDataset(RegressionTaskDataset):
    """Identity Dataset.

    The model should learn to output the same input values. Each input neuron is connected to the
    corresponding output neuron.
    """

    def __getitem__(self, index) -> Tuple[ResidualStreamTT, ResidualStreamTT]:
        sample: ResidualStreamTT = torch.rand(self.sequence_length, self.d_model)
        return sample, sample


class BitReversalDataset(RegressionTaskDataset):
    """Bit Reversal Dataset.

    The model should learn to reverse the order of the input bits. For example, if the input is
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1], the output should be [1, 0, 1, 0, 0, 1, 0, 1, 1, 0].
    """

    def __getitem__(self, index) -> Tuple[ResidualStreamTT, ResidualStreamTT]:
        sample: ResidualStreamTT = torch.randint(
            0, 2, (self.sequence_length, self.d_model)
        ).float()
        target: ResidualStreamTT = sample.flip(dims=[1])
        return sample, target


class BinaryAdditionDataset(RegressionTaskDataset):
    """Binary Addition Dataset.

    The model should learn to add two 5-bit numbers. The first five input neurons represent the
    first number, and the next five input neurons represent the second number. The output is a
    10-bit sequence that represents the binary sum of the two input numbers.
    """

    def binary_addition(self, x: torch.Tensor, y: torch.Tensor) -> ResidualStreamTT:
        """Binary Addition

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_

        Returns:
            ResidualStreamTT: _description_
        """
        carry = 0
        result = torch.zeros_like(x)

        for i in range(x.size(1) - 1, -1, -1):
            result[:, i] = (x[:, i] + y[:, i] + carry) % 2
            carry = (x[:, i] + y[:, i] + carry) // 2

        return result

    def __getitem__(self, index) -> Tuple[ResidualStreamTT, ResidualStreamTT]:
        first_number = torch.randint(0, 2, (self.sequence_length, self.d_model // 2))
        second_number = torch.randint(0, 2, (self.sequence_length, self.d_model // 2))
        concatenated_numbers: ResidualStreamTT = torch.cat(
            [first_number, second_number], dim=1
        ).float()
        binary_sum: ResidualStreamTT = self.binary_addition(first_number, second_number)
        return concatenated_numbers, binary_sum


class OddEvenDataset(RegressionTaskDataset):
    """Odd Even Dataset.

    The model should learn to separate the input bits based on their index (odd or even). The output
    should have the odd-indexed input bits in the first five positions and the even-indexed input
    bits in the last five positions.
    """

    def __getitem__(self, index) -> Tuple[ResidualStreamTT, ResidualStreamTT]:
        x: ResidualStreamTT = torch.randint(
            0, 2, (self.sequence_length, self.d_model)
        ).float()
        y: ResidualStreamTT = torch.stack([x[:, 1::2], x[:, ::2]], dim=1).view(
            self.sequence_length, self.d_model
        )
        return x, y


@pytest.mark.parametrize(
    "dataset_class",
    [IdentityDataset, BitReversalDataset, BinaryAdditionDataset, OddEvenDataset],
)
def test_feed_forward_learns_on_dataset(dataset_class: Type[RegressionTaskDataset]):
    """Test the feed forward network learns to solve the regression tasks.

    Args:
        dataset_class (RegressionTaskDataset): Dataset to test.
    """

    # Set random seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Parameters
    samples = 10000
    epochs = 15
    d_vocab = 10
    d_model = 10
    d_hidden = 10

    # Model
    model = FeedForward(d_model, d_hidden)

    # Dataset
    dataset = dataset_class(samples, d_vocab, d_model)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train
    for _epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs: BatchLogitsTT = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            for batch_idx, output in enumerate(outputs):
                for token_idx, token in enumerate(output):
                    target_token = targets[batch_idx][token_idx]
                    # We use low absolute difference as the numbers should be either 0 or 1.
                    if torch.allclose(token, target_token, rtol=0, atol=0.3):
                        correct += 1
                    total += 1

    accuracy = correct / total
    assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
