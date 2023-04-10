from typing import Tuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from alan_transformer.feed_forward import FeedForward
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes

from ..types import ResidualStreamTT


class XorDataset(Dataset):
    """XOR Dataset.

    Creates a simple dataset that can be solved with a 2-layer feed forward network, but not a
    1-layer one.

    For each token vector (e.g. [0, 0, 1, 1, 0]), the target is [1, 0 ... 0] if XOR is true and [0, 1, 0,
    ... 0] otherwise.
    """

    samples: list[Float[Tensor, "pos d_model"]] = []
    targets: list[Float[Tensor, "pos d_model"]] = []

    def __init__(self, num_samples: int, d_model: int, num_tokens: int):
        for _ in range(num_samples):
            # Generate sample as a (num_tokens, d_model) tensor of 0s and 1s
            probability_one = 1 / d_model  # Prob of each element being 1
            sample_probabilities = torch.full((num_tokens, d_model), probability_one)
            sample = torch.bernoulli(sample_probabilities)
            self.samples.append(sample.float())

            # Generate target as a (num_tokens, d_model) tensor where the first element is 1 if the
            # XOR is true and 0 otherwise. The next element is 1 if XOR is false and zero otherwise.
            # All other elements are zero.
            target = torch.zeros((num_tokens, d_model))
            for token_idx, token in enumerate(sample):
                xor_true: bool = int(token.sum().item()) == 1
                target[token_idx][0] = 1.0 if xor_true else 0.0
                # target[token_idx][1] = 1.0 if not xor_true else 0.0
            self.targets.append(target.float())

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self, idx
    ) -> Tuple[Float[Tensor, "pos d_model"], Float[Tensor, "pos d_model"]]:
        return self.samples[idx], self.targets[idx]


class TestFeedForward:
    def test_learn_solve_non_linear_function(self):
        """Test that the feed forward module can learn to solve XOR."""
        # Set random seeds
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        samples = 1000
        epochs = 100
        num_tokens = 1
        d_model = 3
        d_hidden = d_model * 10

        model = FeedForward(d_model, d_hidden)

        dataset = XorDataset(samples, d_model, num_tokens)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters())

        # Train the model
        for _epoch in range(epochs):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(loss)

        # Test the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs: ResidualStreamTT = model(inputs)
                _, predicted = torch.max(outputs, 2)
                _, target_indices = torch.max(targets, 2)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == target_indices).sum().item()

        accuracy = correct / total
        assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
