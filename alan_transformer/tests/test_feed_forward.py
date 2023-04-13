from typing import Tuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
import pytest

from alan_transformer.feed_forward import FeedForward
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes

from ..types import ResidualStreamTT
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, num_samples: int, sequence_length: int, d_model: int):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.d_model = d_model

    def __len__(self):
        return self.num_samples


class IdentityDataset(CustomDataset):
    def __getitem__(self, index):
        x = torch.rand(self.sequence_length, self.d_model)
        return x, x


class BitReversalDataset(CustomDataset):
    def __getitem__(self, index):
        x = torch.randint(0, 2, (self.sequence_length, self.d_model))
        y = x.flip(dims=[1])
        return x, y


class BinaryAdditionDataset(CustomDataset):
    def __getitem__(self, index):
        x = torch.randint(0, 2, (self.sequence_length, self.d_model // 2))
        y = torch.randint(0, 2, (self.sequence_length, self.d_model // 2))
        z = (x + y).fmod(2)
        return torch.cat([x, y], dim=1), z


class OddEvenDataset(CustomDataset):
    def __getitem__(self, index):
        x = torch.randint(0, 2, (self.sequence_length, self.d_model))
        y = torch.stack([x[:, 1::2], x[:, ::2]], dim=1).view(
            self.sequence_length, self.d_model
        )
        return x, y


@pytest.mark.parametrize(
    "dataset_class",
    [IdentityDataset, BitReversalDataset, BinaryAdditionDataset, OddEvenDataset],
)
def test_feed_forward_learns_on_dataset(dataset_class: CustomDataset):
    # Set random seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Parameters
    samples = 10000
    epochs = 10
    d_vocab = 10
    d_model = 10
    d_hidden = 10

    # Model
    model = FeedForward(d_model, d_hidden)

    # Dataset
    dataset = IdentityDataset(samples, d_vocab, d_model)
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
            outputs = model(inputs)
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
            for _batch in outputs:
                # We define correct as the model output being within 1e-3 (relative distance) from
                # the target
                if torch.allclose(inputs, targets, rtol=1e-3, atol=0):
                    correct += 1
                total += 1

    accuracy = correct / total
    assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
