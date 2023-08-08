"""Embedding and Unembedding Module Tests.

Embedding and Unembedding modules, when combined into a zero-layer transformer (i.e. with no
attention or MLP layers between them), theoretically act as a single linear transformation between
the input tokens and the output logits (token log probabilities). This is because there are no
non-linearities (e.g. activation functions) between the embedding and unembedding layers, so they
are mathematically equivalent to a single linear transformation. This is shown in A Mathematical
Framework for Transformer Circuits (https://transformer-circuits.pub/2021/framework/index.html).

The tests here are designed to verify that, when combined, the embedding and unembedding layers can
indeed learn some basic things that we would expect them to be able to learn given that they are
just doing this simple linear transformation. Specifically, we want to verify that they can learn
bigram statistics (i.e. the probability of a next token given just the current token, e.g. to learn
that Barack -> Obama).
"""
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.components.embed_unembed import Embed, Unembed
from transformer_from_scratch.types import (
    BatchLogits,
    BatchResidualStream,
    BatchTokenIndices,
    Logits,
    TokenIndices,
)


class UnorderedIntegersDataset(Dataset):
    """Ordered integers dataset.

    The samples are contiguous sets of numbers (e.g. {3,1,2,4,0}). The targets are simply the next
    numbers after this (e.g. {4,2,3,5,1}).

    Bigram statistics are the frequencies with which one token comes after another. In this case,
    we're trying to train a model to predict n + 1 for each number in a set. Given that the
    zero-layer transformer should be able to learn these, it should be able to learn to output the
    token with index n + 1, for each token (of index n) in the input.
    """

    samples: list[TokenIndices] = []
    targets: list[Logits] = []

    def __init__(self, num_samples: int, d_vocab: int):
        """Initialise the dataset.

        Args:
            num_samples (int): Number of samples
            d_vocab (int): Vocab size (e.g. if 3 then it'll generate randomly ordered sets of
            {0,1,2}).
        """
        for _ in range(num_samples):
            # The sample is a random list of integers
            sample = list(range(0, d_vocab - 1))
            random.shuffle(sample)
            sample_tensor: TokenIndices = torch.tensor(sample, dtype=torch.long)
            self.samples.append(sample_tensor)

            # The target is this, plus 1 for each element in the sample
            target_indices = sample_tensor + 1
            target_one_hot: Logits = torch.nn.functional.one_hot(
                target_indices, num_classes=d_vocab
            )
            self.targets.append(target_one_hot.float())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[TokenIndices, Logits]:
        return self.samples[idx], self.targets[idx]


class ZeroLayerModel(nn.Module):
    """Zero Layer Model.

    Contains just embedding and unembedding layers.
    """

    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        config = TransformerConfig(d_vocab=d_vocab, d_model=d_model)
        self.embed = Embed(config)
        self.unembed = Unembed(config)

    def forward(self, input: BatchTokenIndices) -> BatchLogits:
        """Forward Pass."""
        residual_stream: BatchResidualStream = self.embed(input)
        return self.unembed(residual_stream)


def test_learn_order_integers() -> None:
    """Test that a model with just the embedding and unembedding can learn bigram statistics."""
    # Set random seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Set up the model
    d_vocab = 100
    d_model = d_vocab
    model = ZeroLayerModel(d_vocab, d_model)

    # Create the data
    samples = 1000
    epochs = 10
    dataset = UnorderedIntegersDataset(samples, d_vocab)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    for _epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            logits: BatchLogits = model(inputs)
            _, predicted = torch.max(logits, 2)
            _, target_indices = torch.max(targets, 2)
            total += targets.size(0) * targets.size(1)
            correct += (predicted == target_indices).sum().item()

    accuracy = correct / total
    assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
