"""Multi-Head Attention Module Tests.

These tests try to verify that the attention heads can do their two key tasks - moving information
between tokens, and transforming information. We also unit test some of the underlying methods used
(e.g. the attention mask) to ensure that they work as expected."""
import math
import random

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from transformer_from_scratch.components.attention import (
    BatchAttentionOutput,
    BatchAttentionPattern,
    BatchKey,
    BatchQuery,
    BatchValue,
    MultiHeadAttention,
)
from transformer_from_scratch.components.config import TransformerConfig
from transformer_from_scratch.types import BatchResidualStream, ResidualStream
from transformer_from_scratch.types import TensorShapeLabels as D


class TestMask:
    """Attention Mask Tests."""

    def test_mask(self):
        """Test that it masks correctly"""
        attention_pattern: BatchAttentionPattern = (
            torch.tensor([[1.0, 1], [1, 1]]).unsqueeze(0).unsqueeze(0)
        )
        expected: BatchAttentionPattern = (
            torch.tensor([[1.0, float("-inf")], [1, 1]]).unsqueeze(0).unsqueeze(0)
        )

        attention_layer = MultiHeadAttention(
            TransformerConfig(d_head=2, d_model=4, n_ctx=2)
        )
        res = attention_layer.mask(attention_pattern)

        assert torch.allclose(res, expected)


class TestAttentionCalculation:
    """Attention Calculation Tests."""

    def test_attention_simple(self):
        """Test a simple attention calculation"""
        # Create the query, key and value
        query: BatchQuery = torch.tensor([[1.0, 2], [3, 4]]).unsqueeze(0).unsqueeze(0)
        key: BatchKey = torch.tensor([[5.0, 6], [7, 8]]).unsqueeze(0).unsqueeze(0)
        value: BatchValue = (
            torch.tensor([[9.0, 10], [11, 12]]).unsqueeze(0).unsqueeze(0)
        )

        # Create the expected output
        numerator = query @ key.transpose(-2, -1)
        denominator: float = float(math.sqrt(2))
        frac = numerator / denominator
        masked_attention_pattern = torch.tril(frac).masked_fill(
            frac == 0,
            float("-inf"),
        )
        expected = torch.softmax(masked_attention_pattern, dim=-1) @ value

        # Create the attention layer
        attention_layer = MultiHeadAttention(
            TransformerConfig(d_head=2, d_model=4, n_ctx=10)
        )

        # Calculate the output
        output: BatchAttentionOutput = attention_layer.attention(query, key, value)

        # Check the output
        assert torch.allclose(output, expected)


ResidualStreamToken = Float[Tensor, f" {D.RESIDUAL_FEATURE}"]
BatchResidualStreamToken = Float[Tensor, f"{D.BATCH} {D.RESIDUAL_FEATURE}"]


class FlaggedToken(Dataset):
    """Flagged Token Dataset

    Dataset designed to test that the model can learn to attend to a token, and that it can learn a
    what to do after attending to that token.

    In this dataset, we have a source token that has a 1 (flag) in the first dimension and random
    values in the others. All other tokens have a 0.1 in the first dimension, and also have random
    values in the others. In this scenario, a single attention layer should be able to attend to the
    flagged token, and output 1.2x its values.

    The reason the flag works is that the model can learn key weights (W_k) such that the key (K) is
    [10,0,...,0] for the token with the flag and [0.1,0,...,0] otherwise. Similarly, the model can
    learn query weights (Q_q) such that the query (Q) becomes [1,0,...,0] for the destination
    (last) token and [0.01, 0,.000,0 for all other tokens]. When multiplied together we get a
    pre-softmax attention pattern of I * 0.01, except for the element corresponding to the
    destination and source tokens which is 1. After softmax, the attention pattern then becomes
    close to 1 for the destination-source combination and close to 0 for all other combinations.

    Finally the value weights can become simply be I * 1.2, so that the output is 1.2x the source
    token.
    """

    samples: list[ResidualStream] = []
    targets: list[ResidualStreamToken] = []

    def __init__(self, sequence_length: int, d_model: int, num_samples: int):
        for _ in range(num_samples):
            # Source token
            source_token: ResidualStreamToken = torch.rand((1, d_model))
            source_token[0, 0] = 10

            # Other tokens
            other_tokens = torch.rand((sequence_length - 1, d_model))
            other_tokens[:, 0] = 0.1

            # Create the sample & targets
            sample = torch.concat([source_token, other_tokens])
            target = source_token * 1.2
            self.samples.append(sample)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]


class TestMultiHeadAttention:
    """Multi-Head Attention Module Tests."""

    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_attend_flagged_token(self, num_heads: int):
        """Test that the model can attend to a specific (flagged) token."""
        # Set random seeds
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        # Parameters
        samples = 100000
        d_model = 16
        d_head = int(d_model / num_heads)
        sequence_length = 10

        # Model
        model = MultiHeadAttention(
            TransformerConfig(d_head=d_head, d_model=d_model, n_ctx=sequence_length)
        )

        # Dataset
        dataset = FlaggedToken(sequence_length, d_model, samples)
        train_size = int(len(dataset) * 0.95)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1)

        # Loss & optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        # Train
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs: BatchResidualStream = model(inputs)
            predicted_next_tokens: BatchResidualStreamToken = outputs[:, -1, :]
            loss: BatchResidualStreamToken = criterion(
                predicted_next_tokens, targets.squeeze(1)
            )
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predicted_next_tokens = outputs[:, -1, :].squeeze(1)
                for index, prediction in enumerate(predicted_next_tokens):
                    expected = targets[index]
                    # Exclude the flag token
                    if torch.allclose(prediction, expected, rtol=0.1, atol=0.1):
                        correct += 1
                    total += 1

        accuracy = correct / total
        assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
