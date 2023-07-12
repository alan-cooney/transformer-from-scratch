"""Positional Encoding Module Tests."""
import math

import torch

from transformer_from_scratch.components.positional_encoding import (
    SinusoidalPositionalEncoding,
)
from transformer_from_scratch.types import BatchResidualStreamTT, ResidualStreamTT


def test_positional_encoding_each_token_unique():
    """Test that each token will have a unique positional encoding vector."""
    # Use the larger of the two model sizes from the original paper
    d_model: int = 1024
    sequence_length: int = 512

    # Create a sequence (batch size 1) of all zeros (so that we'll end up with just the positional
    # encoding once it is encoded)
    embedding: BatchResidualStreamTT = torch.zeros(
        1,
        sequence_length,
        d_model,
    )

    # Add positional encoding
    layer = SinusoidalPositionalEncoding(d_model, sequence_length)
    encoding: ResidualStreamTT = layer(embedding).squeeze(0)

    # Check recursively that each position vector is unique
    for pos in range(sequence_length):
        for compare_pos in range(pos + 1, sequence_length):
            assert not torch.allclose(
                encoding[pos],
                encoding[compare_pos],
            )


def test_linear_function_for_relative_positions():
    """Test that relative positions can be obtained with just matrix multiplication.

    Check that the positional encoding of a token at position `initial_position + relative_position`
    can be calculated as a linear function of the positional encoding of the token at position
    `initial_position`.
    """
    d_model: int = 4
    max_positions: int = 10
    relative_position: int = 2
    initial_position: int = 3
    embedding: BatchResidualStreamTT = torch.zeros(
        1,
        max_positions,
        d_model,
    )
    layer = SinusoidalPositionalEncoding(d_model, max_positions)
    encoding: BatchResidualStreamTT = layer(embedding)

    # Compute the linear function matrix M
    angle_scaling_factors = relative_position / (
        10000 ** (torch.arange(0, d_model, 2) / d_model)
    )
    linear_fn_matrix = torch.zeros(d_model, d_model)
    linear_fn_matrix[0::2, 0::2] = torch.diag(torch.cos(angle_scaling_factors))
    linear_fn_matrix[0::2, 1::2] = torch.diag(torch.sin(angle_scaling_factors))
    linear_fn_matrix[1::2, 0::2] = torch.diag(-torch.sin(angle_scaling_factors))
    linear_fn_matrix[1::2, 1::2] = torch.diag(torch.cos(angle_scaling_factors))

    # Calculate the encoding for pos+k as a linear function of the encoding at pos
    calculated_encoding = torch.matmul(
        linear_fn_matrix, encoding[0, initial_position].unsqueeze(-1)
    ).squeeze()

    # Compare the calculated encoding to the actual encoding at pos+k
    assert torch.allclose(
        encoding[0, initial_position + relative_position], calculated_encoding
    )


def test_positional_encoding_against_specific_position_and_dimension_scalar_calculation():
    """Check the positional encoding is as expected for a specific position and dimension.

    This uses a scalar calculation (as opposed to the vectorized calculation in the module).
    """
    # Set the embedding to zeros
    d_model: int = 4
    embedding: BatchResidualStreamTT = torch.zeros(1, 4, d_model)

    # Check for a specific position & dimension
    position: int = 2
    dimension: int = 3
    expected = math.cos(position / (10000 ** ((dimension - 1) / d_model)))

    # Compare against the module
    layer = SinusoidalPositionalEncoding(d_model, 1024)
    encoding: BatchResidualStreamTT = layer(embedding)
    assert torch.allclose(
        encoding[0, position, dimension],
        torch.tensor(expected),
    )


def test_numerically_stable():
    """Test that the encoding is numerically stable.

    Check our Positional Encoding doesn't create any NaN values.
    """
    d_model: int = 768
    max_positions: int = 1024
    embedding: BatchResidualStreamTT = torch.zeros(1, max_positions, d_model) * 100
    layer = SinusoidalPositionalEncoding(d_model, max_positions)
    encoding: BatchResidualStreamTT = layer(embedding)
    assert not torch.isnan(encoding).any()


def test_positional_encoding_same_across_batch_items():
    """Test that the positional encoding is the same for each batch item."""
    d_model: int = 4
    max_positions: int = 10
    batch_items: int = 2
    embedding: BatchResidualStreamTT = torch.zeros(
        batch_items,
        max_positions,
        d_model,
    )
    layer = SinusoidalPositionalEncoding(d_model, max_positions)
    encoding: BatchResidualStreamTT = layer(embedding)

    # Check that the positional encoding is the same for both bach items
    assert torch.allclose(encoding[0], encoding[1])
