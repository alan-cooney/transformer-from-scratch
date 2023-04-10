import math

import torch
from jaxtyping import Float
from torch import Tensor

from alan_transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_positional_encoding_each_token_unique(self):
        """Test that each token will have a unique positional encoding vector."""
        d_model: int = 1024  # Larger model from the original paper
        sequence_length: int = 512
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(
            1,
            sequence_length,
            d_model,
        )
        layer = PositionalEncoding(d_model, sequence_length)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)
        single_batch_encoding: Float[Tensor, "pos d_model"] = encoding[0]

        # Check that each position vector is unique
        for pos in range(sequence_length):
            for compare_pos in range(pos + 1, sequence_length):
                assert not torch.allclose(
                    single_batch_encoding[pos],
                    single_batch_encoding[compare_pos],
                )

    def test_linear_function_for_relative_positions(self):
        """Test that relative positions can be obtained with just matrix multiplication.

        Check that the positional encoding of a token at position pos + k can be calculated as a
        linear function of the positional encoding of the token at position pos.
        """
        d_model: int = 4
        max_positions: int = 10
        k: int = 2
        pos: int = 3
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(
            1,
            max_positions,
            d_model,
        )
        layer = PositionalEncoding(d_model, max_positions)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)

        # Compute the linear function matrix M
        B = k / (10000 ** (torch.arange(0, d_model, 2) / d_model))
        M = torch.zeros(d_model, d_model)
        M[0::2, 0::2] = torch.diag(torch.cos(B))
        M[0::2, 1::2] = torch.diag(torch.sin(B))
        M[1::2, 0::2] = torch.diag(-torch.sin(B))
        M[1::2, 1::2] = torch.diag(torch.cos(B))

        # Calculate the encoding for pos+k as a linear function of the encoding at pos
        calculated_encoding = torch.matmul(M, encoding[0, pos].unsqueeze(-1)).squeeze()

        # Compare the calculated encoding to the actual encoding at pos+k
        assert torch.allclose(encoding[0, pos + k], calculated_encoding)

    def test_positional_encoding_against_specific_position_and_dimension_scalar_calculation(
        self,
    ):
        """Check the positional encoding is as expected for a specific position and dimension.

        This uses a scalar calculation (as opposed to the vectorized calculation in the module).
        """
        # Set the embedding to zeros
        d_model: int = 4
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(1, 4, d_model)

        # Check for a specific position & dimension
        position: int = 2
        dimension: int = 3
        expected = math.cos(position / (10000 ** ((dimension - 1) / d_model)))

        # Compare against the module
        layer = PositionalEncoding(d_model, 1024)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)
        assert torch.allclose(
            encoding[0, position, dimension],
            torch.tensor(expected),
        )

    def test_numerically_stable(self):
        """Test that the encoding is numerically stable.

        Check our Positional Encoding doesn't create any NaN values.
        """
        d_model: int = 768
        max_positions: int = 1024
        embedding: Float[Tensor, "batch pos d_model"] = (
            torch.zeros(1, max_positions, d_model) * 100
        )
        layer = PositionalEncoding(d_model, max_positions)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)
        assert not torch.isnan(encoding).any()

    def test_positional_encoding_same_across_batch_items(self):
        """Test that the positional encoding is the same for each batch item."""
        d_model: int = 4
        max_positions: int = 10
        batch_items: int = 2
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(
            batch_items,
            max_positions,
            d_model,
        )
        layer = PositionalEncoding(d_model, max_positions)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)

        # Check that the positional encoding is the same for both bach items
        assert torch.allclose(encoding[0], encoding[1])
