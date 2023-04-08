import math

import torch
from jaxtyping import Float
from torch import Tensor

from alan_transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_positional_encoding_for_specific_token(self):
        """Test by calculating a specific token and comparing to the module.

        Here we calculate the encoding for just a specific token and check it's the same as the
        module positional encoding.
        """
        # Set the embedding to zeros
        d_model: int = 4
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(1, 4, d_model)

        # Create the expected positional encoding
        position: int = 2
        expected_positional_encoding = []
        for dim in range(d_model):
            if dim % 2 == 0:
                pos_embed = math.sin(position / (10000 ** (dim / d_model)))
            else:
                pos_embed = math.cos(position / (10000 ** ((dim - 1) / d_model)))
            expected_positional_encoding.append(pos_embed)

        # Compare against the module
        layer = PositionalEncoding(d_model, 1024)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)
        assert torch.allclose(
            encoding[0, position, :],
            torch.tensor(expected_positional_encoding),
        )

    def test_numerically_stable(self):
        """Test that the encoding is numerically stable.

        This checks our Positional Encoding doesn't create any NaN values.
        """
        d_model: int = 768
        max_positions: int = 1024
        embedding: Float[Tensor, "batch pos d_model"] = (
            torch.zeros(1, max_positions, d_model) * 100
        )
        layer = PositionalEncoding(d_model, max_positions)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)
        assert not torch.isnan(encoding).any()

    def test_unique_positional_encodings(self):
        """Test that each token has a unique positional encoding vector."""
        d_model: int = 4
        max_positions: int = 10
        embedding: Float[Tensor, "batch pos d_model"] = torch.zeros(
            1,
            max_positions,
            d_model,
        )
        layer = PositionalEncoding(d_model, max_positions)
        encoding: Float[Tensor, "batch pos d_model"] = layer(embedding)

        unique_encodings = torch.unique(encoding, dim=1)
        assert unique_encodings.shape[1] == max_positions

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
