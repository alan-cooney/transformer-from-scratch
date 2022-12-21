import math

import torch
from torchtyping import TensorType as TT

from alan_transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_pos_encoding_2_token(self):
        """Test the pos=2 token"""
        # Set the embedding to zeros
        d_model: int = 4
        embedding: TT["batch", "pos", "d_model"] = torch.zeros(
            1, 4, d_model)

        # Create the expected positional encoding
        position: int = 2
        expected_positional_encoding = []
        for dim in range(d_model):
            if dim % 2 == 0:
                pos_embed = math.sin(position / (10000 ** (dim/d_model)))
            else:
                pos_embed = math.cos(position / (10000 ** ((dim - 1)/d_model)))
            expected_positional_encoding.append(pos_embed)

        # Compare against the res
        layer = PositionalEncoding(d_model, 1024)
        encoding: TT["batch", "pos", "d_model"] = layer(embedding)
        assert torch.allclose(encoding[0, position, :], torch.tensor(
            expected_positional_encoding))
