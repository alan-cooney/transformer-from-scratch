import math

import torch

from alan_transformer.attention import (AttentionOutputType,
                                        AttentionPatternType, KeyType,
                                        MultiHeadAttention, QueryType,
                                        ResidualStreamType, ValueType)
from alan_transformer.tests.utils.mock_parameter import MockParameter


class TestAttention:
    def test_mask(self):
        """Test that it masks correctly"""
        attention_pattern: AttentionPatternType = torch.tensor(
            [[1., 1], [1, 1]]).unsqueeze(0).unsqueeze(0)
        expected: AttentionPatternType = torch.tensor(
            [[1., float("-inf")], [1, 1]]).unsqueeze(0).unsqueeze(0)

        attention_layer = MultiHeadAttention(d_head=2, d_model=4)
        res = attention_layer.mask(attention_pattern)

        assert torch.allclose(res, expected)

    def test_attention_simple(self):
        """Test a simple attention calculation"""
        # Create the query, key and value
        query: QueryType = torch.tensor(
            [[1., 2], [3, 4]]).unsqueeze(0).unsqueeze(0)
        key: KeyType = torch.tensor(
            [[5., 6], [7, 8]]).unsqueeze(0).unsqueeze(0)
        value: ValueType = torch.tensor(
            [[9., 10], [11, 12]]).unsqueeze(0).unsqueeze(0)

        # Create the expected output
        numerator = query @ key.transpose(-2, -1)
        denominator: float = float(math.sqrt(2))
        frac = numerator / denominator
        masked_attention_pattern = torch.tril(
            frac).masked_fill(frac == 0, float("-inf"))
        expected = torch.softmax(masked_attention_pattern, dim=-1) @ value

        # Create the attention layer
        attention_layer = MultiHeadAttention(d_head=2, d_model=4)

        # Calculate the output
        output: AttentionOutputType = attention_layer.attention(
            query, key, value)

        # Check the output
        assert torch.allclose(output, expected)

    def test_forward_mock_weights_ones(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameter)

        # Create a mock residual stream (that sums to 1)
        d_head: int = 64
        d_model: int = 768
        input: ResidualStreamType = torch.ones(
            1, 10, d_model) / d_model

        # Get the output
        attention_layer = MultiHeadAttention(d_head, d_model)
        res = attention_layer(input)

        # Expect the res to be the same as the input
        assert torch.allclose(res, input, atol=1e4)
