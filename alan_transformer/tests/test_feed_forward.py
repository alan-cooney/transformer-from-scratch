import torch

from alan_transformer.feed_forward import FeedForward
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes


class TestFeedForward:
    def test_weights_biases_ones(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameterOnes)

        # Create the layer
        layer = FeedForward(d_model=2, d_hidden=2)

        # Create the mock input & expected output
        input = torch.tensor([[1.0, 1], [-2, -2]]).unsqueeze(0)
        expected = torch.tensor([[7.0, 7], [1.0, 1]]).unsqueeze(0)

        # Check
        res = layer(input)
        assert torch.allclose(res, expected)
