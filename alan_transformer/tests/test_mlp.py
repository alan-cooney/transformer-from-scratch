import torch

from alan_transformer.mlp import FeedForward


class TestFeedForward:

    def test_weights_biases_ones(self, mocker):
        # Mock the weight random initialisation (use ones instead)
        mocker.patch("torch.rand", new=torch.ones)

        # Create the layer
        layer = FeedForward(d_model=2, d_hidden=2)

        # Create the mock input & expected output
        input = torch.tensor(
            [[1., 1], [-2, -2]]).unsqueeze(0)
        expected = torch.tensor([[7., 7], [1., 1]]).unsqueeze(0)

        # Check
        res = layer(input)
        assert torch.allclose(res, expected)
