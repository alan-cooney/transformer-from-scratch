import torch

from alan_transformer.layer import Layer
from alan_transformer.tests.utils.mock_parameter import MockParameter


class TestLayer:
    def test_layer_forward_ones(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameter)

        input = torch.ones(1, 2, 2)

        layer = Layer(2, 2, 2)

        res = layer(input)

        # After normalization we should get zeros
        assert torch.allclose(res, torch.zeros_like(input))
