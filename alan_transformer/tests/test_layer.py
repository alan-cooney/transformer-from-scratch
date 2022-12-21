from torch import nn
import torch

from alan_transformer.layer import Layer


class TestLayer:
    def test_layer_forward_ones(self, mocker):
        # Mock the weight random initialisation (use ones instead)
        mocker.patch("torch.rand", new=torch.ones)

        input = torch.ones(1, 2, 2)

        layer = Layer(2, 2, 2)

        res = layer(input)

        # After normalization we should get zeros
        assert torch.allclose(res, torch.zeros_like(input))
