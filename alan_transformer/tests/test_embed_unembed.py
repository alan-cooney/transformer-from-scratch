import math

import torch

from alan_transformer.embed_unembed import Embed, Unembed
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes
from jaxtyping import Float
from torch import Tensor


class TestEmbed:
    def test_embed(self, mocker):
        # Create the layer
        d_vocab = 10
        d_model = 5
        n_tokens = 1
        layer = Embed(d_vocab, d_model)

        # Set the dummy parameters so that each token is embedded as a list of
        # that number (e.g. token 3 -> [3 for _ in d_model])
        state = layer.state_dict()
        new_embed_weights: Float[Tensor, "vocab d_model"] = torch.arange(
            0, d_vocab).repeat(d_model, 1).T
        state["embed_weights"] = new_embed_weights
        # Set the bias as 0 for simplicity
        state["embed_bias"] = torch.zeros(d_model)
        layer.load_state_dict(state)

        mock_tokens = torch.tensor([3, 1, 0])
        expected = mock_tokens.repeat(
            d_model, 1).T * math.sqrt(d_model)
        res = layer(mock_tokens.unsqueeze(0))

        assert torch.allclose(res, expected.unsqueeze(0))


class TestUnembed:
    def test_unembed(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameterOnes)

        # Create the mock tokens in the residual stream
        # Divide by d_model so that after the multiplication with the weights, we
        # get ones
        d_vocab = 10
        d_model = 5
        n_tokens = 5
        tokens = torch.ones((1, n_tokens, d_model)) / d_model

        res = Unembed(d_vocab, d_model)(tokens)
        # Expected has plus one for the bias
        expected = torch.ones((1, n_tokens, d_vocab)) + 1

        assert torch.allclose(res, expected)
