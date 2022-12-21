import math

import torch

from alan_transformer.embed_unembed import Embed, Unembed
from alan_transformer.tests.utils.mock_parameter import MockParameter


class TestEmbed:
    def test_embed(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameter)

        # Create the mock tokens pre-embedding
        # Divide by d_vocab so that after the multiplication with the weights, we
        # get ones
        d_vocab = 10
        d_model = 5
        n_tokens = 5
        tokens = torch.ones((1, n_tokens, d_vocab)) / d_vocab

        res = Embed(d_vocab, d_model)(tokens)
        # Expected has plus one for the bias
        expected = torch.ones((1, n_tokens, d_model)) * math.sqrt(d_model) + 1

        assert torch.allclose(res, expected)


class TestUnembed:
    def test_unembed(self, mocker):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameter)

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
