import torch

from alan_transformer.transformer import Transformer
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes


class TestTransformer:
    """Transformer tests

    Unlike the constituent parts, testing the full transformer is challenging as
    there are a lot of calculations in each layer. As such we'll just use a
    snapshot test here.
    """

    def test_architecture_snapshot(self, mocker, snapshot):
        # Mock the weight initialisation (use ones instead)
        mocker.patch("torch.nn.Parameter", new=MockParameterOnes)

        d_vocab = 10
        model = Transformer(d_vocab=d_vocab)

        mock_tokens = torch.ones(1, 1).long()
        res = model(mock_tokens)

        snapshot.assert_match(res.tolist())
