import torch

from alan_transformer.transformer import Transformer


class TestTransformer:
    def test_architecture_snapshot(self, mocker, snapshot):
        # Mock the weight random initialisation (use ones instead)
        mocker.patch("torch.rand", new=torch.ones)

        d_vocab = 10
        model = Transformer(d_vocab=d_vocab)

        mock_tokens = torch.ones(1, 1, d_vocab) / d_vocab
        res = model(mock_tokens)

        snapshot.assert_match(res.tolist())
