import torch
from torch.nn import functional as F

from alan_transformer.train import cross_entropy_loss
from alan_transformer.types import LogitsTT, TokensTT


class TestCrossEntropyLoss:
    def test_completely_accurate(self):
        # Create some sample inputs and outputs that are 100% accurate
        # Note the beginning and end of the sample data will be discarded by the
        # loss function
        sample_tokens = torch.tensor([[0, 1, 2, 3]])
        inputs: TokensTT = sample_tokens[:, :-1]  # 1,2,3

        # Note that with the logits we must set the correct values as very high,
        # as when softmax is applied (to get the probs) this will result in a
        # value on 1 for each correct token.
        logits: LogitsTT = F.one_hot(sample_tokens[:, 1:]).float() * 999999  # 2,3,4

        # Check the loss is 0
        loss = cross_entropy_loss(inputs, logits).item()
        assert loss == 0
