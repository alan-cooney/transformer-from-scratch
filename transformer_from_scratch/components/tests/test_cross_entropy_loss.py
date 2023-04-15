"""Test Cross Entropy Loss."""
import torch
from torch.nn import functional as F

from transformer_from_scratch.components.cross_entropy_loss import cross_entropy_loss
from transformer_from_scratch.types import BatchLogitsTT, BatchTokenIndicesTT


def test_where_logits_are_fully_accurate():
    """Test loss where the logits are fully accurate (so it should be 0)."""
    # Create some sample inputs and outputs that are 100% accurate
    # Note the beginning and end of the sample data will be discarded by the
    # loss function
    sample_tokens = torch.tensor([[0, 1, 2, 3]])
    inputs: BatchTokenIndicesTT = sample_tokens[:, :-1]  # 1,2,3

    # Note that with the logits we must set the correct values as very high,
    # as when softmax is applied (to get the probs) this will result in a
    # value on 1 for each correct token.
    logits: BatchLogitsTT = F.one_hot(sample_tokens[:, 1:]).float() * 999999  # 2,3,4

    # Check the loss is 0
    loss = cross_entropy_loss(inputs, logits).item()
    assert loss == 0
