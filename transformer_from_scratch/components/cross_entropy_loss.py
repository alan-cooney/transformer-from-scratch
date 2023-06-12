"""Cross Entropy Loss."""

import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from transformer_from_scratch.types import BatchLogitsTT, BatchTokenIndicesTT
from transformer_from_scratch.types import TensorShapeLabels as D

BatchTargetIndicesTT = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1}"]
BatchTargetIndicesUnsqueezeTT = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1} ONE"]
BatchLogitsExceptLastTT = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION_MINUS_1} {D.VOCAB}",
]

def cross_entropy_loss(inputs: BatchTokenIndicesTT, logits: BatchLogitsTT):
    """Language Model Cross Entropy Loss

    Loss is calculated as the average negative log probs of the correct tokens.

    https://arxiv.org/pdf/1706.03762.pdf (p8)

    Params:
        Input: Input tokens
        logits: Logits from the forward pass

    Returns:
        Log loss
    """
    # Targets are inputs except for the first one (which we aren't predicting)
    # Logits except last exclude the last one (which we don't have a target for)
    target: BatchTargetIndicesTT = inputs[:, 1:]
    logits_except_last: BatchLogitsExceptLastTT = logits[
        :,
        :-1,
        :,
    ].float()

    log_probs: BatchLogitsExceptLastTT = F.log_softmax(
        logits_except_last,
        dim=-1,
    )

    # Predicted log probs are the log probs of the correct tokens
    index: BatchTargetIndicesUnsqueezeTT = target.unsqueeze(-1)
    predicted_log_probs = log_probs.gather(-1, index)

    # Cross entropy loss
    return -predicted_log_probs.mean()
