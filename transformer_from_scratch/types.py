"""Shared types

Note that we duplicate the dimension annotations in the docstring, so that they also show in e.g.
VSCode code hints (tooltips)."""

from jaxtyping import Float, Int
from torch import Tensor

# Hacky way to import a consistent version of StrEnum across python versions
# https://tomwojcik.com/posts/2023-01-02/python-311-str-enum-breaking-change
try:
    from enum import StrEnum  # type: ignore
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backwards compatible string enum."""


class TensorShapeLabels(StrEnum):  # type: ignore
    """Tensor Shape Labels

    Labels (aka dimension names) used in describing tensors.
    """

    BATCH = "BATCH"
    """Batch index within a batch."""

    POSITION = "POSITION"
    """Position within a prompt."""

    RESIDUAL_FEATURE = "RESIDUAL_FEATURE"
    """Residual stream feature within a token (aka `d_model`)."""

    VOCAB = "VOCAB"
    """Vocabulary index (token index) within the vocabulary (aka `d_vocab`)."""

    HEAD = "HEAD"
    """Head index within the multi-head attention layer."""

    HEAD_FEATURE = "HEAD_FEATURE"
    """Head feature within a token."""

    HIDDEN_FEATURE = "HIDDEN_FEATURE"
    """Hidden feature within a token (aka `d_hidden`)."""

    POSITION_MINUS_1 = "POSITION_MINUS_1"
    """Position within a prompt (that excludes the first or last token)."""

    RESIDUAL_FEATURE_HALF = "RESIDUAL_FEATURE_HALF"
    """Residual feature within a token,where we only store half the features in one tensor e.g. for
    positional encoding aka `d_model / 2`)."""


# Alias TensorShapeLabels for convenience
D = TensorShapeLabels  # pylint: disable=invalid-name


TokenIndicesTT = Int[Tensor, f" {D.POSITION}"]
"""Token Indices.

Shape: (POSITION,)
"""

BatchTokenIndicesTT = Int[Tensor, f"{D.BATCH} {D.POSITION}"]
"""Batch Token Indices.

Shape: (BATCH, POSITION)
"""

ResidualStreamTT = Float[Tensor, f"{D.POSITION} {D.RESIDUAL_FEATURE}"]
"""Residual Stream.

Shape: (POSITION, RESIDUAL_FEATURE)
"""

BatchResidualStreamTT = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION} {D.RESIDUAL_FEATURE}",
]
"""Batch Residual Stream.

Shape: (BATCH, POSITION, RESIDUAL_FEATURE)
"""

LogitsTT = Float[Tensor, f"{D.POSITION} {D.VOCAB}"]
"""Logits.

Shape: (POSITION, VOCAB)
"""

BatchLogitsTT = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION} {D.VOCAB}",
]
"""Batch of Logits.

Shape: (BATCH, POSITION, VOCAB)
"""
