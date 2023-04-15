"""Shared types

Note that we duplicate the dimension annotations in the docstring, so that they also show in e.g.
VSCode code hints (tooltips)."""

from enum import Enum
from jaxtyping import Float, Int
from torch import Tensor

#
# Dimension names
#

POS = "POS"
"""Position dimension.

Represents the position of tokens in the sequence."""

POS_MINUS_1 = "POS_MINUS_1"
"""Position dimension minus 1."""

BATCH = "BATCH"
"""Batch dimension.

Represents the number of prompts in the batch."""

D_MODEL = "D_MODEL"
"""Model dimension.

Represents the dimensions of token vectors in the residual stream."""

D_MODEL_HALF = "D_MODEL_HALF"
"""Half the model dimension.

Used for positional interweaved encoding."""

D_VOCAB = "D_VOCAB"
"""Vocabulary dimension.

Represents the number of tokens in the vocabulary."""

HEAD = "HEAD"
"""Head index dimension.

Represents the number of heads in the multi-head attention layer."""

D_HEAD = "D_HEAD"
"""Head dimension.

Represents the dimensions of token vectors in the attention calculations."""

D_HIDDEN = "D_HIDDEN"
"""Feed Forward (MLP) hidden dimension."""

#
# Tensor Types
#

TokenIndicesTT = Int[Tensor, f" {POS}"]
"""Token Indices

POS
"""

BatchTokenIndicesTT = Int[Tensor, f"{BATCH} {POS}"]
"""Batch Token Indices

BATCH POS
"""

ResidualStreamTT = Float[Tensor, f"{POS} {D_MODEL}"]
"""Residual stream

POS D_MODEL
"""

BatchResidualStreamTT = Float[Tensor, f"{BATCH} {POS} {D_MODEL}"]
"""Batch Residual stream

BATCH POS D_MODEL
"""

LogitsTT = Float[Tensor, f"{POS} {D_VOCAB}"]
"""Logits

POS D_VOCAB
"""

BatchLogitsTT = Float[Tensor, f"{BATCH} {POS} {D_VOCAB}"]
"""Batch of logits

BATCH POS D_VOCAB
"""
