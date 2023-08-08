"""Multi-Head Attention Module"""
import math

import torch
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.types import BatchResidualStream
from transformer_from_scratch.types import TensorShapeLabels as D

BatchQuery = Float[Tensor, f"{D.BATCH} {D.HEAD} DEST {D.HEAD_FEATURE}"]
BatchKey = Float[Tensor, f"{D.BATCH} {D.HEAD} SRC {D.HEAD_FEATURE}"]
BatchKeyTranspose = Float[Tensor, f"{D.BATCH} {D.HEAD} {D.HEAD_FEATURE} SRC"]
BatchValue = Float[Tensor, f"{D.BATCH} {D.HEAD} SRC {D.HEAD_FEATURE}"]
BatchQKVWeight = Float[Tensor, f"{D.HEAD} {D.RESIDUAL_FEATURE} {D.HEAD_FEATURE}"]
BatchWeightOutput = Float[Tensor, f"{D.RESIDUAL_FEATURE} {D.RESIDUAL_FEATURE}"]
BatchAttentionPattern = Float[Tensor, f"{D.BATCH} {D.HEAD} DEST SRC"]
BatchAttentionOutput = Float[
    Tensor, f"{D.BATCH} {D.HEAD} {D.POSITION} {D.HEAD_FEATURE}"
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Module.

    This module takes an input residual stream and applies multiple attention heads that operate
    in parallel to process and transform the information. The output of these attention heads
    is then combined and returned as the result, which can be added back to the residual stream
    in the Transformer architecture.
    """

    minus_infinity_triangle: Float[Tensor, "max_tokens max_tokens"]

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the MultiHeadAttention module."""
        super().__init__()

        # Check the params
        if config.d_model % config.d_head != 0:
            raise ValueError("d_model must be a multiple of d_head")

        # Set number of heads
        n_heads: int = int(config.d_model / config.d_head)

        # Store d_head sqrt for the attention calculation
        self.d_head_sqrt: float = math.sqrt(config.d_head)

        # Create the parameters
        self.weight_query: BatchQKVWeight = nn.Parameter(
            torch.empty(n_heads, config.d_model, config.d_head),
        )
        self.weight_key: BatchQKVWeight = nn.Parameter(
            torch.empty(n_heads, config.d_model, config.d_head),
        )
        self.weight_value: BatchQKVWeight = nn.Parameter(
            torch.empty(n_heads, config.d_model, config.d_head),
        )
        self.weight_out: BatchWeightOutput = nn.Parameter(
            torch.empty(config.d_model, config.d_model),
        )

        # Initialise the weights
        # Use Kaiming for the QKV weights as we have non-linear functions after them. Use Xavier for
        # the output weights as we have no activation function after it.
        nn.init.kaiming_normal_(self.weight_query)
        nn.init.kaiming_normal_(self.weight_key)
        nn.init.kaiming_normal_(self.weight_value)
        nn.init.xavier_normal_(self.weight_out)

        # Create the minus infinity mask
        minus_infinity = torch.full((config.n_ctx, config.n_ctx), float("-inf"))
        minus_infinity_triangle = torch.triu(minus_infinity, diagonal=1)
        self.register_buffer("minus_infinity_triangle", minus_infinity_triangle)

    def mask(self, attention_pattern: BatchAttentionPattern) -> BatchAttentionPattern:
        """Mask the attention pattern.

        Each attention pattern is of shape (dest, src), and each element represents the attention
        that a destination token should pay to a source token. For example if we have the tokens
        "The| man| wal|ked| the| dog" (6 tokens) then we would have a 6x6 attention pattern matrix,
        where for example element 3,1 would be the attention that the 3rd token ("ked") should pay
        to the first token ("man", noting it's 0-indexed).

        An important part however is that for example we don't want the first token "The" to pay
        attention to any of the other tokens as these are all in the future. To do this, we apply an
        upper triangular mask to the attention pattern, which sets all values above the diagonal
        line to minus infinity. This is what the mask method does.

        https://arxiv.org/pdf/1706.03762.pdf (p6)
        """
        n_tokens: int = attention_pattern.shape[-1]
        return attention_pattern + self.minus_infinity_triangle[:n_tokens, :n_tokens]

    def attention(
        self,
        query: BatchQuery,
        key: BatchKey,
        value: BatchValue,
    ) -> BatchAttentionOutput:
        """Attention Calculation.

        The attention calculation does two things for each destination token - it both moves
        information from source tokens to the destination token, and it also transforms information
        (by multiplying the resulting values by the value weights). The calculation is as follows:

        Attention(Q,K,V) = softmax( (Q K^T) / (sqrt(d_head)) ) * V

        https://arxiv.org/pdf/1706.03762.pdf (p4)

        Args:
            query (Query): Query
            key (Key): Key
            value (Value): Value

        Returns:
            AttentionOutput: Attention Output (softmax * value)
        """
        # Calculate the numerator
        key_transpose: BatchKeyTranspose = rearrange(
            key,
            "batch head pos d_head -> batch head d_head pos",
        )
        numerator: BatchAttentionPattern = query @ key_transpose

        # Apply softmax over the attention pattern
        attention_pattern: BatchAttentionPattern = numerator / self.d_head_sqrt
        masked_attention: BatchAttentionPattern = self.mask(attention_pattern)
        softmax_part: BatchAttentionPattern = torch.softmax(
            masked_attention,
            dim=-1,  # Apply over the last (src) dimension
        )

        return einsum(
            "batch head dest src, batch head src d_head -> batch head dest d_head",
            softmax_part,
            value,
        )

    def forward(self, residual_stream: BatchResidualStream) -> BatchResidualStream:
        """Attention layer forward pass.

        https://arxiv.org/pdf/1706.03762.pdf (p5)
        """
        # Create the query, key and value
        query: BatchQuery = einsum(
            "batch pos d_model, head d_model d_head -> batch head pos d_head",
            residual_stream,
            self.weight_query,
        )
        key: BatchKey = einsum(
            "batch pos d_model, head d_model d_head -> batch head pos d_head",
            residual_stream,
            self.weight_key,
        )
        value: BatchValue = einsum(
            "batch pos d_model, head d_model d_head -> batch head pos d_head",
            residual_stream,
            self.weight_value,
        )

        # Get the attention & concat
        attn: BatchAttentionOutput = self.attention(query, key, value)
        attn_concat: BatchResidualStream = rearrange(
            attn,
            # (head d_head) is the same size as d_model
            "batch head pos d_head -> batch pos (head d_head)",
        )

        # Multiply by W_O
        multi_head_out: BatchResidualStream = einsum(
            "batch pos d_model_a, d_model_a d_model_b -> batch pos d_model_b",
            attn_concat,
            self.weight_out,
        )

        # Return the attention output
        return multi_head_out
