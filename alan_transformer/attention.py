import math

import torch
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
from torchtyping import TensorType as TT

QueryTT = TT["batch", "head", "dest", "d_head"]
KeyTT = TT["batch", "head", "src", "d_head"]
ValueTT = TT["batch", "head", "src", "d_head"]
QKVWeightTT = TT["head", "d_model", "d_head"]
AttentionPatternTT = TT["batch", "head", "dest", "src"]
AttentionOutputTT = TT["batch", "head", "pos", "d_head"]
ResidualStreamTT = TT["batch", "pos", "d_model"]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Sub-Layer"""

    def __init__(self, d_head: int = 64, d_model: int = 768, max_tokens: int = 1024) -> None:
        """Create the attention layer"""
        super().__init__()

        # Check the params
        assert d_model % d_head == 0, "d_model must be a multiple of d_head"

        # Set number of heads
        n_heads: int = int(d_model / d_head)

        # Store d_head sqrt for the attention calculation
        self.d_head_sqrt: float = math.sqrt(d_head)

        # Create the parameters
        self.weight_query: QKVWeightTT = nn.Parameter(
            torch.empty(n_heads, d_model, d_head))
        self.weight_key: QKVWeightTT = nn.Parameter(
            torch.empty(n_heads, d_model, d_head))
        self.weight_value: QKVWeightTT = nn.Parameter(
            torch.empty(n_heads, d_model, d_head))
        self.weight_out: TT["d_model", "d_model"] = nn.Parameter(
            torch.empty(d_model, d_model))

        # Create the minus infinity mask
        minus_infinity = torch.full((max_tokens, max_tokens), float("-inf"))
        minus_infinity_triangle = torch.triu(minus_infinity, diagonal=1)
        self.register_buffer("minus_infinity_triangle",
                             minus_infinity_triangle)

    def mask(self, attention_pattern: AttentionPatternTT) -> AttentionPatternTT:
        """Mask the attention pattern

        Values are masked out with minus infinity

        https://arxiv.org/pdf/1706.03762.pdf (p6)
        """
        n_tokens: int = attention_pattern.shape[-1]
        return attention_pattern + self.minus_infinity_triangle[:n_tokens, :n_tokens]

    def attention(self, query: QueryTT, key: KeyTT, value: ValueTT) -> AttentionOutputTT:
        """Attention Calculation

        Attention(Q,K,V) = softmax( (Q K^T) / (sqrt(d_head)) ) * V

        https://arxiv.org/pdf/1706.03762.pdf (p4)

        Args:
            query (QueryTT): Query
            key (KeyTT): Key
            value (ValueTT): Value

        Returns:
            AttentionOutputTT: Attention Output (softmax * value)
        """
        # Calculate the numerator
        key_transpose: TT["batch", "head", "d_head", "pos"] = rearrange(
            key,
            "batch head pos d_head -> batch head d_head pos"
        )
        numerator: AttentionPatternTT = query @ key_transpose

        # Apply softmax over the attention pattern
        attention_pattern: AttentionPatternTT = numerator / self.d_head_sqrt
        masked_attention: AttentionPatternTT = self.mask(
            attention_pattern)
        softmax_part: AttentionPatternTT = torch.softmax(
            masked_attention,
            dim=-1  # Apply over the last (src) dimension
        )

        return einsum(
            "batch head dest src, batch head src d_head -> batch head dest d_head",
            softmax_part,
            value
        )

    def forward(self, residual_stream: ResidualStreamTT) -> ResidualStreamTT:
        """Attention layer forward pass

        Note the residual stream is not added back (or normalized)

        https://arxiv.org/pdf/1706.03762.pdf (p5)
        """
        # Create the query, key and value
        query: QueryTT = einsum("batch pos d_model, head d_model d_head -> batch head pos d_head",
                                residual_stream, self.weight_query)
        key: KeyTT = einsum("batch pos d_model, head d_model d_head -> batch head pos d_head",
                            residual_stream, self.weight_key)
        value: ValueTT = einsum("batch pos d_model, head d_model d_head -> batch head pos d_head",
                                residual_stream, self.weight_value)

        # Get the attention & concat
        attn: AttentionOutputTT = self.attention(query, key, value)
        attn_concat: ResidualStreamTT = rearrange(
            attn,
            # (head d_head) is the same size as d_model
            "batch head pos d_head -> batch pos (head d_head)"
        )

        # Multiply by W_O
        multi_head_out: ResidualStreamTT = einsum(
            "batch pos d_model, d_model d_model -> batch pos d_model",
            attn_concat, self.weight_out)

        # Return the attention output
        return multi_head_out
