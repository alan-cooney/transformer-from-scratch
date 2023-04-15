from torch import nn

from alan_transformer.attention import MultiHeadAttention
from alan_transformer.feed_forward import FeedForward
from alan_transformer.types import BatchResidualStreamTT


class Layer(nn.Module):
    """Full layer (attention + MLP).

    https://arxiv.org/pdf/1706.03762.pdf (p3)
    """

    def __init__(
        self,
        d_model: int = 768,
        d_head: int = 64,
        d_hidden: int = 2048,
        max_tokens: int = 1024,
    ):
        """Initialise the full layer."""
        super().__init__()

        # Create the feed forward and attention sub-layers
        self.feed_forward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_head, d_model, max_tokens)
        self.layer_norm_attn = nn.LayerNorm(d_model)

    def forward(self, residual_stream: BatchResidualStreamTT) -> BatchResidualStreamTT:
        """Forward pass."""
        # Attention
        attn = self.attention(residual_stream)
        attn_add_norm = residual_stream + self.layer_norm_ff(attn)

        # Feed forward
        ff = self.feed_forward(attn_add_norm)
        return attn_add_norm + self.layer_norm_attn(ff)
