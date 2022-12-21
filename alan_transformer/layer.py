
from torch import nn
from alan_transformer.attention import MultiHeadAttention
from alan_transformer.mlp import FeedForward
from torchtyping import TensorType as TT

ResidualStreamType = TT["batch", "pos", "d_model"]


class Layer(nn.Module):
    """Full layer (attention + MLP)

    https://arxiv.org/pdf/1706.03762.pdf (p3)"""

    def __init__(self, d_model: int = 768, d_head: int = 64, d_hidden: int = 2048):
        super().__init__()

        # Create the feed forward and attention sub-layers
        self.feed_forward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_head=d_head, d_model=d_model)
        self.layer_norm_attn = nn.LayerNorm(d_model)

    def forward(self, residual_stream: ResidualStreamType) -> ResidualStreamType:
        """Forward pass"""
        # Attention
        attn = self.attention(residual_stream)
        attn_add_norm = self.layer_norm_ff(residual_stream + attn)

        # Feed forward
        ff = self.feed_forward(attn_add_norm)
        ff_add_norm = self.layer_norm_attn(attn_add_norm + ff)

        return ff_add_norm
