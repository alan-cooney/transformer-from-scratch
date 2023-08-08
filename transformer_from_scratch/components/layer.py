"""Full layer (attention + feed forward)."""
from torch import nn

from transformer_from_scratch.components.attention import MultiHeadAttention
from transformer_from_scratch.components.config import TransformerConfig
from transformer_from_scratch.components.mlp import MLP
from transformer_from_scratch.types import BatchResidualStream


class Layer(nn.Module):
    """Full layer (attention + feed forward).

    The layer receives the residual stream as an input. It then applies the multi-head attention
    sub-layer and adds the output back on to the residual stream. The updated residual stream is
    then passed through the feed forward sub-layer and the output is added back once more on to the
    residual stream (which is then the layer output). In this way, the residual stream acts like
    shared memory that the sub-layers interact with.

    https://arxiv.org/pdf/1706.03762.pdf (p3)
    """

    def __init__(self, config: TransformerConfig):
        """Initialise the full layer."""
        super().__init__()

        # Create the feed forward and attention sub-layers
        self.feed_forward = MLP(config)
        self.layer_norm_ff = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.layer_norm_attn = nn.LayerNorm(config.d_model)

    def forward(self, residual_stream: BatchResidualStream) -> BatchResidualStream:
        """Forward pass.

        Args:
            residual_stream (ResidualStream): Residual stream

        Returns:
            ResidualStream: Updated residual stream
        """
        # Attention
        attn = self.attention(residual_stream)
        attn_add_norm = residual_stream + self.layer_norm_ff(attn)

        # Feed forward
        mlp_output = self.feed_forward(attn_add_norm)
        return attn_add_norm + self.layer_norm_attn(mlp_output)
