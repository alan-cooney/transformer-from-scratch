"""Full layer (attention + feed forward)."""
from torch import nn

from transformer_from_scratch.components.attention import MultiHeadAttention
from transformer_from_scratch.components.feed_forward import FeedForward
from transformer_from_scratch.types import BatchResidualStreamTT


class Layer(nn.Module):
    """Full layer (attention + feed forward).

    The layer receives the residual stream as an input. It then applies the multi-head attention
    sub-layer and adds the output back on to the residual stream. The updated residual stream is
    then passed through the feed forward sub-layer and the output is added back once more on to the
    residual stream (which is then the layer output). In this way, the residual stream acts like
    shared memory that the sub-layers interact with.

    https://arxiv.org/pdf/1706.03762.pdf (p3)
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        d_hidden: int,
        max_tokens: int,
    ):
        """Initialise the full layer.

        Args:
            d_model (int): Number of residual stream features per token.
            d_head (int, optional): Number of head features per token.
            d_hidden (int, optional): Number of hidden layer features per token.
            max_tokens (int, optional): Maximum number of tokens in a sequence.
        """
        super().__init__()

        # Create the feed forward and attention sub-layers
        self.feed_forward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_head, d_model, max_tokens)
        self.layer_norm_attn = nn.LayerNorm(d_model)

    def forward(self, residual_stream: BatchResidualStreamTT) -> BatchResidualStreamTT:
        """Forward pass.

        Args:
            residual_stream (ResidualStreamTT): Residual stream

        Returns:
            ResidualStreamTT: Updated residual stream
        """
        # Attention
        attn = self.attention(residual_stream)
        attn_add_norm = residual_stream + self.layer_norm_ff(attn)

        # Feed forward
        mlp_output = self.feed_forward(attn_add_norm)
        return attn_add_norm + self.layer_norm_attn(mlp_output)
