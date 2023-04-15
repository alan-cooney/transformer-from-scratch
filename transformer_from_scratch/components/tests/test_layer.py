"""Test the full layer (attention + feed forward).

As we've tested the ability of the underling sub-layers to learn, in these tests we'll just check
that they have been combined correctly."""
import torch
from torch import nn

from transformer_from_scratch.components.attention import MultiHeadAttention
from transformer_from_scratch.components.feed_forward import FeedForward
from transformer_from_scratch.components.layer import Layer
from transformer_from_scratch.types import BatchResidualStreamTT


def test_layer_adds_attention_and_feed_forward_output(mocker):
    """Test if Layer correctly adds attention and feed forward outputs to the residual stream."""
    # Set up the inputs
    batch_size = 2
    seq_len = 3
    d_model = 4
    d_head = 2
    d_hidden = 8
    max_tokens = 1024

    residual_stream: BatchResidualStreamTT = torch.ones(
        batch_size, seq_len, d_model
    ).float()

    # Mock MultiHeadAttention and FeedForward to return constant values (that we can check are added
    # to the residual stream)
    mocker.patch.object(
        MultiHeadAttention,
        "forward",
        return_value=torch.full_like(residual_stream, 0.1),
    )
    mocker.patch.object(
        FeedForward, "forward", return_value=torch.full_like(residual_stream, 0.2)
    )

    # Mock layer norm to not normalize
    mocker.patch.object(nn.LayerNorm, "forward", side_effect=lambda x: x)

    # Instantiate Layer and perform forward pass
    layer = Layer(d_model, d_head, d_hidden, max_tokens)
    output = layer(residual_stream)

    # Check if the output is correct
    expected_output = torch.full_like(residual_stream, 1.3)
    assert torch.allclose(output, expected_output)
