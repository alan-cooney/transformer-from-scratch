"""Transformer Tests.

Unlike the underlying components, full integration testing of the Transformer as a whole can be time
intensive. As such we focus on testing the underlying architecture here."""
import pytest
import torch

from transformer_from_scratch.transformer import Transformer

@pytest.mark.parametrize(
    "d_head, d_hidden, d_model, d_vocab, max_tokens, n_layers",
    [
        (64, 2048, 768, 50432, 1024, 12),
        (32, 1024, 512, 25000, 512, 6),
        (16, 512, 256, 10000, 256, 4),
    ],
)
def test_transformer_init_correctly(
    d_head, d_hidden, d_model, d_vocab, max_tokens, n_layers
):
    """Check that the Transformer is initialised correctly."""
    transformer = Transformer(
        d_head=d_head,
        d_hidden=d_hidden,
        d_model=d_model,
        d_vocab=d_vocab,
        max_tokens=max_tokens,
        n_layers=n_layers,
    )

    weights = transformer.state_dict()

    # Check the weights shapes, as a way of checking the underlying components are correctly
    # initialised
    assert weights["embed.embed_weights"].shape == (d_vocab, d_model)
    assert weights["unembed.unembed_weights"].shape == (d_model, d_vocab)
    assert weights["layers.0.attention.weight_query"].shape == (
        d_model // d_head,
        d_model,
        d_head,
    )
    assert weights["layers.0.feed_forward.weight_inner"].shape == (d_model, d_hidden)


@pytest.mark.parametrize(
    "d_head, d_hidden, d_model, d_vocab, max_tokens, n_layers",
    [
        (64, 2048, 768, 50432, 1024, 12),
        (32, 1024, 512, 25000, 512, 6),
        (16, 512, 256, 10000, 256, 4),
    ],
)
def test_transformer_forward(d_head, d_hidden, d_model, d_vocab, max_tokens, n_layers):
    """Check that a forward pass can be run."""
    # Create an instance of the Transformer with the specified parameters
    transformer = Transformer(
        d_head=d_head,
        d_hidden=d_hidden,
        d_model=d_model,
        d_vocab=d_vocab,
        max_tokens=max_tokens,
        n_layers=n_layers,
    )

    # Create a sample input tensor of shape (batch_size, seq_len)
    batch_size, seq_len = 2, 50
    tokens = torch.randint(low=0, high=d_vocab, size=(batch_size, seq_len))

    # Test the forward method
    logits = transformer(tokens)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_len, d_vocab)
