"""Transformer Tests.

Unlike the underlying components, full integration testing of the Transformer as a whole can be time
intensive. As such we focus on testing the underlying architecture here."""
import pytest
import torch
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.transformer import Transformer


@pytest.mark.parametrize(
    "d_head, d_mlp, d_model, d_vocab, n_ctx, n_layers",
    [
        (64, 2048, 768, 50432, 1024, 12),
        (32, 1024, 512, 25000, 512, 6),
        (16, 512, 256, 10000, 256, 4),
    ],
)
def test_transformer_init_correctly(d_head, d_mlp, d_model, d_vocab, n_ctx, n_layers):
    """Check that the Transformer is initialised correctly."""
    config = TransformerConfig(
        d_head=d_head,
        d_mlp=d_mlp,
        d_model=d_model,
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        n_layers=n_layers,
        n_heads=int(d_head / d_model),
    )
    transformer = Transformer(config)

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
    assert weights["layers.0.feed_forward.weight_inner"].shape == (d_model, d_mlp)


@pytest.mark.parametrize(
    "d_head, d_mlp, d_model, d_vocab, n_ctx, n_layers",
    [
        (64, 2048, 768, 50432, 1024, 12),
        (32, 1024, 512, 25000, 512, 6),
        (16, 512, 256, 10000, 256, 4),
    ],
)
def test_transformer_forward(d_head, d_mlp, d_model, d_vocab, n_ctx, n_layers):
    """Check that a forward pass can be run."""
    # Create an instance of the Transformer with the specified parameters
    config = TransformerConfig(
        d_head=d_head,
        d_mlp=d_mlp,
        d_model=d_model,
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        n_layers=n_layers,
        n_heads=int(d_head / d_model),
    )
    transformer = Transformer(config)

    # Create a sample input tensor of shape (batch_size, seq_len)
    batch_size, seq_len = 2, 50
    tokens = torch.randint(low=0, high=d_vocab, size=(batch_size, seq_len))

    # Test the forward method
    logits = transformer(tokens)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_len, d_vocab)


def test_transformer_memorize_dataset():
    """Check the transformer can learn to memorize a simple dataset."""
    # Create a simple dataset
    d_vocab = 10
    dataset = torch.randint(low=0, high=d_vocab, size=(10, 10))

    # Create an instance of the Transformer
    config = TransformerConfig(
        d_head=4,
        d_mlp=32,
        d_model=16,
        d_vocab=d_vocab,
        n_ctx=10,
        n_layers=2,
        n_heads=4,
    )
    transformer = Transformer(config)

    # Create a simple optimizer
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

    # Train the Transformer
    for _ in range(1000):
        # Reset the gradients
        optimizer.zero_grad()

        # Run a forward pass
        logits = transformer(dataset)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(
            input=logits.reshape(-1, d_vocab),
            target=dataset.reshape(-1),
        )

        # Compute the gradients
        loss.backward()

        # Update the weights
        optimizer.step()

    # Check that the loss is low
    assert loss < 1e-2
