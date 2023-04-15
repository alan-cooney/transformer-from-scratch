# Transformer From Scratch

A transformer built from scratch in PyTorch, using Test Driven Development (TDD) & modern
development best-practices. 

It is intended to be used as reference for curricula such as Jacob
Hilton's [Deep Leaning
Curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md).
In particular, because each module (e.g. the positional encoding) is individually tested, it's easy
to build all the discrete parts of a transformer and quickly understand what is broken (rather than
trying to debug a large model).

Only basic PyTorch linear algebra functions are used (e.g. tensor multiplication), with no use of
higher-level modules such as `Linear`.

## Setup

This project comes with a [DevContainer](https://containers.dev/) for one-click setup (e.g. with
[GitHub Codespaces](https://github.com/features/codespaces)). The quickest way to get started is to
use this DevContainer, which will install all dependencies. Alternatively you can clone the repo and
run `poetry install` yourself.

## Architecture

A decoder-only architecture is used (i.e. similar to GPT-2). Apart from this however, the
implementation is based off the original [Attention is All You
Need](https://arxiv.org/abs/1706.03762) paper. Terminology is consistent with [A Mathematical
Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html),
e.g. the code explicitly refers to the "residual stream". The resulting transformer model looks like
this (but with 8 layers):

![Transformer architecture from A Mathematical Framework for Transformer
Circuits](docs/images/transformer-architecture.png "Reference: A Mathematical Framework for Transformer
Circuits")

## Testing Strategy

The transformer is split into modules (e.g. `Encoder`). Each module is then tested to verify that it
can learn to do what we expect.

For example, we know from [A Mathematical
Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) that
an `Encoder` and `Decoder` (with separate weights) tend to learn **bigram statistics** - the probability
of the next token given just the current token (e.g. Barak -> Obama). We therefore verify that these
two modules can do this together. Similarly, the `MultiHeadAttention` module should be able to move
information between layers, so we directly verify this.

To run the tests, run `poetry run pytest`.

## Types

The project uses [Google Jaxtyping](https://github.com/google/jaxtyping) (it also works with
PyTorch), to type tensors (e.g. `BatchTokenIndicesTT = Int[Tensor, "BATCH POSITION"]`). The
underlying data type (`Int`/`Float`) is checked with [mypy](https://mypy.readthedocs.io/en/stable/),
and runtime type checking is enabled for all tests with pytest. Runtime type checking is not enabled
during training, as this would have a large performance impact.
