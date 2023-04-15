import pytest
import torch
import torch.optim as optim
import torch.nn.functional as F
from random import randint
from alan_transformer.train import cross_entropy_loss
from alan_transformer.transformer import Transformer
from jaxtyping import Float, Int
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from alan_transformer.types import BatchLogitsTT, BatchTokenIndicesTT, TokenIndicesTT
import random


class InductionHeadDataset(Dataset):
    """Induction Head Dataset.

    One thing that 2-layer (and above) transformers can do is to learn to repeat a pattern. This
    dataset therefore contains a sequence of tokens, where the first few tokens are a random pattern
    and then the rest of the sequence is the pattern repeated (potentially more than one time).
    """

    samples: list[TokenIndicesTT]

    def __init__(self, num_samples: int, sequence_length: int, d_vocab: int):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.d_vocab = d_vocab

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _index) -> TokenIndicesTT:
        initial_pattern_length = random.randint(2, self.sequence_length - 2)

        sample: list[int] = []
        for token_idx in range(self.sequence_length):
            # Create the initial pattern first
            if token_idx < initial_pattern_length:
                token = random.randint(1, self.d_vocab - 1)
                sample.append(token)

            # If we've reached the end of the initial pattern, then repeat it until the end of
            # the sequence.
            else:
                token = sample[token_idx % initial_pattern_length]
                sample.append(token)

        sample_tensor: TokenIndicesTT = torch.tensor(sample, dtype=torch.long)
        return sample_tensor


def test_transformer_induction_heads():
    d_vocab = 50000
    num_samples = 10000
    sequence_length = 20
    epochs = 1
    model = Transformer(d_vocab=d_vocab, n_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = F.cross_entropy
    dataset = InductionHeadDataset(num_samples, sequence_length, d_vocab)
    train_size = int(len(dataset) * 0.95)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Train the model
    for _epoch in range(epochs):
        model.train()
        for i, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            logits: BatchLogitsTT = model(inputs)
            loss = cross_entropy_loss(inputs, logits)
            loss.backward()
            optimizer.step()

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs in test_loader:
            outputs: BatchLogitsTT = model(inputs)

            penultimate_token_logits = outputs[:, -2, :]
            penultimate_tokens = torch.argmax(penultimate_token_logits, dim=-1)
            correct_last_tokens = inputs[:, -1]

            for token_idx, correct_last_token in enumerate(correct_last_tokens):
                predicted = penultimate_tokens[token_idx]
                if correct_last_token == predicted:
                    correct += 1
                total += 1

    accuracy = correct / total
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy}"
