import math

import torch
from jaxtyping import Float
from torch import Tensor

from alan_transformer.embed_unembed import Embed, Unembed
from alan_transformer.tests.utils.mock_parameter import MockParameterOnes

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import random


# class TestEmbed:
#     def test_embed(self, mocker):
#         # Create the layer
#         d_vocab = 10
#         d_model = 5
#         layer = Embed(d_vocab, d_model)

#         # Set the dummy parameters so that each token is embedded as a list of
#         # that number (e.g. token 3 -> [3 for _ in d_model])
#         state = layer.state_dict()
#         new_embed_weights: Float[Tensor, "vocab d_model"] = (
#             torch.arange(0, d_vocab).repeat(d_model, 1).T
#         )
#         state["embed_weights"] = new_embed_weights
#         # Set the bias as 0 for simplicity
#         state["embed_bias"] = torch.zeros(d_model)
#         layer.load_state_dict(state)

#         mock_tokens = torch.tensor([3, 1, 0])
#         expected = mock_tokens.repeat(d_model, 1).T * math.sqrt(d_model)
#         res = layer(mock_tokens.unsqueeze(0))

#         assert torch.allclose(res, expected.unsqueeze(0))


# class TestUnembed:
#     def test_unembed(self, mocker):
#         # Mock the weight initialisation (use ones instead)
#         mocker.patch("torch.nn.Parameter", new=MockParameterOnes)

#         # Create the mock tokens in the residual stream
#         # Divide by d_model so that after the multiplication with the weights, we
#         # get ones
#         d_vocab = 10
#         d_model = 5
#         n_tokens = 5
#         tokens = torch.ones((1, n_tokens, d_model)) / d_model

#         res = Unembed(d_vocab, d_model)(tokens)
#         # Expected has plus one for the bias
#         expected = torch.ones((1, n_tokens, d_vocab)) + 1

#         assert torch.allclose(res, expected)


class OrderedIntegersDataset(Dataset):
    """Ordered integers dataset.

    The samples are contiguous sets of numbers (e.g. {3,1,2,4,0}). The targets are simply the ordered
    version of these sets (e.g. {0,1,2,3,4}).
    """

    def __init__(self, num_samples: int, d_vocab: int):
        """Initialise the dataset.

        Args:
            num_samples (int): Number of samples
            d_vocab (int): Vocab size (e.g. if 3 then it'll generate randomly ordered sets of
            {0,1,2}).
        """
        self.samples = []
        self.targets = []
        for _ in range(num_samples):
            sample = list(range(0, d_vocab))
            random.shuffle(sample)
            target = sorted(sample)
            self.samples.append(sample)
            self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.targets[idx])


class ZeroLayerModel(nn.Module):
    """Zero Layer Model.

    Contains just embedding and unembedding layers.
    """

    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.unembed = Unembed(d_vocab, d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.unembed(x)
        return x


class TestBigrams:
    """Test that a model with just the embedding and unembedding can learn bigram statistics.

    Bigram statistics are the frequencies with which one token comes after another. In this case,
    we're trying to train a model to order a randomly ordered set of integers (i.e. to learn that 1
    -> 2, 2 -> 3 and so on)

    The ability of a zero-layer model (i.e. one with just embedding and unembedding layers) to do
    this is proven in A Mathematical Framework for Transformer Circuits.

    Reference: https://transformer-circuits.pub/2021/framework/index.html
    """

    def test_learn_order_integers(self):
        # Set random seeds
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples = 64
        epochs = 1
        d_vocab = 10
        d_model = d_vocab
        model = ZeroLayerModel(d_vocab, d_model).to(device)

        dataset = OrderedIntegersDataset(samples, d_vocab)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Train the model
        for epoch in range(epochs):
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                print(inputs.shape)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, d_vocab), targets.view(-1))
                loss.backward()
                optimizer.step()

            print(loss)

        # Test the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 2)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        # assert accuracy > 0.9, f"Expected accuracy > 0.9, but got {accuracy}"
