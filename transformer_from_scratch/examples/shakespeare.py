"""Shakespeare example."""
import re
import urllib
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, random_split
from transformers import GPTNeoXTokenizerFast

import wandb
from transformer_from_scratch.train import train_loop
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.types import BatchTokenIndicesTT


def create_tokenizer() -> GPTNeoXTokenizerFast:
    """Create the tokenizer."""
    return GPTNeoXTokenizerFast.from_pretrained("gpt2", pad_token="<|endoftext|>")


class TokenDataset(Dataset):
    def __init__(self, token_indices):
        self.token_indices = token_indices

    def __len__(self):
        return len(self.token_indices)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.token_indices[idx])
        return tokens


def create_dataset(
    data_dir: Path = Path(__file__).parent / ".data",
    max_length: int = 1024,
) -> Dataset:
    """Create the Shakespeare Dataset (one prompt per line)

    Args:
        data_dir: Directory to store the data
        load_if_exists: Flag to load the data from disk if it already exists
    """

    # Download text file
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "shakespeare.txt"
    data_url = "https://www.gutenberg.org/files/100/100-0.txt"
    if not data_path.exists():
        urllib.request.urlretrieve(data_url, data_path)

    # Load text and tokenize
    tokenizer = create_tokenizer()
    with open(data_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Replace multiple newline characters with a single newline character
    text = re.sub(r"\n+", "\n", text)

    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)

    tokens = tokenizer.encode(text, truncation=False)

    # Split into chunks of max_length
    examples = []
    for i in range(0, len(tokens), max_length):
        prompt_tokens = tokens[i : i + max_length]

        # Exclude if it's the last prompt and it is too short
        if len(prompt_tokens) < max_length:
            break

        examples.append(prompt_tokens)

    dataset = TokenDataset(examples)
    return dataset


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Convert a dataset into a dataloader

    Args:
        dataset: Dataset
        batch_size: Batch size
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=0 # Adding more workers doesn't appear to impact performance
    )


def train_shakespeare(batch_size: int = 12) -> None:
    """Train a transformer on the Shakespeare dataset

    Args:
        batch_size (int, optional): Batch size.
    """
    max_length_prompt = 256
    dataset = create_dataset(max_length=max_length_prompt)
    train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = create_dataloader(train_dataset, batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size)

    model = Transformer(
        d_model=256, d_head=64, d_hidden=1024, n_layers=2, max_tokens=max_length_prompt
    )

    train_loop(model, train_dataloader, test_dataloader, epochs=1)


if __name__ == "__main__":
    # wandb.login()
    # wandb.init(project="transformer-from-scratch")
    train_shakespeare()
