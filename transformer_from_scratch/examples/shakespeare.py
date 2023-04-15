import urllib
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import GPTNeoXTokenizerFast

import wandb
from transformer_from_scratch.train import train_loop
from transformer_from_scratch.transformer import Transformer


def create_tokenizer() -> GPTNeoXTokenizerFast:
    return GPTNeoXTokenizerFast.from_pretrained("gpt2", pad_token="<|endoftext|>")


def tokenize_prompt(
    text: List[str],
    tokenizer: GPTNeoXTokenizerFast,
) -> Dict[str, Float[Tensor, "mapping_batch_item pos"]]:
    """Tokenize a prompt

    Designed to be used by a dataset mapping function, so it returns a dict with
    key "input_ids" and value of the tokenized text.

    Args:
        text: List of strings to tokenize (batched by the dataset mapping
        function)
        tokenizer: Tokenizer
    """
    tokenized = tokenizer(
        text,
        padding="max_length",  # Pad to the max length
        truncation=True,  # Truncate to the max length
        max_length=1024,  # 1024 is the default max length for our transformer,
        is_split_into_words=False,
        return_attention_mask=False,
        return_tensors="pt",  # Return a pytorch tensor per prompt
    )

    # Set return type as dict
    return {"input_ids": tokenized["input_ids"]}


def create_dataset(
    data_dir: Path = Path(__file__).parent / ".data",
    load_if_exists: bool = True,
) -> DatasetDict:
    """Create the Shakespeare Dataset (one prompt per line)

    Args:
        data_dir: Directory to store the data
        load_if_exists: Flag to load the data from disk if it already exists
    """
    # Return the dataset from disk if it already exists
    dataset_path = data_dir / "shakespeare_dataset"
    if dataset_path.exists() and load_if_exists:
        return load_from_disk(dataset_path)

    # Download text file
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "shakespeare.txt"
    data_url = "https://www.gutenberg.org/files/100/100-0.txt"
    urllib.request.urlretrieve(data_url, data_path)

    # Load as a dataset
    dataset = load_dataset("text", data_files=str(data_path))

    # Tokenize it
    tokenizer = create_tokenizer()
    dataset = dataset.map(
        lambda examples: tokenize_prompt(examples["text"], tokenizer),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids"])

    # Save the dataset
    dataset.save_to_disk(dataset_path)
    return dataset


def create_dataloader(dataset: DatasetDict, batch_size: int) -> DataLoader:
    """Convert a dataset into a dataloader

    Args:
        dataset: Dataset
        batch_size: Batch size
    """
    return DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)


def train_shakespeare(batch_size: int = 4) -> None:
    dataset = create_dataset()
    dataloader = create_dataloader(dataset, batch_size)

    model = Transformer()

    train_loop(
        model,
        dataloader,
    )


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="transformer-from-scratch")
    train_shakespeare()
