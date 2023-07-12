"""Code example."""
import os

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Tokenizer

import wandb
from transformer_from_scratch.train import train_loop
from transformer_from_scratch.transformer import Transformer

CONTEXT_LENGTH = 128

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_datasets():
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid", split="validation"
    )

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle().select(range(50000)),
            "valid": ds_valid.shuffle().select(range(500)),
        }
    )
    return raw_datasets


def tokenize(element):
    """Tokenize code."""
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == CONTEXT_LENGTH:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def get_tokenized_datasets():
    """Get the tokenized datasets."""
    raw_datasets = get_datasets()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_names = {
        split: os.path.join(cache_dir, f"{split}_cache.arrow")
        for split in raw_datasets.keys()
    }
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        cache_file_names=cache_file_names,  # Cache results
    )
    return tokenized_datasets["train"], tokenized_datasets["valid"]


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


def train_code(batch_size: int = 256) -> None:
    """Train a transformer on the Code dataset
    Args:
        batch_size (int, optional): Batch size.
    """
    max_length_prompt = CONTEXT_LENGTH
    train_dataset, test_dataset = get_tokenized_datasets()
    train_dataloader = create_dataloader(train_dataset, batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size)

    model = Transformer(
        d_model=256, d_head=64, d_hidden=1024, n_layers=2, max_tokens=max_length_prompt
    )
    # config = AutoConfig.from_pretrained(
    #     "gpt2",
    #     vocab_size=len(tokenizer),
    #     n_ctx=CONTEXT_LENGTH,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    # model = GPT2LMHeadModel(config)

    train_loop(model, train_dataloader, test_dataloader, epochs=1)


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="transformer-from-scratch")
    train_code()
