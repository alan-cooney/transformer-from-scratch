from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import math
import urllib
from typing import Tuple
from pathlib import Path
from transformers import GPTNeoXTokenizerFast
from datasets import load_dataset, Dataset, load_from_disk

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from alan_transformer.transformer import Transformer
from alan_transformer.train import train_loop

def create_dataset(data_dir = Path(".data"), load_if_exists: bool = True) -> Dataset:
    """_summary_

    Args:
        data_dir (_type_, optional): _description_. Defaults to Path(".data").
        load_if_exists (bool, optional): _description_. Defaults to True.

    Returns:
        Dataset: _description_
    """
    # Return the dataset from disk if it already exists
    dataset_path = data_dir / "shakespeare_dataset"
    if dataset_path.exists() and load_if_exists:
        return load_from_disk(dataset_path)
    
    # Download text file
    data_dir = Path(__file__).parent.parent / ".data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "shakespeare.txt"
    data_url = "https://www.gutenberg.org/files/100/100-0.txt"
    urllib.request.urlretrieve(data_url, data_path)

    # Load as a dataset
    raw_dataset = load_dataset("text", data_files=str(data_path))

    # Tokenize it
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b", pad_token = "<|endoftext|>")
    tokenized_dataset = raw_dataset.map(
            lambda examples: tokenizer(
                examples["text"], 
                padding="max_length", # Pad to the max length
                truncation=True, # Truncate to the max length
                max_length=1024, # 1024 is the default max length for our transformer,
                is_split_into_words=False,
                return_tensors="pt" # Return a pytorch tensor per prompt
            )
        )

    # Save the dataset
    tokenized_dataset.save_to_disk(dataset_path)
    
    # Create the dataloader
    dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    
    return dataloader

