"""Shakespeare example tests"""
import torch
from transformer_from_scratch.examples.shakespeare import (
    create_dataloader,
    create_dataset,
    create_tokenizer,
)


def test_create_dataset_first_item():
    """Check the first item of the created dataset."""
    dataset = create_dataset()

    # Get the first prompt
    first_prompt_tokens = dataset[0]["input_ids"]
    tokenizer = create_tokenizer()
    first_prompt = tokenizer.decode(first_prompt_tokens)
    end_of_sequence_token = "<|endoftext|>"

    expected = (
        "\ufeffThe Project Gutenberg eBook of The Complete Works of William Shakespeare"
        + ", by William Shakespeare"
    )

    assert first_prompt.replace(end_of_sequence_token, "") == expected


def test_create_dataloader_random_sorting():
    """Check the dataloader is randomly sorted."""
    dataset = create_dataset()

    # Create two different dataloaders
    torch.manual_seed(1)
    dataloader1 = create_dataloader(dataset, 4)
    torch.manual_seed(2)
    dataloader2 = create_dataloader(dataset, 4)

    # Get the first prompt from each dataloader
    first_prompt1 = next(iter(dataloader1))["input_ids"][0]
    first_prompt2 = next(iter(dataloader2))["input_ids"][0]

    # The two prompts should not be the same if shuffling is working correctly
    assert (first_prompt1 != first_prompt2).any()
