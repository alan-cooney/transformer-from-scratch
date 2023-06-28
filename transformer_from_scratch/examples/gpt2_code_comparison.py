"""GPT 2 Code Comparison.

Trains a model with HuggingFace from scratch, to understand how this compares to the custom model.
Based on:

https://huggingface.co/learn/nlp-course/chapter7/6#initializing-a-new-model"""
import os

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

import wandb

CONTEXT_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


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
    return tokenized_datasets


def train():
    wandb.init(project="gpt2-code-comparison", sync_tensorboard=False)
    tokenized_datasets = get_tokenized_datasets()
    model = GPT2LMHeadModel(config)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=".checkpoints",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        # fp16=True,
        # push_to_hub=True,
        report_to="wandb",
        use_mps_device=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()


if __name__ == "__main__":
    train()
