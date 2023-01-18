from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim, save
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from tqdm import tqdm, trange

import wandb
from alan_transformer.transformer import Transformer
from alan_transformer.types import LogitsTT, TokensTT


def cross_entropy_loss(
    inputs: TokensTT,
    logits: LogitsTT,
) -> TT[()]:
    """Loss function

    Loss is calculated from the difference between log probs of the 

    https://arxiv.org/pdf/1706.03762.pdf (p8)

    Params:
        Input: Input tokens
        logits: Logits from the forward pass
s
    Returns:
        Log loss
    """
    # Targets are inputs except for the first one (which we aren't predicting)
    # Logits except last exclude the last one (which we don't have a target for)
    target: TT["batch", "pos_minus_1"] = inputs[:, 1:]
    logits_except_last: TT["batch", "pos_minus_1", "d_vocab"] = \
        logits[:, :-1, :].float()

    log_probs: TT["batch", "pos_minus_1", "d_vocab"] = \
        F.log_softmax(logits_except_last, dim=-1)

    # Predicted log probs are the log probs of the correct tokens
    index: TT["batch", "pos_mins_1", 1] = target.unsqueeze(-1)
    predicted_log_probs = log_probs.gather(-1, index)

    # Cross entropy loss
    return -predicted_log_probs.mean()


def train_loop(
    model: Transformer,
    dataloader: DataLoader,
    epochs: int = 1,
    save_frequency_batches: int = 10,
    checkpoint_dir: Path = Path(".checkpoints"),
    d_vocab: int = 50432,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_batches: Optional[int] = None
) -> None:
    """Train loop

    Args:
        model: Transformer Model
        dataloader: Data Loader
        epochs: Number of Epochs
        save_frequency_batches: Save model parameters every x batches
        checkpoint_dir: Checkpoint directory to save parameters to. Defaults to
            Path(".checkpoints")
        d_vocab: Vocab size
    """
    # Note that the paper also uses a warmup period of 4000 steps (which has not
    # been done here)
    # , betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    # Create the checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move model to the device
    model.to(device)

    # Loop over epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Loop over batches
        with tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)) as tracked_batches:
            for batch_index, batch in tracked_batches:

                # Check not over max_batches
                if max_batches and batch_index >= max_batches:
                    break

                # Move inputs to the device
                inputs: TT["batch", "pos",
                           "vocab"] = batch["input_ids"].to(device)

                # Forward pass
                optimizer.zero_grad()
                logits: LogitsTT = model(inputs)
                loss = cross_entropy_loss(inputs, logits)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Print
                if batch_index % 10 == 0:
                    loss_string = loss.item()
                    tracked_batches.set_postfix(loss=loss_string)

                # Log
                if batch_index % 10 == 0 and wandb.run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "batch": batch_index,
                        "loss": loss.item()
                    })

                # Save model parameters
                # if (batch_index + 1) % save_frequency_batches == 0:
                #     save(model.state_dict(), checkpoint_dir /
                #          f"model_{epoch}_{batch_index}.pt")
