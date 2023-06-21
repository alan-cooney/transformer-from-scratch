"""Training Utilities."""
from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Int
from torch import Tensor, optim
from torch import save as torch_save
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from transformer_from_scratch.components.cross_entropy_loss import cross_entropy_loss
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.types import BatchLogitsTT, BatchTokenIndicesTT
from transformer_from_scratch.types import TensorShapeLabels as D

TargetIndicesTT = Int[Tensor, f" {D.POSITION_MINUS_1}"]
BatchTargetIndicesTT = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1}"]


def get_default_device() -> torch.device:
    """Get the default device to use.

    Returns:
        torch.device: Device to use.
    """
    if torch.backends.mps.is_built():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate(
    model: Module,
    test_dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate the model on a test dataloader

    Args:
        model (Transformer): Transformer model
        test_dataloader (DataLoader): Test dataloader
        device (torch.device): Pytorch device

    Returns:
        float: Accuracy (portion of tokens that are correctly predicted)
    """
    total, correct = 0, 0

    with torch.no_grad():
        for batch in test_dataloader:
            inputs: BatchTokenIndicesTT = batch["input_ids"].to(device)
            inputs_except_last: BatchTargetIndicesTT = inputs[:, :-1]
            labels: BatchTargetIndicesTT = inputs[:, 1:]
            outputs = model(inputs_except_last)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    return correct / total


def learning_rate_scheduler(step: int, d_model: int) -> float:
    """Learning rate scheduler

    Args:
        step (int): Step.
        d_model (int): Number of residual stream features per token.

    Returns:
        float: Learning rate
    """
    warmup_steps = 4000
    return d_model * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5))


def train_loop(
    model: Transformer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 1,
    checkpoint_dir: Path = Path(".checkpoints"),
    device=get_default_device(),
    max_batches: Optional[int] = None,
) -> None:
    """Train loop

    Args:
        model: The Transformer model to train.
        train_dataloader: Dataloader for training data.
        test_dataloader: Dataloader for test data.
        epochs (int): Number of epochs to train for. Defaults to 1.
        checkpoint_dir (Path): Directory to save model parameters. Defaults to ".checkpoints".
        device (torch.device): Device to use for training. Defaults to GPU if available, else CPU.
        max_batches (Optional[int]): Maximum number of batches to process. Defaults to None, which
            processes all batches in the dataloader.
    """
    # Initialise training
    model.to(device)

    # Note that the paper also uses a warmup period of 4000 steps (which has not
    # been done here)
    # , betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    d_model = model.d_model  # Residual stream dimensions
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_scheduler(step, d_model),
    )

    # Create the checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_dir / "model_latest.pt"

    # Load model parameters if already check pointed
    if latest_checkpoint.exists():
        model.load_state_dict(torch.load(latest_checkpoint))

    # Loop over epochs
    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        # Set to training mode
        model.train()

        # Loop over batches
        with tqdm(
            enumerate(train_dataloader),
            desc="Steps",
            total=len(train_dataloader),
            position=1,
        ) as tracked_batches:
            for step, batch in tracked_batches:
                # Check not over max_batches
                if max_batches and step >= max_batches:
                    break

                # Move inputs to the device
                inputs: BatchTokenIndicesTT = batch["input_ids"].to(device)

                # Forward pass
                optimizer.zero_grad()
                logits: BatchLogitsTT = model(inputs)
                loss = cross_entropy_loss(inputs, logits)

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Add loss & lr to tqdm console output
                if step % 10 == 0:
                    tracked_batches.set_postfix(loss=loss.item())
                    tracked_batches.set_postfix(lr=scheduler.get_last_lr()[0])

                # Log to Wandb
                if step % 10 == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step,
                    )

                # Every 500 steps, evaluate & log this (accuracy)
                # if step % 500 == 0:
                #     model.eval()
                #     test_accuracy = evaluate(model, test_dataloader, device)
                #     if wandb.run is not None:
                #         wandb.log(
                #             {"epoch": epoch, "test_accuracy": test_accuracy},
                #             step
                #         )
                #     model.train()

                # Every x steps, save a checkpoint
                if step % 5000 == 0 and step > 0:
                    torch_save(model.state_dict(), checkpoint_dir / f"model_{step}.pt")
                    # torch_save(model.state_dict(), latest_checkpoint)
