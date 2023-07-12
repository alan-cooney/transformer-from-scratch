"""Training Utilities."""
from pathlib import Path
from typing import Optional
import torch
from jaxtyping import Int
from torch import Tensor, optim
from torch import save as torch_save
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from importlib.util import find_spec
import wandb
from transformer_from_scratch.components.cross_entropy_loss import cross_entropy_loss
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.types import BatchLogitsTT, BatchTokenIndicesTT
from transformer_from_scratch.types import TensorShapeLabels as D

TargetIndicesTT = Int[Tensor, f" {D.POSITION_MINUS_1}"]
BatchTargetIndicesTT = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1}"]

try:
    import torch_xla

    torch_xla_available = True
except ImportError:
    torch_xla_available = False


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
            inputs: BatchTokenIndicesTT = batch.to(device)
            inputs_except_last: BatchTargetIndicesTT = inputs[:, :-1]
            labels: BatchTargetIndicesTT = inputs[:, 1:]
            outputs = model(inputs_except_last)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    return correct / total


def train_loop(
    model: Transformer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 1,
    checkpoint_dir: Path = Path(".checkpoints"),
    device=get_default_device(),
    max_batches: Optional[int] = None,
    learning_rate: float = 0.00004,
    weight_decay: float = 0.1,
    warmup_steps: int = 250,
    gradient_clipping: float = 1.0,
) -> None:
    """Train loop

    Default training arguments taken from EleutherAI GPT 3 small

    https://github.com/EleutherAI/gpt-neo/blob/master/configs/gpt3_small_256.json

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

    # Setup the optimizer (using GPT-1 defaults as our model is similar)
    optimizer = optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.95),
        eps=1e-8,
        lr=learning_rate,  # HF tutorial uses 5e-4
        weight_decay=weight_decay,  # HF tutorial uses 0.1
    )

    # Create the checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_dir / "model_latest.pt"

    # Load model parameters if already check pointed (note we still start the learning schedule from scratch)
    # if latest_checkpoint.exists():
    #     model.load_state_dict(torch.load(latest_checkpoint))

    # Loop over epochs
    with tqdm(
        range(epochs),
        desc="Epochs",
        total=epochs,
        position=0,
        postfix={"test_accuracy": 0},
    ) as training_epochs:
        for epoch in training_epochs:
            # Setup the learning rate scheduler
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_steps,
                max_epochs=len(train_dataloader) * epochs,
            )

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
                    inputs: BatchTokenIndicesTT = torch.stack(
                        batch["input_ids"], -1
                    ).to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    logits: BatchLogitsTT = model(inputs)
                    loss = cross_entropy_loss(inputs, logits)

                    # Backward pass
                    loss.backward()
                    clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
                    scheduler.step()

                    # Log
                    tracked_batches.set_postfix(
                        {
                            "loss": loss.item()
                            # , "lr": scheduler._get_lr(step)
                        }
                    )
                    if step % 10 == 0 and wandb.run is not None:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "loss": loss.item(),
                                # "lr": scheduler._get_lr(step),
                            },
                            step,
                        )

                    # Every x steps, save a checkpoint
                    if step % 1000 == 0 and step > 0:
                        torch_save(
                            model.state_dict(), checkpoint_dir / f"model_{step}.pt"
                        )
                        torch_save(model.state_dict(), latest_checkpoint)

                # Evaluate each epoch
                model.eval()
                test_accuracy = evaluate(model, test_dataloader, device)

                # Log
                training_epochs.set_postfix(
                    {
                        "test_accuracy": test_accuracy,
                    }
                )
                if wandb.run is not None:
                    wandb.log(
                        {"epoch": epoch, "test_accuracy": test_accuracy},
                        len(train_dataloader) * epoch,
                    )

                model.train()

            # Save final model
            torch_save(model.state_dict(), latest_checkpoint)
