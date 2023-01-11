from torch import nn, optim, save
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from tqdm import tqdm

from pathlib import Path
from alan_transformer.transformer import Transformer, TokenizedType


def train_loop(
    model: Transformer,
    dataloader: DataLoader, 
    epochs: int = 1, 
    save_frequency_batches: int = 10,
    checkpoint_dir: Path = Path(".checkpoints")
    ) -> None:
    """Train loop

    Args:
        model (Transformer): Transformer Model
        dataloader (DataLoader): Data Loader
        epochs (int, optional): Number of Epochs. Defaults to 1.
        save_frequency_batches (int, optional): Save model parameters every x batches. Defaults to 10.
        checkpoint_dir (Path, optional): Checkpoint directory to save parameters
            to. Defaults to Path(".checkpoints").
    """
    # Loss and optimizer settings loosely from
    # https://arxiv.org/pdf/1706.03762.pdf (p8)
    # Note that a residual dropout of 0.1 was also used in the paper (which has
    # not been done here)
    loss_fn = nn.CrossEntropyLoss()
    
    # Note that the paper also uses a warmup period of 4000 steps (which has not
    # been done here)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    
    # Create the checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Loop over epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        
        # Loop over batches
        for batch, (inputs, targets) in tqdm(enumerate(dataloader), desc="Batches"):
            # Forward pass
            logits: TokenizedType = model(inputs)
            loss = loss_fn(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Print
            # if batch % 100 == 0:
            #     print(f"Batch {batch} loss: {loss.item():.4f}")
            
            # Save model parameters
            if (batch + 1) % save_frequency_batches == 0:
                save(model.state_dict(), checkpoint_dir / f"model_{epoch}_{batch}.pt")
                
