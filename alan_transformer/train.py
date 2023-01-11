from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim, save
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from tqdm import tqdm

from alan_transformer.transformer import TokenizedType, Transformer


def train_loop(
    model: Transformer,
    dataloader: DataLoader, 
    epochs: int = 1, 
    save_frequency_batches: int = 10,
    checkpoint_dir: Path = Path(".checkpoints"),
    d_vocab: int = 50432
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
        for batch_index, batch in tqdm(enumerate(dataloader), desc="Batches"):
            # One-hot-encode the inputs
            inputs_tokenized: TT["batch", "pos"] = torch.stack(batch["input_ids"])
            inputs: TT["batch", "pos", "vocab"] = F.one_hot(inputs_tokenized, num_classes=d_vocab).float()
            inputs_excluding_last_pos = inputs[:, :-1, :]
            targets: TT["batch", "pos", "vocab"] = inputs[:, 1:, :]
            
            # Forward pass
            logits: TokenizedType = model(inputs_excluding_last_pos)
            loss = loss_fn(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Print
            # if batch_index % 1 == 0:
            #     print(loss)
            #     print(f"Batch {batch_index} loss: {loss.item():.4f}")
            
            # Save model parameters
            if (batch_index + 1) % save_frequency_batches == 0:
                save(model.state_dict(), checkpoint_dir / f"model_{epoch}_{batch_index}.pt")
                
