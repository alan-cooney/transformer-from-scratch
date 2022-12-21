import torch
from fancy_einsum import einsum
from torchtyping import TensorType as TT
from einops import rearrange


EmbeddingType = TT["batch", "pos", "d_model"]

def positional_encoding(embedding: EmbeddingType)-> EmbeddingType:
    """Applies positional encoding to embedding
    
    PE(pos,2i) = sin(pos/(10000^(2i/d_model)))
    PE(pos,2i+1) = cos(pos/(10000^(2i/d_model)))
    
    https://arxiv.org/pdf/1706.03762.pdf (p6)
    """
    # Get the dimensions
    d_model: int = embedding.shape[-1]
    d_pos: int = embedding.shape[-2]
    
    # Create everything inside the parentheses
    # (pos/(10000^(2i/d_model))
    positions: TT["pos", 1] = torch.arange(0, d_pos).unsqueeze(1)
    dimensions_2: TT["d_model_half"] = torch.arange(0, d_model, 2)
    inner: TT["pos", "d_model_half"] = positions / (10000 ** (dimensions_2 / d_model))
    print(positions.shape, dimensions_2.shape, inner.shape, embedding.shape)
    
    # Create interweaved positional encoding
    pos_encoding: EmbeddingType = torch.zeros_like(embedding)
    pos_encoding[:, :, 0::2] = torch.sin(inner)
    pos_encoding[:, :, 1::2] = torch.cos(inner)
    
    # Add to the embedding
    return pos_encoding + embedding
    
    