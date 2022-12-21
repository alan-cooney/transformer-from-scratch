import torch
from torchtyping import TensorType as TT


EmbeddingType = TT["batch", "pos", "d_model"]
PositionalEncodingType = TT["max_tokens", "d_model"]


class PositionalEncoding(torch.nn.Module):
    """Positional encoding"""

    def __init__(self, d_model: int, max_tokens: int) -> None:
        """Applies positional encoding to embedding

        PE(pos,2i) = sin(pos/(10000^(2i/d_model)))
        PE(pos,2i+1) = cos(pos/(10000^(2i/d_model)))

        https://arxiv.org/pdf/1706.03762.pdf (p6)
        """
        # Create the positional encoding
        # Create everything inside the parentheses
        # (pos/(10000^(2i/d_model))
        positions: TT["pos", 1] = torch.arange(0, max_tokens).unsqueeze(1)
        dimensions_2: TT["d_model_half"] = torch.arange(0, d_model, 2)
        inner: TT["pos", "d_model_half"] = positions / \
            (10000 ** (dimensions_2 / d_model))

        # Create interweaved positional encoding
        pos_encoding: TT["pos", "d_model"] = torch.zeros(max_tokens, d_model)
        pos_encoding[:, 0::2] = torch.sin(inner)
        pos_encoding[:, 1::2] = torch.cos(inner)
        self.pos_encoding: PositionalEncodingType = pos_encoding

        super().__init__()

    def forward(self, embedding: EmbeddingType) -> EmbeddingType:
        embedding_n_tokens: int = embedding.shape[-2]
        trimmed_pos_encoding: TT["pos",
                                 "d_model"] = self.pos_encoding[:embedding_n_tokens, :]
        return trimmed_pos_encoding + embedding
