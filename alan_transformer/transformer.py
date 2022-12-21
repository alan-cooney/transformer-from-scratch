from torch import nn
from alan_transformer.layer import Layer
from alan_transformer.embed_unembed import Embed, Unembed
from alan_transformer.positional_encoding import PositionalEncoding
from torchtyping import TensorType as TT


TokenizedType = TT["batch", "pos", "d_vocab"]
ResidualStreamType = TT["batch", "pos", "d_model"]


class Transformer(nn.Module):
    """Transformer"""

    def __init__(
        self,
        d_head: int = 64,
        d_hidden: int = 2048,
        d_model: int = 768,
        d_vocab: int = 50000,
        max_tokens: int = 1024,
        n_layers: int = 12,
    ) -> None:
        super().__init__()

        # Embedding and unembedding
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_tokens=max_tokens)

        # Layers
        self.layers = []
        for _layer_idx in range(n_layers):
            self.layers.append(
                Layer(d_model=d_model, d_head=d_head, d_hidden=d_hidden))

    def forward(self, tokens: TokenizedType) -> TokenizedType:
        # Embed + positional encoding
        x: ResidualStreamType = self.embed(tokens)
        x = self.positional_encoding(x)

        # Loop through layers
        for layer in self.layers:
            x = layer(x)

        # Unembed and return
        return self.unembed(x)
