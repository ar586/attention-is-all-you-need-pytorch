import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """
    Single Transformer encoder block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional attention mask

        Returns:
            Tensor of same shape
        """

        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
class Encoder(nn.Module):
    """
    Transformer Encoder consisting of stacked encoder blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        from model.embedding import TokenEmbedding
        from model.positional_encoding import PositionalEncoding

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (batch_size, seq_len)
            mask: optional attention mask

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """

        # Token + positional embedding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through stacked encoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        return x
