import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    """
    Single Transformer decoder block.
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
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (batch_size, tgt_seq_len, d_model)
            enc_out: (batch_size, src_seq_len, d_model)
            tgt_mask: causal mask for decoder self-attention
            src_mask: padding mask for encoder-decoder attention

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model)
        """

        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # Encoder-decoder attention
        attn2 = self.enc_dec_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x

class Decoder(nn.Module):
    """
    Transformer Decoder consisting of stacked decoder blocks.
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
                DecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (batch_size, tgt_seq_len)
            enc_out: (batch_size, src_seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model)
        """

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)

        return x
