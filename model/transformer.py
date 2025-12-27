import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    """
    Full Transformer architecture.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Final linear layer maps to vocab
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)

        Returns:
            logits: (batch_size, tgt_seq_len, tgt_vocab_size)
        """

        # Encode source sequence
        enc_out = self.encoder(src, src_mask)

        # Decode target sequence
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_layer(dec_out)

        return logits
