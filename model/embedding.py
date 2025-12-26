import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Converts token indices into dense embedding vectors.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)
