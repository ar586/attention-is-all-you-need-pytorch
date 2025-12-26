import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            Q, K, V: (batch_size, heads, seq_len, d_k)
            mask: (batch_size, 1, 1, seq_len) or None

        Returns:
            output: (batch_size, heads, seq_len, d_k)
            attention_weights: (batch_size, heads, seq_len, seq_len)
        """

        d_k = Q.size(-1)

        # (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        # (B, H, T, d_k)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
