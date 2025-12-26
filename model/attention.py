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
    


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits embedding into multiple heads.
        (B, T, d_model) -> (B, H, T, d_k)
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines multiple heads.
        (B, H, T, d_k) -> (B, T, d_model)
        """
        B, H, T, d_k = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(B, T, H * d_k)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
        """

        # Linear projections
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Apply attention
        out, _ = self.attention(Q, K, V, mask)

        # Combine heads
        out = self.combine_heads(out)

        # Final projection
        return self.W_o(out)

