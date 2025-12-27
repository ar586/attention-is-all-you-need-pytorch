import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int):
    """
    seq: (batch_size, seq_len)
    returns: (batch_size, 1, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int):
    """
    returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
