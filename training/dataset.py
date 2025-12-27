import torch
from torch.utils.data import Dataset


class ToyTextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.data = [self.stoi[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)
