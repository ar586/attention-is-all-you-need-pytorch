import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import Transformer
from training.dataset import ToyTextDataset
from training.utils import create_causal_mask


# ---- text generation helper ----
def generate_text(model, dataset, start_text, max_len=30):
    model.eval()

    input_ids = [dataset.stoi[ch] for ch in start_text]
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    for _ in range(max_len):
        seq_len = input_ids.size(1)
        tgt_mask = create_causal_mask(seq_len)

        with torch.no_grad():
            logits = model(input_ids, input_ids, tgt_mask=tgt_mask)

        next_token_logits = logits[0, -1]
        next_token = torch.argmax(next_token_logits).item()

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]])],
            dim=1
        )

    return "".join(dataset.itos[i] for i in input_ids[0].tolist())


# ---- training loop ----
def main():
    text = "hello transformer"
    seq_len = 5
    batch_size = 2
    epochs = 50

    dataset = ToyTextDataset(text, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset.vocab)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            tgt_mask = create_causal_mask(x.size(1))

            logits = model(x, x, tgt_mask=tgt_mask)
            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ---- generation call (INSIDE main) ----
    print("\n--- Text Generation ---")
    print(generate_text(model, dataset, start_text="hello ", max_len=20))


# ---- entry point ----
if __name__ == "__main__":
    main()
