# Attention Is All You Need — From Scratch (PyTorch)

This repository showcases a **from-scratch implementation of the original Transformer
architecture** introduced in the paper *Attention Is All You Need* (Vaswani et al., 2017).

The goal of this project is to demonstrate **deep architectural understanding**.
All components — attention, masking, encoder, decoder, and training —
are implemented manually in PyTorch **without using high-level Transformer libraries**.

This repository is intended as a **clean, educational reference**
for understanding how Transformers work internally.

## What This Repository Demonstrates

This project implements the original Transformer architecture end to end, focusing on
clarity and correctness rather than library usage.

Specifically, it includes:

- A complete Encoder–Decoder Transformer architecture.
- Scaled dot-product attention and multi-head self-attention.
- Sinusoidal positional encoding.
- Residual connections and layer normalization.
- Stacked encoder blocks.
- Decoder blocks with masked self-attention and encoder–decoder (cross) attention.
- Padding masks and causal (look-ahead) masks.
- An end-to-end training pipeline on a toy language modeling task.
- Autoregressive text generation using a trained model.

## Architecture Overview

### Encoder

The encoder maps an input token sequence into a sequence of contextual representations.
It consists of:

- Token embedding with sinusoidal positional encoding
- A stack of identical encoder blocks
- Each encoder block contains:
  - Multi-head self-attention
  - A position-wise feed-forward network
  - Residual connections followed by layer normalization

### Decoder

The decoder generates the output sequence autoregressively, one token at a time,
while attending to the encoder’s output.

It consists of:

- Token embedding with sinusoidal positional encoding
- A stack of identical decoder blocks
- Each decoder block contains:
  - Masked multi-head self-attention (to prevent access to future tokens)
  - Encoder–decoder (cross) attention
  - A position-wise feed-forward network
  - Residual connections followed by layer normalization

The final decoder output is projected to the vocabulary space using a linear layer.


## Project Structure

The repository is organized to keep model architecture, training logic,
and experiments clearly separated.

model/
├── embedding.py            # Token embedding layer
├── positional_encoding.py  # Sinusoidal positional encoding
├── attention.py            # Scaled dot-product and multi-head attention
├── encoder.py              # Encoder blocks and encoder stack
├── decoder.py              # Decoder blocks and decoder stack
└── transformer.py          # Full Transformer (encoder + decoder)

training/
├── dataset.py              # Toy dataset for language modeling
├── utils.py                # Masking utilities
└── train.py                # Training loop and text generation

The `model/` directory contains the complete Transformer architecture,
implemented in a modular and reusable way.

The `training/` directory contains a minimal training pipeline used to
verify correctness and demonstrate autoregressive text generation.


### Requirements
- Python 3.8+
- PyTorch

Install dependencies:

```bash
pip install torch


### Training

Run the training script from the project root:

```bash
python training/train.py



