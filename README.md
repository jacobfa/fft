# SPECTRE: Frequency-Domain Token Mixing for Transformers

A lightweight PyTorch implementation of **SPECTRE** (Spectral Token Routing) ― a drop-in replacement for self-attention that mixes tokens in the frequency domain with a single real FFT.  This repository exposes:

* **`spectre.py`** – the core frequency-mixing layer (`SPECTRELayer`), a residual block (`SPECTREBlock`), and supporting utilities such as the low-overhead **prefix-FFT cache** for fast autoregressive decoding.
* **`model.py`** – reference **Vision Transformer** (`SpectreViT`) and **Causal Language Model** (`SpectreLM`) that stack `SPECTREBlock`s.

> ⚡️ *SPECTRE achieves comparable accuracy to attention while being **O(N log N)** in sequence length and easily parallelisable across heads.*


## Quick Start

### Vision – CIFAR-10

```python
from model import SpectreViT
import torch, torchvision as tv

# model
net = SpectreViT(img_size=32, patch_size=4, embed_dim=384, depth=12, n_heads=6)

# dummy forward pass
x = torch.randn(8, 3, 32, 32)
logits = net(x)              # (8, 10)
print(logits.shape)
```

### Language – GPT-style

```python
from model import SpectreLM
import torch

vocab_size = 50257
seq_len     = 1024

lm = SpectreLM(vocab_size, max_seq_len=seq_len,
               d_model=768, n_layers=12, n_heads=12)

idx = torch.randint(0, vocab_size, (4, seq_len))  # toy batch
logits, _ = lm(idx)                                # (4, L, vocab)
print(logits.shape)
```

### Streaming / Incremental Decoding

```python
caches = lm.init_caches(device="cuda")
for t in range(seq_len):
    next_token = idx[:, t:t+1]
    logits, caches = lm(next_token, caches=caches, incremental=True)
```

---

## API Highlights

| Module | Purpose |
|--------|---------|
| `PrefixFFTCache` | Maintains running real-FFT coefficients and mean queries during generation (⚡ constant-time updates). |
| `SPECTRELayer` | Frequency-domain token mixer with optional low-rank outer product **(rank *r*)** and wavelet refinement branch. Replaces Multi-Head Attention 1-to-1. |
| `SPECTREBlock` | Transformer block: **LN → SPECTRE → residual → FFN → residual**. |
| `SpectreViT` | Vision Transformer for small images (e.g. CIFAR-10, Tiny-ImageNet). |
| `SpectreLM` | GPT-like decoder-only language model with tied embeddings and streaming support. |

### Key Constructor Args

```text
SPECTRELayer(
  d_model:     int,           # token embedding dim
  n_heads:     int,           # #heads (d_model must be divisible)
  max_seq_len: int = 8192,    # pos-encoding & FFT size
  low_rank:    int | None = None,   # optional outer-product rank r
  use_wavelet: bool = False,  # enable Haar refinement branch
  share_gates: bool = True,   # share frequency gates across heads
)
```

---

## Training Tips

* **Warm-up** the `PrefixFFTCache` length‐scales by training with full sequences first, then enable streaming.
* When `use_wavelet=True`, start with **`skip_init ≈ 0.9`** (branch mostly skipped) and let it learn to deviate.
* For long sequences set `max_seq_len` to the *longest* you expect; the FFT size is fixed after init.

---

## Citation

If you use this implementation, please consider citing the original paper:

```bibtex
@misc{feinashley2025spectrefftbasedefficientdropin,
      title={SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts}, 
      author={Jacob Fein-Ashley and Neelesh Gupta and Rajgopal Kannan and Viktor Prasanna},
      year={2025},
      eprint={2502.18394},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18394}, 
}
```

---

## License

MIT License. See `LICENSE` for details.
