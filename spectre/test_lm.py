from __future__ import annotations
import argparse, csv, time, sys
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2Config

from model import SpectreLM 

# ─── Patch ColumnParallelLinear (handles all import paths) ──────────────────
_DummyCPL = None
def _ensure_cpl():
    global _DummyCPL
    if _DummyCPL is None:
        class _DummyCPL(nn.Linear):
            def __init__(self, *a, **kw): super().__init__(*a, **kw)
        _DummyCPL = _DummyCPL
    return _DummyCPL

def _patch_cpl(module_name: str):
    try:
        mod = __import__(module_name, fromlist=["*"])
        if not isinstance(getattr(mod, "ColumnParallelLinear", None), type):
            setattr(mod, "ColumnParallelLinear", _ensure_cpl())
    except ModuleNotFoundError:
        pass

_patch_cpl("flash_attn.modules.mlp")
from flash_attn.models.gpt import GPTLMHeadModel
_patch_cpl("flash_attn.models.gpt")

# ════════════════════════════════════════════════════════════════════════════
# Helper attention layers
# ════════════════════════════════════════════════════════════════════════════
class SimpleCausalAttention(nn.Module):
    """Naïve causal scaled‑dot‑product attention."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads: raise ValueError("dim % heads != 0")
        self.h, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + torch.full((n, n), float("-inf"), device=x.device).triu(1)
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)

class FlashSDPAttention(nn.Module):
    """`torch.scaled_dot_product_attention` using Flash backend."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads: raise ValueError("dim % heads != 0")
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = [t.reshape(b * self.h, n, self.dh) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        out = out.reshape(b, self.h, n, self.dh).transpose(2, 1).reshape(b, n, d)
        return self.proj(out)

# ════════════════════════════════════════════════════════════════════════════
# Tiny GPT wrapper (baseline & Flash‑SDPA)
# ════════════════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class GPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float, attn_cls: type[nn.Module]):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = attn_cls(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class TinyGPT(nn.Module):
    """Quick GPT with selectable attention."""
    def __init__(self, *, vocab: int, seq: int, dim: int,
                 depth: int, heads: int, ratio: float, attn_cls: type[nn.Module]):
        super().__init__()
        self.seq_len = seq
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.blocks = nn.ModuleList([
            GPTBlock(dim, heads, ratio, attn_cls) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, tokens: torch.Tensor):
        if tokens.size(1) > self.seq_len:
            raise RuntimeError(f"max seq_len={self.seq_len}, got {tokens.size(1)}")
        x = self.tok(tokens) + self.pos[:, : tokens.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

# ════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ════════════════════════════════════════════════════════════════════════════
def timed(model: nn.Module, dev: torch.device, b: int, n: int,
          runs: int, vocab: int):
    """Return mean±std forward‑pass latency (ms)."""
    model.to(dev).eval()
    tokens = torch.randint(vocab, (b, n), device=dev, dtype=torch.long)
    with torch.no_grad():
        # warm‑up
        for _ in range(5):
            model(tokens)
        lat = []
        for _ in range(runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                model(tokens)
                end.record()
                torch.cuda.synchronize()
                lat.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(tokens)
                lat.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(lat)), float(np.std(lat))

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seq_lens", default="128,512,1024,2048,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--batch_sizes", default="1,32")
    args = ap.parse_args()

    dev = torch.device(args.device)
    seq_lens    = [int(s) for s in args.seq_lens.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    runs, vocab = args.runs, 50_000

    depth, dim, heads, ratio = 12, 512, 8, 4.0
    results: list[dict] = []

    for n in seq_lens:
        for b in batch_sizes:
            # build models
            cfg = GPT2Config(
                vocab_size=vocab,
                n_positions=n,
                n_embd=dim,
                n_layer=depth,
                n_head=heads,
                use_flash_attn=True,
                rotary_pct=0.0,
                pad_vocab_size_multiple=1,
            )
            base_models = {
                "baseline": TinyGPT(vocab=vocab, seq=n, dim=dim, depth=depth,
                                    heads=heads, ratio=ratio,
                                    attn_cls=SimpleCausalAttention),
                "sdpa":     TinyGPT(vocab=vocab, seq=n, dim=dim, depth=depth,
                                    heads=heads, ratio=ratio,
                                    attn_cls=FlashSDPAttention).to(
                                        torch.float16 if dev.type=="cuda" else torch.float32),
                "spectre":  SpectreLM(vocab_size=vocab, seq_len=n,
                                      embed_dim=dim, depth=depth,
                                      n_heads=heads, mlp_ratio=ratio,
                                      drop=0.0, dp_rate=0.0,
                                      ).to(
                                        torch.float16 if dev.type=="cuda" else torch.float32),
                "flash_gpt": GPTLMHeadModel(cfg, device=dev,
                                             dtype=torch.float16 if dev.type=="cuda"
                                                  else torch.float32),
            }

            print(f"\n=== seq {n} | batch {b} ===")
            for name, m in base_models.items():
                try:
                    mean, std = timed(m, dev, b, n, runs, vocab)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{name:>10}: OOM (seq={n}, batch={b}), skipping")
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                thru = (b * n) / (mean / 1e3)
                print(f"{name:>10}: {mean:7.2f} ± {std:5.2f} ms | {thru:9.0f} tok/s")
                # record
                results.append({
                    "model": name,
                    "seq_len": n,
                    "batch_size": b,
                    "avg_latency_ms": mean,
                    "std_latency_ms": std,
                    "avg_throughput_tok_s": thru,
                    "speedup_vs_baseline": None,  # fill after all runs
                })

    # compute speedups
    for n in seq_lens:
        for b in batch_sizes:
            base = next((r["avg_latency_ms"] for r in results
                         if r["model"]=="baseline" and r["seq_len"]==n and r["batch_size"]==b), None)
            if base is None:
                continue
            for r in results:
                if r["seq_len"]==n and r["batch_size"]==b and r["model"]!="baseline":
                    r["speedup_vs_baseline"] = base / r["avg_latency_ms"]

    # Save CSV
    Path("plots").mkdir(exist_ok=True)
    if results:
        keys = results[0].keys()
        with Path("benchmark_results_lm.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(results)
        print("\nSaved benchmark_results_lm.csv")
    else:
        print("\nNo results to save (all configurations OOM).")

    # Plots
    metrics = [
        ("avg_latency_ms", "Latency (ms)", False),
        ("avg_throughput_tok_s", "Throughput (tok/s)", True),
        ("speedup_vs_baseline", "Speed‑up vs Baseline (×)", False),
    ]
    order = ["baseline", "sdpa", "flash_gpt", "spectre"]

    for n in seq_lens:
        sub = [r for r in results if r["seq_len"] == n]
        if not sub:
            continue
        for key, ylabel, logy in metrics:
            plt.figure(figsize=(7, 4))
            for m in order:
                xs = []
                ys = []
                for bb in batch_sizes:
                    rec = next((r for r in sub if r["model"]==m and r["batch_size"]==bb), None)
                    if rec and rec[key] is not None:
                        xs.append(bb)
                        ys.append(rec[key])
                if xs:
                    plt.plot(xs, ys, marker="o", label=m)
            plt.title(f"{ylabel} @ seq={n}")
            plt.xlabel("Batch size"); plt.ylabel(ylabel)
            if logy: plt.yscale("log")
            plt.legend(); plt.tight_layout()
            out_file = f"plots/{key.replace('avg_', '')}_seq{n}.png"
            plt.savefig(out_file); plt.close()
    print("Saved plots in plots/")

if __name__ == "__main__":
    main()
