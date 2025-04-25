#!/usr/bin/env python3
# coding: utf-8

"""
Compare three multi-head attention variants under DDP:
  1) "sdpa":      Tiny GPT using Flash-SDP (torch.scaled_dot_product_attention).
  2) "flash_gpt": GPTLMHeadModel from flash_attn.models.gpt
  3) "spectre":   Your FFT-based SpectreLM.

Example usage:
  torchrun --nproc_per_node=4 test_lm.py --seq_lens=512,1024 --batch_sizes=1,8

This script:
  - Initializes a distributed process group (NCCL).
  - Builds each model, wraps in DDP, and benchmarks forward-pass latency.
  - Rank 0 will save CSV results and a log-log latency plot.
"""

import os
import argparse, csv
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Monkey-patch flash_attn to avoid `isinstance(..., None)` TypeError
# ---------------------------------------------------------------------------
import flash_attn.models.gpt as fagpt
if getattr(fagpt, "ColumnParallelLinear", None) is None:
    # Provide a dummy class so `isinstance(self.lm_head, ColumnParallelLinear)` won't crash
    class ColumnParallelLinear(nn.Module):
        pass
    fagpt.ColumnParallelLinear = ColumnParallelLinear

# Now we can safely import GPTLMHeadModel
from flash_attn.models.gpt import GPTLMHeadModel
from transformers import GPT2Config

# Your Spectre model (adjust path/import as needed).
from model import SpectreLM

# ---------------------------------------------------------------------------
# 1) SDPA Implementation (Tiny GPT)
# ---------------------------------------------------------------------------
class FlashSDPAttention(nn.Module):
    """Uses torch.scaled_dot_product_attention (Flash-optimized)."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, self.dh)  # (b, n, 3, h, dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                  # each => (b, h, n, dh)
        # Flatten heads into batch
        q, k, v = [t.reshape(b*self.h, n, self.dh) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        # Reshape back
        out = out.reshape(b, self.h, n, self.dh).transpose(2, 1).reshape(b, n, d)
        return self.proj(out)

class TinyMLP(nn.Module):
    """Simple MLP block."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.act(self.fc1(x)))

class GPTBlock(nn.Module):
    """One GPT-style block with flash-SDP attention."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = FlashSDPAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = TinyMLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SDPAGPT(nn.Module):
    """A 'Tiny GPT' using Flash-SDP."""
    def __init__(self, vocab_size: int, seq_len: int,
                 dim: int, depth: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.seq_len = seq_len
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([
            GPTBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_embed.weight

    def forward(self, tokens: torch.Tensor):
        if tokens.size(1) > self.seq_len:
            raise RuntimeError(f"Max seq_len={self.seq_len}, got {tokens.size(1)}")
        x = self.tok_embed(tokens) + self.pos_embed[:, : tokens.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

# ---------------------------------------------------------------------------
# 2) FlashAttention GPT (no parallel head)
# ---------------------------------------------------------------------------
def build_flash_gpt(vocab_size: int, seq_len: int,
                    dim: int, depth: int, heads: int):
    cfg = GPT2Config(
        vocab_size = vocab_size,
        n_positions= seq_len,
        n_embd     = dim,
        n_layer    = depth,
        n_head     = heads,
        use_flash_attn = True,
        rotary_pct = 0.0,
        pad_vocab_size_multiple=1,
    )
    # By passing process_group=None, we avoid parallel head creation.
    model = GPTLMHeadModel(cfg, process_group=None)
    return model

# ---------------------------------------------------------------------------
# 3) SpectreLM (Your FFT-based model)
# ---------------------------------------------------------------------------
def build_spectre(vocab_size: int, seq_len: int,
                  dim: int, depth: int, heads: int, mlp_ratio: float):
    model = SpectreLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=dim,
        depth=depth,
        n_heads=heads,
        mlp_ratio=mlp_ratio,
        drop=0.0,
        dp_rate=0.0
    )
    return model

# ---------------------------------------------------------------------------
# Benchmark / DDP utilities
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_lens", default="512,1024",
                    help="Comma-separated sequence lengths to test")
    ap.add_argument("--batch_sizes", default="1,8",
                    help="Comma-separated local batch sizes to test")
    ap.add_argument("--runs", type=int, default=3,
                    help="Number of timed iterations per config")
    ap.add_argument("--vocab_size", type=int, default=50000)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--dim",   type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    return ap.parse_args()

def main():
    # 1) Initialize the process group (requires `torchrun`)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    # 2) Set the local GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    args = parse_args()
    if rank == 0:
        print(f"DDP on {world_size} processes. Rank={rank}, local GPU={local_rank}")

    seq_lens    = [int(x) for x in args.seq_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    model_names = ["sdpa", "flash_gpt", "spectre"]

    def build_model(name: str, seq_len: int):
        if name == "sdpa":
            return SDPAGPT(
                vocab_size=args.vocab_size,
                seq_len=seq_len,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_ratio=args.mlp_ratio,
            )
        elif name == "flash_gpt":
            return build_flash_gpt(
                vocab_size=args.vocab_size,
                seq_len=seq_len,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads
            )
        elif name == "spectre":
            return build_spectre(
                vocab_size=args.vocab_size,
                seq_len=seq_len,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                mlp_ratio=args.mlp_ratio
            )
        else:
            raise ValueError(f"Unknown model name {name}")

    # -----------------------------------------------------------------------
    # Timed forward pass
    # -----------------------------------------------------------------------
    def timed_pass(model: nn.Module, local_b: int, seqlen: int, runs: int):
        """Time the forward pass across 'runs' iterations, returning avg ms across all GPUs."""
        model.eval()
        tokens = torch.randint(args.vocab_size, (local_b, seqlen),
                               device=device, dtype=torch.long)

        # Warm-up
        with torch.no_grad():
            for _ in range(2):
                _ = model(tokens)

        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            dist.barrier()  # sync all ranks before timing
            start.record()
            with torch.no_grad():
                _ = model(tokens)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        # local average
        local_avg_ms = float(np.mean(times))
        # all-reduce to get global average
        t_tensor = torch.tensor([local_avg_ms], device=device)
        dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
        global_avg_ms = t_tensor.item() / world_size
        return global_avg_ms

    results = []

    try:
        for model_name in model_names:
            for seq_len in seq_lens:
                # Build the model
                model = build_model(model_name, seq_len)

                # For FlashAttention-based models, half precision is typical
                if model_name in ["flash_gpt", "spectre"]:
                    model = model.half()

                model = model.to(device)
                ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

                # Example skip for extremely large seq_len
                if (model_name == "sdpa" and seq_len >= 16384) or \
                   (model_name == "flash_gpt" and seq_len >= 262144) or \
                     (model_name == "spectre" and seq_len >= 262144):
                    if rank == 0:
                        print(f"Skipping {model_name} for seq_len={seq_len} (>=16k).")
                    del ddp_model
                    torch.cuda.empty_cache()
                    continue

                for b in batch_sizes:
                    if rank == 0:
                        print(f"[{model_name}] seq={seq_len}, local_batch={b}")
                    try:
                        avg_ms = timed_pass(ddp_model, b, seq_len, args.runs)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if rank == 0:
                                print("  OOM => skipping.")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                    # throughput across all GPUs
                    tokens_s = (b * world_size * seq_len) / (avg_ms / 1e3)
                    if rank == 0:
                        print(f"  Avg latency: {avg_ms:.1f} ms | throughput ~ {tokens_s:,.0f} tok/s")
                        results.append({
                            "model": model_name,
                            "seq_len": seq_len,
                            "local_batch_size": b,
                            "world_size": world_size,
                            "avg_latency_ms": avg_ms,
                            "tokens_per_s": tokens_s,
                        })

                # Cleanup
                del ddp_model
                del model
                torch.cuda.empty_cache()

    finally:
        # Ensure we cleanly shut down the process group
        dist.barrier()
        dist.destroy_process_group()

    # Only rank 0 saves results
    if rank == 0 and results:
        Path("plots").mkdir(exist_ok=True)
        # Save CSV
        csv_path = Path("benchmark_results.csv")
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV => {csv_path}")

        # Make a quick log-log plot
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            grouped[(r["model"], r["local_batch_size"])].append(r)

        plt.figure(figsize=(7,5))
        plt.title(f"Latency vs Seq Len (DDP x {world_size} GPUs)")

        # Collect all tested seq_lens
        all_seqs = sorted(set(r["seq_len"] for r in results))

        # color per model, style per batch size
        model_colors = {"sdpa": "blue", "flash_gpt": "green", "spectre": "orange"}
        style_map = {1:"o-", 2:"s-", 4:"^-", 8:"x-", 16:"*-", 32:"D-"}

        for (mn, bsz), recs in grouped.items():
            color = model_colors.get(mn, "black")
            style = style_map.get(bsz, "o-")
            recs_sorted = sorted(recs, key=lambda x: x["seq_len"])
            xs = [rc["seq_len"] for rc in recs_sorted]
            ys = [rc["avg_latency_ms"] for rc in recs_sorted]
            lbl = f"{mn}(b={bsz})"
            plt.plot(xs, ys, style, color=color, label=lbl)

        # Reference lines: O(n), O(n log n), O(n^2)
        if all_seqs:
            import math
            min_seq = min(all_seqs)
            anchor_lat = next((rc["avg_latency_ms"] for rc in results
                               if rc["seq_len"] == min_seq), 1.0)

            def ref_n(x):
                return (x / min_seq) * anchor_lat
            def ref_nlogn(x):
                return ((x * math.log2(x + 1)) / 
                        (min_seq * math.log2(min_seq + 1))) * anchor_lat
            def ref_n2(x):
                return (x * x) / (min_seq * min_seq) * anchor_lat

            xs_ref = np.array(all_seqs)
            plt.plot(xs_ref, [ref_n(xx) for xx in xs_ref], "--", color="gray", label="O(n)")
            plt.plot(xs_ref, [ref_nlogn(xx) for xx in xs_ref], "--", color="magenta", label="O(n log n)")
            plt.plot(xs_ref, [ref_n2(xx) for xx in xs_ref], "--", color="red", label="O(n^2)")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Sequence Length (log)")
        plt.ylabel("Latency (ms, log)")
        plt.legend()
        plt.tight_layout()

        out_plot = Path("plots/latency_vs_seq_len.png")
        plt.savefig(out_plot)
        plt.close()
        print(f"Saved plot => {out_plot}")

    elif rank == 0:
        print("\nNo valid results to save.")


if __name__ == "__main__":
    main()
