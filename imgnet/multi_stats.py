import os
import time
import statistics
import argparse
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Color variants and styles as specified
# ---------------------------------------------------------------------
variant_colors = {
    "Base":  "#F94144",  # bright red
    "Large": "#90BE6D",  # bright green
    "Huge":  "#577590",  # deep, dusty navy
}

fftnet_style = {
    "linestyle": "--",
    "marker": "o",
    "linewidth": 2,
    "color": variant_colors["Base"]   # assign bright red for FFTNetViT
}
vit_style = {
    "linestyle": "-",
    "marker": "s",
    "linewidth": 2,
    "color": variant_colors["Large"]  # assign bright green for ViT
}
# ---------------------------------------------------------------------

from fftnet_vit import FFTNetViT
from transformer import ViT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################################################
# 1) Model Creation: We'll just define 'Base' config for both FFTNetViT & ViT
##############################################################################
def create_model(model_name):
    """
    Returns:
        model (nn.Module): The instantiated PyTorch model
        desc  (str): A short descriptive string for logging
    """
    fftnet_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 768,
        'depth': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'num_heads': 12,
        'adaptive_spectral': True
    }
    vit_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 768,
        'depth': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'num_heads': 12
    }
    if model_name == "FFTNetViT":
        return FFTNetViT(**fftnet_config), "FFTNetViT (Base)"
    elif model_name == "ViT":
        return ViT(**vit_config), "ViT (Base)"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

##############################################################################
# 2) DDP Setup and Single-Experiment Runner
##############################################################################

def ddp_setup(rank, world_size):
    """
    Initializes the default process group for DDP (using 'nccl' backend).
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """ Clean up the distributed process group. """
    dist.destroy_process_group()

def measure_latency(model, input_tensor, num_runs=10, warmup=3):
    """
    Measures inference latency for a single model on a single GPU.
    Returns a list of times (in seconds) for each run.
    """
    model.eval()
    with torch.no_grad():
        # Warm-up runs (not timed)
        for _ in range(warmup):
            _ = model(input_tensor)
        
        # Timed runs
        timings = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
    return timings

def ddp_worker(
    rank, world_size, model_name, global_batch_size, num_runs, warmup, return_dict
):
    """
    The function that each process will run under mp.spawn for DDP.
    - rank: process rank (0 to world_size - 1)
    - world_size: total number of GPUs
    - model_name: "FFTNetViT" or "ViT"
    - global_batch_size: the sum of the local batch sizes across all GPUs
    - return_dict: a multiprocessing dict to store final results in rank 0
    """
    ddp_setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    model, desc = create_model(model_name)
    model.to(device)

    # Wrap with DistributedDataParallel
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )

    # Each GPU gets a split of the global batch
    local_bs = global_batch_size // world_size
    # If local_bs is 0, the shape could be (0, 3, 224, 224), causing cuFFT errors.
    if local_bs <= 0:
        # Raise an error or just return. We'll raise an error for clarity:
        raise ValueError(
            f"Global batch size {global_batch_size} is too small for {world_size} GPUs. "
            "Resulting local batch size is 0."
        )
    
    dummy_input = torch.randn(local_bs, 3, 224, 224).to(device)

    # Measure per-rank latencies
    times = measure_latency(ddp_model, dummy_input, num_runs=num_runs, warmup=warmup)
    avg_time = statistics.mean(times)
    std_time = statistics.pstdev(times) if len(times) > 1 else 0.0

    # Gather from all ranks
    avg_tensor = torch.tensor([avg_time], device=device)
    std_tensor = torch.tensor([std_time], device=device)
    avg_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    std_list = [torch.zeros(1, device=device) for _ in range(world_size)]

    dist.all_gather(avg_list, avg_tensor)
    dist.all_gather(std_list, std_tensor)

    if rank == 0:
        desc_str = f"[DDP] {desc}"
        logger.info(desc_str)
        logger.info(f"  Global Batch Size: {global_batch_size}, GPUs: {world_size}")
        
        # Convert to floats
        avg_times = [t.item() for t in avg_list]
        std_times = [t.item() for t in std_list]

        # Simple approach: total time = max of average latencies from all ranks
        total_time = max(avg_times)
        throughput = global_batch_size / total_time
        
        logger.info("  Per-GPU Latencies:")
        for i, (a, s) in enumerate(zip(avg_times, std_times)):
            logger.info(f"    GPU {i}: {a*1e3:.2f} ms (std: {s*1e3:.2f} ms)")
        logger.info(f"  Effective total time: {total_time*1e3:.2f} ms")
        logger.info(f"  Throughput: {throughput:.2f} images/sec\n{'-'*60}")

        # Store in return_dict so the main process can retrieve the results
        return_dict[model_name] = {
            "avg_time": total_time,
            "std_time": max(std_times),
            "throughput": throughput
        }
    cleanup()

def run_ddp_experiment(model_name, global_batch_size, num_runs, warmup):
    """
    Launches a DDP experiment for a single model & batch size.
    Returns a dict with keys ["avg_time", "std_time", "throughput"] from rank 0.
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning("DDP requires >=2 GPUs. Cannot proceed. Returning None.")
        return None
    
    # Ensure global_batch_size is divisible by world_size to avoid local_bs=0
    if global_batch_size % world_size != 0:
        logger.error(
            f"Global batch size {global_batch_size} not divisible by "
            f"the number of GPUs {world_size}. Skipping this run."
        )
        return None

    import multiprocessing
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    mp.spawn(
        ddp_worker,
        args=(world_size, model_name, global_batch_size, num_runs, warmup, return_dict),
        nprocs=world_size,
        join=True
    )
    
    return return_dict.get(model_name, None)

##############################################################################
# 3) Compare Both Models Across Multiple Batch Sizes & Plot
##############################################################################

def compare_models_ddp(batch_sizes, num_runs, warmup):
    """
    For each batch size, runs DDP for both models (FFTNetViT & ViT).
    Collects latencies and throughputs, then plots them as PDFs.
    
    Why "CUFFT_INVALID_SIZE" can happen:
      If the local batch size = 0 on any GPU, the tensor shape for one dimension
      becomes zero. The cuFFT library will throw an error for an invalid transform
      size. The fix is to ensure "global_batch_size >= world_size" and is evenly
      divisible so each GPU has at least 1 sample.
    """
    model_names = ["FFTNetViT", "ViT"]
    # We'll store results in a dictionary:
    # results[model_name]["batch_sizes"] = [...]
    # results[model_name]["latencies"]   = [...]
    # results[model_name]["throughputs"] = [...]
    results = {
        "FFTNetViT": {"batch_sizes": [], "latencies": [], "throughputs": []},
        "ViT":       {"batch_sizes": [], "latencies": [], "throughputs": []}
    }

    for bs in batch_sizes:
        logger.info(f"\n===== GLOBAL BATCH SIZE: {bs} =====")
        for m in model_names:
            outcome = run_ddp_experiment(m, bs, num_runs, warmup)
            if outcome is None:
                # Means we either can't do DDP (1 GPU) or batch_size wasn't divisible
                results[m]["batch_sizes"].append(bs)
                results[m]["latencies"].append(float("nan"))
                results[m]["throughputs"].append(float("nan"))
            else:
                avg_time = outcome["avg_time"]
                throughput = outcome["throughput"]
                results[m]["batch_sizes"].append(bs)
                results[m]["latencies"].append(avg_time)
                results[m]["throughputs"].append(throughput)

    # Now we create 2 comparison plots:
    #  (1) Latency vs Batch Size (two lines: one for FFTNetViT, one for ViT)
    #  (2) Throughput vs Batch Size (two lines)
    # We'll save them as PDF files.

    # 1) Latency vs Batch Size
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["FFTNetViT"]["batch_sizes"],
        [l * 1000 for l in results["FFTNetViT"]["latencies"]],  # sec -> ms
        **fftnet_style,
        label="FFTNetViT",
    )
    plt.plot(
        results["ViT"]["batch_sizes"],
        [l * 1000 for l in results["ViT"]["latencies"]],  # sec -> ms
        **vit_style,
        label="ViT",
    )
    plt.title("Latency vs Batch Size (DDP)")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Latency (ms)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ddp_latency_comparison.pdf")  # Save as PDF
    plt.close()

    # 2) Throughput vs Batch Size
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["FFTNetViT"]["batch_sizes"],
        results["FFTNetViT"]["throughputs"],
        **fftnet_style,
        label="FFTNetViT",
    )
    plt.plot(
        results["ViT"]["batch_sizes"],
        results["ViT"]["throughputs"],
        **vit_style,
        label="ViT",
    )
    plt.title("Throughput vs Batch Size (DDP)")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Throughput (images/sec)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ddp_throughput_comparison.pdf")  # Save as PDF
    plt.close()

    logger.info("\nSaved comparison plots (PDF):")
    logger.info("  ddp_latency_comparison.pdf")
    logger.info("  ddp_throughput_comparison.pdf")

##############################################################################
# 4) Main Function & CLI
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser("DDP Comparison for FFTNetViT vs. ViT")
    parser.add_argument("--batch_sizes", type=int, nargs="+", 
                        default=[8,16,32,128,256,512],  # updated default
                        help="List of global batch sizes to test. Must be divisible by #GPUs.")
    parser.add_argument("--num_runs", type=int, default=40,
                        help="Number of timed runs for latency measurement.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs.")
    return parser.parse_args()

def main():
    args = parse_args()

    if torch.cuda.device_count() < 2:
        logger.warning("Less than 2 GPUs detected. This script only runs DDP. Exiting.")
        return

    logger.info(f"Running DDP comparisons for: FFTNetViT vs. ViT")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"num_runs: {args.num_runs}, warmup: {args.warmup}")

    compare_models_ddp(
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        warmup=args.warmup
    )

if __name__ == "__main__":
    main()
