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

# If you need more distinct colors (for multiple image sizes), you can expand:
image_size_colors = {
    32:  "#F8961E",  # orange
    64:  "#F9844A",  # peach
    128: "#277DA1",  # azure
    224: "#90BE6D",  # green
    256: "#F94144",  # red
    512: "#9B5DE5",  # purple
}
# ---------------------------------------------------------------------

from fftnet_vit import FFTNetViT
from transformer import ViT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################################################
# 1) Model Creation (Adjusted to accept dynamic img_size)
##############################################################################
def create_model(model_name, img_size=224):
    """
    Dynamically instantiate the FFTNetViT or ViT model with the given image size.
    
    Args:
        model_name (str): "FFTNetViT" or "ViT"
        img_size (int):   e.g. 32 for CIFAR-10, 224 for ImageNet, etc.

    Returns:
        model (nn.Module): The instantiated PyTorch model
        desc  (str): A short descriptive string for logging
    """
    fftnet_config = {
        'img_size': img_size,
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
        'img_size': img_size,
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
        return FFTNetViT(**fftnet_config), f"FFTNetViT (img_size={img_size})"
    elif model_name == "ViT":
        return ViT(**vit_config), f"ViT (img_size={img_size})"
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
    rank, world_size, model_name, img_size, global_batch_size, num_runs, warmup, return_dict
):
    """
    The function that each process will run under mp.spawn for DDP.
    - rank: process rank (0 to world_size - 1)
    - world_size: total number of GPUs
    - model_name: "FFTNetViT" or "ViT"
    - img_size: e.g., 32 for CIFAR or 224 for ImageNet
    - global_batch_size: the sum of the local batch sizes across all GPUs
    - return_dict: a multiprocessing dict to store final results in rank 0
    """
    ddp_setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    model, desc = create_model(model_name, img_size=img_size)
    model.to(device)

    # Wrap with DistributedDataParallel
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )

    # Each GPU gets a split of the global batch
    local_bs = global_batch_size // world_size
    if local_bs <= 0:
        # Raise an error or just return. We'll raise an error for clarity:
        raise ValueError(
            f"Global batch size {global_batch_size} is too small for {world_size} GPUs. "
            "Resulting local batch size is 0."
        )
    
    dummy_input = torch.randn(local_bs, 3, img_size, img_size).to(device)

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

        # total_time = max of average latencies from all ranks
        total_time = max(avg_times)
        throughput = global_batch_size / total_time
        
        logger.info("  Per-GPU Latencies:")
        for i, (a, s) in enumerate(zip(avg_times, std_times)):
            logger.info(f"    GPU {i}: {a*1e3:.2f} ms (std: {s*1e3:.2f} ms)")
        logger.info(f"  Effective total time: {total_time*1e3:.2f} ms")
        logger.info(f"  Throughput: {throughput:.2f} images/sec\n{'-'*60}")

        # Store in return_dict so the main process can retrieve the results
        return_dict[(model_name, img_size)] = {
            "avg_time": total_time,
            "std_time": max(std_times),
            "throughput": throughput
        }
    cleanup()

def run_ddp_experiment(model_name, img_size, global_batch_size, num_runs, warmup):
    """
    Launches a DDP experiment for a single model, single image size & batch size.
    Returns a dict with keys ["avg_time", "std_time", "throughput"] from rank=0.
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
        args=(world_size, model_name, img_size, global_batch_size, num_runs, warmup, return_dict),
        nprocs=world_size,
        join=True
    )
    
    return return_dict.get((model_name, img_size), None)

##############################################################################
# 3A) Compare Both Models Across Multiple Batch Sizes (One Image Size)
##############################################################################

def compare_models_ddp(batch_sizes, num_runs, warmup, img_size=224):
    """
    For a single 'img_size', compares FFTNetViT & ViT across a list of 'batch_sizes'.
    """
    model_names = ["FFTNetViT", "ViT"]
    results = {
        "FFTNetViT": {"batch_sizes": [], "latencies": [], "throughputs": []},
        "ViT":       {"batch_sizes": [], "latencies": [], "throughputs": []}
    }

    for bs in batch_sizes:
        logger.info(f"\n===== GLOBAL BATCH SIZE: {bs}, IMG_SIZE: {img_size} =====")
        for m in model_names:
            outcome = run_ddp_experiment(m, img_size, bs, num_runs, warmup)
            if outcome is None:
                # Means we either can't do DDP or batch_size wasn't divisible
                results[m]["batch_sizes"].append(bs)
                results[m]["latencies"].append(float("nan"))
                results[m]["throughputs"].append(float("nan"))
            else:
                avg_time = outcome["avg_time"]
                throughput = outcome["throughput"]
                results[m]["batch_sizes"].append(bs)
                results[m]["latencies"].append(avg_time)
                results[m]["throughputs"].append(throughput)

    # -- Plot Latency vs Batch Size
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
    plt.title(f"Latency vs Batch Size (DDP) - Img {img_size}x{img_size}")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Latency (ms)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ddp_latency_comparison_{img_size}.pdf")
    plt.close()

    # -- Plot Throughput vs Batch Size
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
    plt.title(f"Throughput vs Batch Size (DDP) - Img {img_size}x{img_size}")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Throughput (images/sec)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ddp_throughput_comparison_{img_size}.pdf")
    plt.close()

    logger.info(f"\nSaved comparison plots for img_size={img_size}:")
    logger.info(f"  ddp_latency_comparison_{img_size}.pdf")
    logger.info(f"  ddp_throughput_comparison_{img_size}.pdf")

    return results  # Return dictionary if you want further analysis

##############################################################################
# 3B) Compare Both Models Across MULTIPLE Image Sizes & Make Combined Plots
##############################################################################

def compare_models_ddp_across_image_sizes(
    image_sizes, 
    batch_sizes,
    num_runs=10,
    warmup=3
):
    """
    For each image_size in 'image_sizes', and each global batch size in 'batch_sizes',
    run DDP comparisons for both FFTNetViT and ViT. Store results and produce:
      - a pair of plots per image size,
      - optional "combined" plots with multiple lines (one line per (model,img_size)).
    """
    # We'll store results in a nested dict:
    #   results[(model_name, img_size)]: 
    #       {"batch_sizes": [...], "latencies": [...], "throughputs": [...]}
    results = {}

    model_names = ["FFTNetViT", "ViT"]

    # -- Gather data
    for img_size in image_sizes:
        for m in model_names:
            results[(m, img_size)] = {
                "batch_sizes": [],
                "latencies": [],
                "throughputs": []
            }
        logger.info(f"\n\n***** IMAGE SIZE: {img_size}x{img_size} *****")
        for bs in batch_sizes:
            logger.info(f"Global Batch Size: {bs}")
            outcome_fft = run_ddp_experiment("FFTNetViT", img_size, bs, num_runs, warmup)
            outcome_vit = run_ddp_experiment("ViT",       img_size, bs, num_runs, warmup)

            for (model_name, outcome) in zip(["FFTNetViT","ViT"], [outcome_fft, outcome_vit]):
                if outcome is None:
                    results[(model_name, img_size)]["batch_sizes"].append(bs)
                    results[(model_name, img_size)]["latencies"].append(float("nan"))
                    results[(model_name, img_size)]["throughputs"].append(float("nan"))
                else:
                    avg_time   = outcome["avg_time"]
                    throughput = outcome["throughput"]
                    results[(model_name, img_size)]["batch_sizes"].append(bs)
                    results[(model_name, img_size)]["latencies"].append(avg_time)
                    results[(model_name, img_size)]["throughputs"].append(throughput)
        
        # Optionally, we can also generate the 2-plot set for *each* image size,
        # (just as we did in `compare_models_ddp`).
        # We'll do that here for completeness:
        fft_lat = [l*1000 for l in results[("FFTNetViT", img_size)]["latencies"]]
        vit_lat = [l*1000 for l in results[("ViT", img_size)]["latencies"]]
        fft_thr = results[("FFTNetViT", img_size)]["throughputs"]
        vit_thr = results[("ViT", img_size)]["throughputs"]
        x_bs    = results[("FFTNetViT", img_size)]["batch_sizes"]  # same as ViT's

        # (A) Latency vs Batch Size
        plt.figure(figsize=(8, 6))
        plt.plot(x_bs, fft_lat, **fftnet_style, label="FFTNetViT")
        plt.plot(x_bs, vit_lat, **vit_style,     label="ViT")
        plt.title(f"Latency vs Batch Size (DDP) - Img {img_size}x{img_size}")
        plt.xlabel("Global Batch Size", fontweight="bold")
        plt.ylabel("Latency (ms)", fontweight="bold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"ddp_latency_{img_size}.pdf")
        plt.close()

        # (B) Throughput vs Batch Size
        plt.figure(figsize=(8, 6))
        plt.plot(x_bs, fft_thr, **fftnet_style, label="FFTNetViT")
        plt.plot(x_bs, vit_thr, **vit_style,     label="ViT")
        plt.title(f"Throughput vs Batch Size (DDP) - Img {img_size}x{img_size}")
        plt.xlabel("Global Batch Size", fontweight="bold")
        plt.ylabel("Throughput (images/sec)", fontweight="bold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"ddp_throughput_{img_size}.pdf")
        plt.close()

    # --- Now let's create COMBINED plots across all image_sizes in one figure ---
    # 1) Combined Latency vs Batch Size
    plt.figure(figsize=(8, 6))
    for img_size in image_sizes:
        # For each image size, we get the FFTNetViT latencies and make a line:
        x_bs  = results[("FFTNetViT", img_size)]["batch_sizes"]
        y_lat = [l * 1000 for l in results[("FFTNetViT", img_size)]["latencies"]]
        plt.plot(
            x_bs,
            y_lat,
            linestyle="--",
            marker="o",
            linewidth=2,
            color=image_size_colors.get(img_size, "#000000"),  # fallback black if not in dict
            label=f"FFTNetViT - {img_size}x{img_size}"
        )
    # Then do the same for ViT in a loop:
    for img_size in image_sizes:
        x_bs  = results[("ViT", img_size)]["batch_sizes"]
        y_lat = [l * 1000 for l in results[("ViT", img_size)]["latencies"]]
        plt.plot(
            x_bs,
            y_lat,
            linestyle="-",
            marker="s",
            linewidth=2,
            color=image_size_colors.get(img_size, "#000000"),
            label=f"ViT - {img_size}x{img_size}"
        )
    plt.title("Combined Latency vs Batch Size (DDP) - Multiple Image Sizes")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Latency (ms)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_ddp_latency_across_img_sizes.pdf")
    plt.close()

    # 2) Combined Throughput vs Batch Size
    plt.figure(figsize=(8, 6))
    for img_size in image_sizes:
        x_bs  = results[("FFTNetViT", img_size)]["batch_sizes"]
        y_thr = results[("FFTNetViT", img_size)]["throughputs"]
        plt.plot(
            x_bs,
            y_thr,
            linestyle="--",
            marker="o",
            linewidth=2,
            color=image_size_colors.get(img_size, "#000000"),
            label=f"FFTNetViT - {img_size}x{img_size}"
        )
    for img_size in image_sizes:
        x_bs  = results[("ViT", img_size)]["batch_sizes"]
        y_thr = results[("ViT", img_size)]["throughputs"]
        plt.plot(
            x_bs,
            y_thr,
            linestyle="-",
            marker="s",
            linewidth=2,
            color=image_size_colors.get(img_size, "#000000"),
            label=f"ViT - {img_size}x{img_size}"
        )
    plt.title("Combined Throughput vs Batch Size (DDP) - Multiple Image Sizes")
    plt.xlabel("Global Batch Size", fontweight="bold")
    plt.ylabel("Throughput (images/sec)", fontweight="bold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_ddp_throughput_across_img_sizes.pdf")
    plt.close()

    logger.info("\nSaved combined multi-size plots (PDF):")
    logger.info("  combined_ddp_latency_across_img_sizes.pdf")
    logger.info("  combined_ddp_throughput_across_img_sizes.pdf")

    return results  # Return everything for further offline analysis if desired

##############################################################################
# 4) Main Function & CLI
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser("DDP Comparison for FFTNetViT vs. ViT")
    
    # Examples: --image_sizes 32 64 128 224
    parser.add_argument("--image_sizes", type=int, nargs="+", 
                        default=[32, 224],  # CIFAR-10 (32x32) + ImageNet (224x224)
                        help="List of image sizes to test (height=width).")
    parser.add_argument("--batch_sizes", type=int, nargs="+", 
                        default=[8,16,32,64,128],  # choose sizes that are safe for your GPUs
                        help="List of global batch sizes to test.")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of timed runs for latency measurement.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs.")
    return parser.parse_args()

def main():
    args = parse_args()

    if torch.cuda.device_count() < 2:
        logger.warning("Less than 2 GPUs detected. This script only runs DDP. Exiting.")
        return

    logger.info(f"Running DDP comparisons for: FFTNetViT vs. ViT")
    logger.info(f"Image sizes: {args.image_sizes}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"num_runs: {args.num_runs}, warmup: {args.warmup}")

    # (A) If you only want to test a single image size at a time, use compare_models_ddp:
    # compare_models_ddp(batch_sizes=args.batch_sizes,
    #                    num_runs=args.num_runs,
    #                    warmup=args.warmup,
    #                    img_size=224)

    # (B) If you want to test multiple image sizes in one run, use:
    compare_models_ddp_across_image_sizes(
        image_sizes=args.image_sizes,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        warmup=args.warmup
    )

if __name__ == "__main__":
    main()
