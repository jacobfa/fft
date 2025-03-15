import torch
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # Explicit import for custom styles
from ptflops import get_model_complexity_info
from fftnet_vit import FFTNetViT
from transformer import ViT

# Set the plotting style and attractive Seaborn palette.
plt.style.use(['science', 'no-latex', 'ieee'])
# We'll use fixed colors for each variant type.
variant_colors = {"Base": "red", "Large": "blue", "Huge": "black"}
sns.set_palette(sns.color_palette(list(variant_colors.values())))

# Set up logging: both console and file output.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_metrics.log", mode='w')
    ]
)
logger = logging.getLogger()

def compute_metrics(model, input_res=(3, 224, 224)):
    """
    Compute the GMACs and parameter counts using ptflops.
    Since ptflops counts one multiply–add as a single MAC,
    we multiply the GMACs by 2 to get FLOPs.
    """
    gmacs, params = get_model_complexity_info(
        model, input_res, as_strings=False, print_per_layer_stat=False
    )
    flops = gmacs * 2  # Convert GMACs to FLOPs (each MAC = 2 FLOPs)
    return flops, params

def format_metrics(flops, params):
    flops_str = f"{flops / 1e9:.2f} GFLOPs"
    params_str = f"{params / 1e6:.2f} M"
    return flops_str, params_str

def measure_latency(model, input_tensor, num_runs=10, warmup=3):
    """
    Measures inference latency for a given model.
    
    Parameters:
        model: the PyTorch model to evaluate.
        input_tensor: a dummy input tensor.
        num_runs: number of timed runs.
        warmup: number of warmup runs to stabilize performance.
    
    Returns:
        A list of elapsed times (in seconds) for each run.
    """
    model.eval()
    with torch.no_grad():
        # Warmup runs
        for _ in range(warmup):
            _ = model(input_tensor)
    
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
    return timings

def measure_all_latencies_for_batch_sizes(fftnet_variants, vit_variants, device, batch_sizes):
    """
    Measures latency for all FFTNetViT and ViT variants over a range of batch sizes.
    
    Returns:
        A dictionary with structure:
        latencies["FFTNetViT Base"][batch_size] = average_latency_in_seconds
        latencies["ViT Base"][batch_size]       = average_latency_in_seconds
        ...
    """
    latencies = {}
    
    # Merge the variant dictionaries to process them in one loop
    all_variants = {}
    for k, v in fftnet_variants.items():
        all_variants[k] = ("FFTNetViT", v)
    for k, v in vit_variants.items():
        all_variants[k] = ("ViT", v)
    
    # For each variant, instantiate the model, measure latencies for all batch sizes.
    for key, (family, config) in all_variants.items():
        if family == "FFTNetViT":
            model_class = FFTNetViT
        else:
            model_class = ViT
        
        model = model_class(**config).to(device)
        latencies[key] = {}
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, 224, 224).to(device)
            times = measure_latency(model, dummy_input, num_runs=10, warmup=3)
            avg_time = sum(times) / len(times)
            latencies[key][bs] = avg_time  # seconds
    return latencies

def plot_combined_latency_vs_batch_size(fftnet_variants, vit_variants, latencies, batch_sizes):
    """
    Plots average inference latency (ms) versus batch size for corresponding
    FFTNetViT and standard ViT variants.
    """
    plt.figure(figsize=(8, 6))
    
    # Define line and marker styles for each family.
    fftnet_style = {'linestyle': '--', 'marker': 'o'}  # Dashed with circles.
    vit_style    = {'linestyle': '-',  'marker': 's'}  # Solid with squares.
    
    for variant_type in ["Base", "Large", "Huge"]:
        # Retrieve model keys
        fft_key = f"FFTNetViT {variant_type}"
        vit_key = f"ViT {variant_type}"
        
        if fft_key in fftnet_variants and vit_key in vit_variants:
            # Collect latencies for the FFTNet variant
            latencies_fft = [latencies[fft_key][bs] * 1000 for bs in batch_sizes]  # ms
            # Collect latencies for the ViT variant
            latencies_vit = [latencies[vit_key][bs] * 1000 for bs in batch_sizes]  # ms
            
            plt.plot(batch_sizes, latencies_fft, color=variant_colors[variant_type],
                     label=f"{variant_type} (FFTNetViT)", **fftnet_style)
            plt.plot(batch_sizes, latencies_vit, color=variant_colors[variant_type],
                     label=f"{variant_type} (ViT)", **vit_style)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Average Latency (ms)")
    plt.title("Latency vs Batch Size: FFTNetViT vs Standard ViT")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_latency_comparison.pdf")
    plt.close()
    logger.info("Saved combined latency plot: combined_latency_comparison.pdf")

def plot_throughput_vs_batch_size(fftnet_variants, vit_variants, latencies, batch_sizes):
    """
    Plots throughput (images/second) vs. batch size for FFTNetViT and ViT.
    Throughput is computed as (batch_size / average_latency_in_seconds).
    """
    plt.figure(figsize=(8, 6))
    
    fftnet_style = {'linestyle': '--', 'marker': 'o'}
    vit_style    = {'linestyle': '-',  'marker': 's'}
    
    for variant_type in ["Base", "Large", "Huge"]:
        fft_key = f"FFTNetViT {variant_type}"
        vit_key = f"ViT {variant_type}"
        
        if fft_key in fftnet_variants and vit_key in vit_variants:
            # FFTNet throughput
            thpt_fft = []
            # ViT throughput
            thpt_vit = []
            
            for bs in batch_sizes:
                # Throughput = batch_size / latency_in_seconds
                t_fft = batch_sizes[batch_sizes.index(bs)] / latencies[fft_key][bs]
                t_vit = batch_sizes[batch_sizes.index(bs)] / latencies[vit_key][bs]
                thpt_fft.append(t_fft)
                thpt_vit.append(t_vit)
            
            plt.plot(batch_sizes, thpt_fft, color=variant_colors[variant_type],
                     label=f"{variant_type} (FFTNetViT)", **fftnet_style)
            plt.plot(batch_sizes, thpt_vit, color=variant_colors[variant_type],
                     label=f"{variant_type} (ViT)", **vit_style)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/second)")
    plt.title("Throughput vs Batch Size: FFTNetViT vs Standard ViT")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_throughput_comparison.pdf")
    plt.close()
    logger.info("Saved throughput comparison plot: combined_throughput_comparison.pdf")

def plot_speedup_vs_batch_size(fftnet_variants, vit_variants, latencies, batch_sizes):
    """
    Plots speedup vs. batch size. Speedup is defined as:
        speedup = latency(ViT) / latency(FFTNetViT)
    Values > 1 indicate FFTNetViT is faster (lower latency).
    """
    plt.figure(figsize=(8, 6))
    
    for variant_type in ["Base", "Large", "Huge"]:
        fft_key = f"FFTNetViT {variant_type}"
        vit_key = f"ViT {variant_type}"
        
        if fft_key in fftnet_variants and vit_key in vit_variants:
            speedups = []
            for bs in batch_sizes:
                # speedup = (ViT latency) / (FFTNet latency)
                speedup_ratio = latencies[vit_key][bs] / latencies[fft_key][bs]
                speedups.append(speedup_ratio)
            
            plt.plot(batch_sizes, speedups, color=variant_colors[variant_type],
                     marker='o', linestyle='-', label=f"{variant_type}")
    
    plt.axhline(y=1.0, color='gray', linestyle='--', label='No Speedup')
    plt.xlabel("Batch Size")
    plt.ylabel("Speedup (ViT latency / FFTNet latency)")
    plt.title("Speedup vs Batch Size: ViT / FFTNetViT")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_speedup_comparison.pdf")
    plt.close()
    logger.info("Saved speedup comparison plot: combined_speedup_comparison.pdf")

def main():
    # Determine device: use GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FFTNetViT Variants configurations (all with local windowing).
    fftnet_variants = {
        'FFTNetViT Base': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 768,
            'depth': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 12,
            'adaptive_spectral': True,
            'local_window_size': 32,
            'use_local_branch': True,
            'use_global_hann': True
        },
        'FFTNetViT Large': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 1024,
            'depth': 24,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16,
            'adaptive_spectral': True,
            'local_window_size': 32,
            'use_local_branch': True,
            'use_global_hann': True
        },
        'FFTNetViT Huge': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 1280,
            'depth': 32,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16,
            'adaptive_spectral': True,
            'local_window_size': 32,
            'use_local_branch': True,
            'use_global_hann': True
        }
    }
    
    # Standard ViT Variants configurations (ViT-B/16, ViT-L/16, ViT-H/16).
    vit_variants = {
        'ViT Base': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 768,
            'depth': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 12
        },
        'ViT Large': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 1024,
            'depth': 24,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16
        },
        'ViT Huge': {
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 1000,
            'embed_dim': 1280,
            'depth': 32,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16
        }
    }
    
    # Evaluate and log metrics (FLOPs, Params, and latency for batch size = 1).
    logger.info("Evaluating FFTNetViT and Standard ViT Variants (Batch Size 1):")
    for key, config in {**fftnet_variants, **vit_variants}.items():
        family = "FFTNetViT" if "FFTNet" in key else "ViT"
        model = (FFTNetViT if family == "FFTNetViT" else ViT)(**config).to(device)
        flops, params = compute_metrics(model, input_res=(3, 224, 224))
        flops_str, params_str = format_metrics(flops, params)
        logger.info(f"{key} - FLOPs: {flops_str}, Params: {params_str}")
        
        # Latency for batch size 1
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        latencies = measure_latency(model, dummy_input, num_runs=10, warmup=3)
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"{key} - Average Latency (batch size 1): {avg_latency * 1000:.2f} ms")
        logger.info("-" * 50)
    
    # Now measure latencies for a range of batch sizes for all variants
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    latencies_dict = measure_all_latencies_for_batch_sizes(fftnet_variants, vit_variants, device, batch_sizes)
    
    # Generate combined latency vs batch size comparison plot.
    plot_combined_latency_vs_batch_size(fftnet_variants, vit_variants, latencies_dict, batch_sizes)
    
    # Generate throughput vs batch size plot.
    plot_throughput_vs_batch_size(fftnet_variants, vit_variants, latencies_dict, batch_sizes)
    
    # Generate speedup vs batch size plot.
    plot_speedup_vs_batch_size(fftnet_variants, vit_variants, latencies_dict, batch_sizes)

if __name__ == "__main__":
    main()
