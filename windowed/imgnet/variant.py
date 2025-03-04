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

def plot_combined_latency_vs_batch_size(fftnet_variants, vit_variants, device):
    """
    Plots average inference latency versus batch size for corresponding FFTNetViT and standard ViT variants.
    Each variant type (Base, Large, Huge) is compared directly by using the same color across families, while
    different line styles and markers differentiate the model family.
    
    Parameters:
        fftnet_variants: dictionary of FFTNetViT model configurations.
        vit_variants: dictionary of standard ViT model configurations.
        device: torch device.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    plt.figure(figsize=(8, 6))
    
    # Define line and marker styles for each family.
    fftnet_style = {'linestyle': '--', 'marker': 'o'}  # Dashed with circles.
    vit_style = {'linestyle': '-', 'marker': 's'}       # Solid with squares.
    
    # We'll assume each key in the variants dict ends with the variant type (Base, Large, Huge).
    for variant_type in ["Base", "Large", "Huge"]:
        # Retrieve configurations for the corresponding FFTNetViT and ViT models.
        fftnet_key = f"FFTNetViT {variant_type}"
        vit_key = f"ViT {variant_type}"
        
        if fftnet_key in fftnet_variants and vit_key in vit_variants:
            # FFTNetViT measurement.
            model_fft = FFTNetViT(**fftnet_variants[fftnet_key]).to(device)
            latencies_fft = []
            for bs in batch_sizes:
                dummy_input = torch.randn(bs, 3, 224, 224).to(device)
                times = measure_latency(model_fft, dummy_input, num_runs=10, warmup=3)
                avg_time = sum(times) / len(times) * 1000  # in milliseconds
                latencies_fft.append(avg_time)
            plt.plot(batch_sizes, latencies_fft, color=variant_colors[variant_type],
                     label=f"{variant_type} (FFTNetViT)", **fftnet_style)
            
            # Standard ViT measurement.
            model_vit = ViT(**vit_variants[vit_key]).to(device)
            latencies_vit = []
            for bs in batch_sizes:
                dummy_input = torch.randn(bs, 3, 224, 224).to(device)
                times = measure_latency(model_vit, dummy_input, num_runs=10, warmup=3)
                avg_time = sum(times) / len(times) * 1000  # in milliseconds
                latencies_vit.append(avg_time)
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

def main():
    # Determine device: use GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FFTNetViT Variants configurations.
    fftnet_variants = {
        'FFTNetViT Base': {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'num_classes': 1000,
            'embed_dim': 768,
            'depth': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 12,
            'adaptive_spectral': True
        },
        'FFTNetViT Large': {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'num_classes': 1000,
            'embed_dim': 1024,
            'depth': 24,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16,
            'adaptive_spectral': True
        },
        'FFTNetViT Huge': {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'num_classes': 1000,
            'embed_dim': 1280,
            'depth': 32,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_heads': 16,
            'adaptive_spectral': True
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
    
    # Evaluate and log metrics for batch size 1 for each model.
    logger.info("Evaluating FFTNetViT and Standard ViT Variants (Batch Size 1):")
    for key, config in {**fftnet_variants, **vit_variants}.items():
        family = "FFTNetViT" if "FFTNet" in key else "ViT"
        model = (FFTNetViT if family == "FFTNetViT" else ViT)(**config).to(device)
        flops, params = compute_metrics(model, input_res=(3, 224, 224))
        flops_str, params_str = format_metrics(flops, params)
        logger.info(f"{key} - FLOPs: {flops_str}, Params: {params_str}")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        latencies = measure_latency(model, dummy_input, num_runs=10, warmup=3)
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"{key} - Average Latency (batch size 1): {avg_latency * 1000:.2f} ms")
        logger.info("-" * 50)
    
    # Generate combined latency vs batch size comparison plot.
    plot_combined_latency_vs_batch_size(fftnet_variants, vit_variants, device)

if __name__ == "__main__":
    main()
