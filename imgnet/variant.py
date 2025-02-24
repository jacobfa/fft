import torch
import logging
from ptflops import get_model_complexity_info
from fftnet_vit import FFTNetViT
from transformer import ViT

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

def main():
    # FFTNetViT Variants configurations
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
            'adaptive_spectral': True
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
            'adaptive_spectral': True
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
            'adaptive_spectral': True
        }
    }
    
    # Standard ViT Variants configurations (ViT-B/16, ViT-L/16, ViT-H/16)
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

    # Evaluate FFTNetViT variants
    logger.info("Evaluating FFTNetViT Variants:")
    for variant_name, config in fftnet_variants.items():
        logger.info(f"Evaluating variant: {variant_name}")
        model = FFTNetViT(**config)
        flops, params = compute_metrics(model, input_res=(3, 224, 224))
        flops_str, params_str = format_metrics(flops, params)
        logger.info(f"{variant_name} - FLOPs: {flops_str}, Params: {params_str}")
        logger.info("-" * 50)
    
    # Evaluate Standard ViT variants
    logger.info("Evaluating Standard ViT Variants:")
    for variant_name, config in vit_variants.items():
        logger.info(f"Evaluating variant: {variant_name}")
        model = ViT(**config)
        flops, params = compute_metrics(model, input_res=(3, 224, 224))
        flops_str, params_str = format_metrics(flops, params)
        logger.info(f"{variant_name} - FLOPs: {flops_str}, Params: {params_str}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
