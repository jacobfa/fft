import torch
import logging
from ptflops import get_model_complexity_info
from fftnet_vit import FFTNetViT

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
    # Use ptflops to compute FLOPs and parameter counts.
    flops, params = get_model_complexity_info(
        model, input_res, as_strings=True, print_per_layer_stat=False
    )
    return flops, params

def main():
    # Define model configurations for Base, Large, and Huge variants.
    variants = {
        'Base': {
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
        'Large': {
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
        'Huge': {
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

    for variant_name, config in variants.items():
        logger.info(f"Evaluating variant: {variant_name}")
        model = FFTNetViT(**config)
        flops, params = compute_metrics(model, input_res=(3, 224, 224))
        logger.info(f"{variant_name} - FLOPs: {flops}, Params: {params}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
