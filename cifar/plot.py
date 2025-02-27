#!/usr/bin/env python
"""
Plot validation metrics for FFTNetViT and Transformer models.

This script reads the validation metrics saved in:
  - FFTNetsViT_val_metrics.txt
  - Transformer_val_metrics.txt

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])


sns.set_theme(context="paper", style="white", font_scale=1.3)

# Define custom colors (lighter blue and red) and markers for the models.
line_colors = {"FFTNetsViT": "#6495ED",  # Cornflower Blue
               "Transformer": "#B22222"}  # Firebrick Red
markers = {"FFTNetsViT": "o", "Transformer": "s"}

def read_metrics(file_path):
    """Read the metrics CSV file and return a pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File '{file_path}' not found.")

def plot_metric(metric, ylabel, title, output_file):
    """Plot a given metric from both models and save to a PNG file."""
    # Read metrics from the two files.
    df_fftnet = read_metrics("FFTNetsViT_val_metrics.txt")
    df_transformer = read_metrics("Transformer_val_metrics.txt")
    
    plt.figure(figsize=(8, 6))
    
    # Plot FFTNetViT metrics.
    sns.lineplot(
        x="Epoch", 
        y=metric, 
        data=df_fftnet, 
        label="FFTNetsViT", 
        marker=markers["FFTNetsViT"],
        color=line_colors["FFTNetsViT"],
        markersize=5,
        linewidth=1.5
    )
    
    # Plot Transformer metrics.
    sns.lineplot(
        x="Epoch", 
        y=metric, 
        data=df_transformer, 
        label="Transformer", 
        marker=markers["Transformer"],
        color=line_colors["Transformer"],
        markersize=5,
        linewidth=1.5
    )
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved plot to '{output_file}'")

def main():
    # Plot validation loss.
    plot_metric(
        metric="Validation Loss",
        ylabel="Validation Loss",
        title="Validation Loss vs. Epoch",
        output_file="validation_loss.png"
    )
    
    # Plot validation accuracy.
    plot_metric(
        metric="Validation Accuracy",
        ylabel="Validation Accuracy (%)",
        title="Validation Accuracy vs. Epoch",
        output_file="validation_accuracy.png"
    )

if __name__ == "__main__":
    main()
