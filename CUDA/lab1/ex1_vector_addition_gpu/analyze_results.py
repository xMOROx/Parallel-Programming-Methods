#!/usr/bin/env python3
"""
CUDA Vector Addition Performance Analysis
Generates plots from benchmark results showing GPU performance characteristics
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
import os


def load_all_csv_files(data_folder):
    """Load and combine all CSV files from the data folder"""
    csv_files = glob.glob(os.path.join(data_folder, "vector_addition_results_*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return None

    # Load and combine all CSV files
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined data before aggregation: {len(combined_df)} rows")

        # Group by N and Threads_Per_Block and calculate mean times
        # This handles multiple runs with the same configuration
        aggregated_df = (
            combined_df.groupby(["N", "Threads_Per_Block"])
            .agg(
                {
                    "Num_Blocks": "first",  # Should be the same for same N and Threads_Per_Block
                    "GPU_Time_ms": "mean",  # Average GPU time across runs
                    "CPU_Time_ms": "mean",  # Average CPU time across runs
                    "Results_Match": "all",  # All results should match
                }
            )
            .reset_index()
        )

        print(f"Aggregated data (mean times): {len(aggregated_df)} rows")
        return aggregated_df

    return None


def plot_gpu_time_vs_array_size(df):
    """Plot 1: GPU Time vs Array Size for different block sizes"""
    plt.figure(figsize=(12, 8))

    # Get unique block sizes
    block_sizes = sorted(df["Threads_Per_Block"].unique())
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    # Extend colors if needed
    while len(colors) < len(block_sizes):
        colors.extend(colors)

    for i, block_size in enumerate(block_sizes):
        block_data = df[df["Threads_Per_Block"] == block_size]
        block_data = block_data.sort_values("N")

        plt.loglog(
            block_data["N"],
            block_data["GPU_Time_ms"],
            "o-",
            label=f"{block_size} threads/block",
            color=colors[i],
            markersize=6,
            linewidth=2,
        )

    plt.xlabel("Array Size (N)", fontsize=12)
    plt.ylabel("GPU Time (ms)", fontsize=12)
    plt.title(
        "GPU Performance: Execution Time vs Array Size\nfor Different Block Sizes",
        fontsize=14,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_efficiency_vs_block_size(df):
    """Plot 3: GPU Efficiency vs Block Size for different array sizes"""
    plt.figure(figsize=(12, 8))

    # Calculate speedup for each row
    df["Speedup"] = df["CPU_Time_ms"] / df["GPU_Time_ms"]

    # Select representative array sizes (powers of 2)
    array_sizes = [2048, 8192, 32768, 131072, 524288, 2097152]
    array_sizes = [size for size in array_sizes if size in df["N"].values]

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    # Extend colors if needed
    while len(colors) < len(array_sizes):
        colors.extend(colors)

    for i, array_size in enumerate(array_sizes):
        size_data = df[df["N"] == array_size]
        size_data = size_data.sort_values("Threads_Per_Block")

        plt.plot(
            size_data["Threads_Per_Block"],
            size_data["Speedup"],
            "o-",
            label=f"N = {array_size:,}",
            color=colors[i],
            markersize=8,
            linewidth=2,
        )

    plt.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="No speedup")
    plt.xlabel("Threads per Block", fontsize=12)
    plt.ylabel("Speedup (CPU Time / GPU Time)", fontsize=12)
    plt.title(
        "GPU Efficiency: Speedup vs Block Size\nfor Different Array Sizes", fontsize=14
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks([64, 128, 256, 512, 1024])
    plt.tight_layout()

    return plt.gcf()


def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "data")
    plots_folder = os.path.join(script_dir, "plots")

    # Create plots folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)

    print("CUDA Vector Addition Performance Analysis")
    print("=" * 50)

    # Load data
    print(f"Loading data from: {data_folder}")
    df = load_all_csv_files(data_folder)

    if df is None:
        print("No data found. Exiting.")
        return

    print(f"\nData summary:")
    print(f"Array sizes: {sorted(df['N'].unique())}")
    print(f"Block sizes: {sorted(df['Threads_Per_Block'].unique())}")
    print(f"Total measurements: {len(df)}")

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: GPU Time vs Array Size
    print("1. GPU Time vs Array Size for different block sizes")
    fig1 = plot_gpu_time_vs_array_size(df)
    plot1_path = os.path.join(plots_folder, "gpu_time_vs_array_size.png")
    fig1.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"   Saved: {plot1_path}")

    # Plot 2: Efficiency vs Block Size
    print("2. GPU Efficiency vs Block Size")
    fig2 = plot_efficiency_vs_block_size(df)
    plot2_path = os.path.join(plots_folder, "efficiency_vs_block_size.png")
    fig2.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"   Saved: {plot2_path}")

    print(f"\nAll plots saved to: {plots_folder}")

    # Print some interesting statistics
    print("\nKey Findings:")
    print("-" * 30)

    avg_speedup = (df["CPU_Time_ms"] / df["GPU_Time_ms"]).mean()
    max_speedup = (df["CPU_Time_ms"] / df["GPU_Time_ms"]).max()
    best_config = df.loc[(df["CPU_Time_ms"] / df["GPU_Time_ms"]).idxmax()]

    print(f"Average GPU speedup: {avg_speedup:.2f}x")
    print(f"Maximum GPU speedup: {max_speedup:.2f}x")
    print(
        f"Best configuration: N={best_config['N']}, {best_config['Threads_Per_Block']} threads/block"
    )

    # Find optimal block size for largest array
    largest_n = df["N"].max()
    largest_n_data = df[df["N"] == largest_n]
    optimal_block = largest_n_data.loc[largest_n_data["GPU_Time_ms"].idxmin()]
    print(
        f"Optimal block size for N={largest_n}: {optimal_block['Threads_Per_Block']} threads/block"
    )


if __name__ == "__main__":
    main()
