#!/usr/bin/env python3
"""
CUDA Matrix Transpose Performance Analysis
Generates plots from benchmark results showing GPU performance characteristics for matrix transpose
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def load_all_csv_files(data_folder):
    """Load and combine all CSV files from the data folder"""
    csv_files = glob.glob(os.path.join(data_folder, "matrix_transpose_results*.csv"))

    if not csv_files:
        print(f"No matrix transpose CSV files found in {data_folder}")
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

        # Group by N, Block_Size, and Kernel_Type and calculate mean times
        aggregated_df = (
            combined_df.groupby(["N", "Block_Size", "Kernel_Type"])
            .agg(
                {
                    "Num_Blocks": "first",  # Should be the same for same configuration
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


def plot_time_vs_matrix_size(df):
    """Plot 1: Execution Time vs Matrix Size for both kernels"""
    plt.figure(figsize=(14, 8))

    # Get unique block sizes
    block_sizes = sorted(df["Block_Size"].unique())
    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot for each block size
    for i, block_size in enumerate(block_sizes):
        block_data = df[df["Block_Size"] == block_size]

        # Separate naive and shared data
        naive_data = block_data[block_data["Kernel_Type"] == "Naive"].sort_values("N")
        shared_data = block_data[block_data["Kernel_Type"] == "Shared"].sort_values("N")

        # Plot naive kernel
        plt.loglog(
            naive_data["N"],
            naive_data["GPU_Time_ms"],
            "o-",
            label=f"Naive {block_size}x{block_size}",
            color=colors[i % len(colors)],
            linestyle="--",
            markersize=6,
            linewidth=2,
        )

        # Plot shared memory kernel
        plt.loglog(
            shared_data["N"],
            shared_data["GPU_Time_ms"],
            "s-",
            label=f"Shared {block_size}x{block_size}",
            color=colors[i % len(colors)],
            linestyle="-",
            markersize=6,
            linewidth=2,
        )

    plt.xlabel("Matrix Size (N×N)", fontsize=12)
    plt.ylabel("GPU Time (ms)", fontsize=12)
    plt.title(
        "Matrix Transpose Performance: Execution Time vs Matrix Size\nNaive vs Shared Memory Kernels",
        fontsize=14,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_time_vs_block_size(df):
    """Plot 2: Execution Time vs Block Size for different matrix sizes"""
    plt.figure(figsize=(14, 8))

    # Get unique matrix sizes
    matrix_sizes = sorted(df["N"].unique())
    colors = ["blue", "red", "green", "orange", "purple"]

    # Create subplots for naive and shared kernels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot naive kernel
    for i, matrix_size in enumerate(matrix_sizes):
        size_data = df[(df["N"] == matrix_size) & (df["Kernel_Type"] == "Naive")]
        size_data = size_data.sort_values("Block_Size")

        ax1.plot(
            size_data["Block_Size"],
            size_data["GPU_Time_ms"],
            "o-",
            label=f"{matrix_size}×{matrix_size}",
            color=colors[i % len(colors)],
            markersize=8,
            linewidth=2,
        )

    ax1.set_xlabel("Block Size", fontsize=12)
    ax1.set_ylabel("GPU Time (ms)", fontsize=12)
    ax1.set_title("Naive Kernel: Time vs Block Size", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot shared memory kernel
    for i, matrix_size in enumerate(matrix_sizes):
        size_data = df[(df["N"] == matrix_size) & (df["Kernel_Type"] == "Shared")]
        size_data = size_data.sort_values("Block_Size")

        ax2.plot(
            size_data["Block_Size"],
            size_data["GPU_Time_ms"],
            "s-",
            label=f"{matrix_size}×{matrix_size}",
            color=colors[i % len(colors)],
            markersize=8,
            linewidth=2,
        )

    ax2.set_xlabel("Block Size", fontsize=12)
    ax2.set_ylabel("GPU Time (ms)", fontsize=12)
    ax2.set_title("Shared Memory Kernel: Time vs Block Size", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    return fig


def plot_speedup_comparison(df):
    """Plot 3: Speedup comparison between Naive and Shared memory kernels"""
    plt.figure(figsize=(12, 8))

    # Calculate speedup for each configuration
    naive_df = df[df["Kernel_Type"] == "Naive"].copy()
    shared_df = df[df["Kernel_Type"] == "Shared"].copy()

    # Merge to compare same configurations
    merged = pd.merge(
        naive_df, shared_df, on=["N", "Block_Size"], suffixes=("_naive", "_shared")
    )

    merged["Speedup"] = merged["GPU_Time_ms_naive"] / merged["GPU_Time_ms_shared"]

    # Get unique block sizes
    block_sizes = sorted(merged["Block_Size"].unique())
    colors = ["blue", "red", "green", "orange", "purple"]

    for i, block_size in enumerate(block_sizes):
        block_data = merged[merged["Block_Size"] == block_size]
        block_data = block_data.sort_values("N")

        plt.semilogx(
            block_data["N"],
            block_data["Speedup"],
            "o-",
            label=f"Block {block_size}×{block_size}",
            color=colors[i % len(colors)],
            markersize=8,
            linewidth=2,
        )

    plt.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="No speedup")
    plt.xlabel("Matrix Size (N×N)", fontsize=12)
    plt.ylabel("Speedup (Naive Time / Shared Time)", fontsize=12)
    plt.title("Performance Improvement: Shared Memory vs Naive Kernel", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "data")
    plots_folder = os.path.join(script_dir, "plots")

    # Create plots folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)

    print("CUDA Matrix Transpose Performance Analysis")
    print("=" * 50)

    # Load data
    print(f"Loading data from: {data_folder}")
    df = load_all_csv_files(data_folder)

    if df is None:
        print("No data found. Exiting.")
        return

    print(f"\nData summary:")
    print(f"Matrix sizes: {sorted(df['N'].unique())}")
    print(f"Block sizes: {sorted(df['Block_Size'].unique())}")
    print(f"Kernel types: {sorted(df['Kernel_Type'].unique())}")
    print(f"Total measurements: {len(df)}")

    # Check result correctness
    incorrect_results = df[df["Results_Match"] == False]
    if len(incorrect_results) > 0:
        print(f"WARNING: {len(incorrect_results)} measurements with incorrect results!")
    else:
        print("All GPU results match CPU results ✓")

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Time vs Matrix Size
    print("1. Execution Time vs Matrix Size (both kernels)")
    fig1 = plot_time_vs_matrix_size(df)
    plot1_path = os.path.join(plots_folder, "matrix_transpose_time_vs_size.png")
    fig1.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"   Saved: {plot1_path}")

    # Plot 2: Time vs Block Size
    print("2. Execution Time vs Block Size")
    fig2 = plot_time_vs_block_size(df)
    plot2_path = os.path.join(plots_folder, "matrix_transpose_time_vs_block_size.png")
    fig2.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"   Saved: {plot2_path}")

    # Plot 3: Speedup Comparison
    print("3. Speedup: Shared Memory vs Naive Kernel")
    fig3 = plot_speedup_comparison(df)
    plot3_path = os.path.join(plots_folder, "matrix_transpose_speedup_comparison.png")
    fig3.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"   Saved: {plot3_path}")

    print(f"\nAll plots saved to: {plots_folder}")

    # Print some interesting statistics
    print("\nKey Findings:")
    print("-" * 30)

    # Calculate average performance improvement
    naive_df = df[df["Kernel_Type"] == "Naive"]
    shared_df = df[df["Kernel_Type"] == "Shared"]

    if len(naive_df) > 0 and len(shared_df) > 0:
        avg_naive_time = naive_df["GPU_Time_ms"].mean()
        avg_shared_time = shared_df["GPU_Time_ms"].mean()
        avg_improvement = avg_naive_time / avg_shared_time

        print(f"Average Naive kernel time: {avg_naive_time:.3f} ms")
        print(f"Average Shared memory kernel time: {avg_shared_time:.3f} ms")
        print(f"Average performance improvement: {avg_improvement:.2f}x")

    # Find best configurations
    fastest_naive = naive_df.loc[naive_df["GPU_Time_ms"].idxmin()]
    fastest_shared = shared_df.loc[shared_df["GPU_Time_ms"].idxmin()]

    print(
        f"Fastest Naive config: {fastest_naive['N']}×{fastest_naive['N']}, block {fastest_naive['Block_Size']}×{fastest_naive['Block_Size']}"
    )
    print(
        f"Fastest Shared config: {fastest_shared['N']}×{fastest_shared['N']}, block {fastest_shared['Block_Size']}×{fastest_shared['Block_Size']}"
    )


if __name__ == "__main__":
    main()
