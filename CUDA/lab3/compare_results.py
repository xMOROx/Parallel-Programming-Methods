import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def load_and_process_data(data_dir):
    """Load all CSV files from data directory and process them"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded {file}: {len(df)} rows")
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not all_data:
        print("No CSV files found or loaded successfully")
        return None

    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data


def calculate_statistics(data):
    """Calculate mean and standard deviation for each method and matrix size"""
    stats = (
        data.groupby(["Method", "Matrix_Size"])
        .agg(
            {
                "Time_ms": ["mean", "std"],
                "Bandwidth_GB_s": ["mean", "std"],
                "Error": ["mean", "std"],
            }
        )
        .reset_index()
    )

    stats.columns = [
        "Method",
        "Matrix_Size",
        "Time_mean",
        "Time_std",
        "Bandwidth_mean",
        "Bandwidth_std",
        "Error_mean",
        "Error_std",
    ]

    stats = stats.fillna(0)

    return stats


def create_time_plot(stats):
    """Create line plot for execution time with error regions"""
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    simplest_data = stats[stats["Method"] == "simplest"].sort_values("Matrix_Size")
    cuda_data = stats[stats["Method"] == "cuda_samples"].sort_values("Matrix_Size")

    ax.errorbar(
        simplest_data["Matrix_Size"],
        simplest_data["Time_mean"],
        yerr=simplest_data["Time_std"],
        label="Simplest",
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        cuda_data["Matrix_Size"],
        cuda_data["Time_mean"],
        yerr=cuda_data["Time_std"],
        label="CUDA Samples",
        marker="s",
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    ax.fill_between(
        simplest_data["Matrix_Size"],
        simplest_data["Time_mean"] - simplest_data["Time_std"],
        simplest_data["Time_mean"] + simplest_data["Time_std"],
        alpha=0.2,
    )
    ax.fill_between(
        cuda_data["Matrix_Size"],
        cuda_data["Time_mean"] - cuda_data["Time_std"],
        cuda_data["Time_mean"] + cuda_data["Time_std"],
        alpha=0.2,
    )

    
    for _, row in simplest_data.iterrows():
        ax.annotate(
            f'{row["Time_mean"]:.3f}ms',
            (row["Matrix_Size"], row["Time_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    for _, row in cuda_data.iterrows():
        ax.annotate(
            f'{row["Time_mean"]:.3f}ms',
            (row["Matrix_Size"], row["Time_mean"]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title("Execution Time vs Matrix Size", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    time_plot_path = os.path.join(plots_dir, "execution_time_comparison.png")
    plt.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved time plot to: {time_plot_path}")
    plt.show()


def create_bandwidth_plot(stats):
    """Create line plot for bandwidth with error regions"""
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    simplest_data = stats[stats["Method"] == "simplest"].sort_values("Matrix_Size")
    cuda_data = stats[stats["Method"] == "cuda_samples"].sort_values("Matrix_Size")

    ax.errorbar(
        simplest_data["Matrix_Size"],
        simplest_data["Bandwidth_mean"],
        yerr=simplest_data["Bandwidth_std"],
        label="Simplest",
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.errorbar(
        cuda_data["Matrix_Size"],
        cuda_data["Bandwidth_mean"],
        yerr=cuda_data["Bandwidth_std"],
        label="CUDA Samples",
        marker="s",
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    ax.fill_between(
        simplest_data["Matrix_Size"],
        simplest_data["Bandwidth_mean"] - simplest_data["Bandwidth_std"],
        simplest_data["Bandwidth_mean"] + simplest_data["Bandwidth_std"],
        alpha=0.2,
    )
    ax.fill_between(
        cuda_data["Matrix_Size"],
        cuda_data["Bandwidth_mean"] - cuda_data["Bandwidth_std"],
        cuda_data["Bandwidth_mean"] + cuda_data["Bandwidth_std"],
        alpha=0.2,
    )

    
    for _, row in simplest_data.iterrows():
        ax.annotate(
            f'{row["Bandwidth_mean"]:.1f}',
            (row["Matrix_Size"], row["Bandwidth_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    for _, row in cuda_data.iterrows():
        ax.annotate(
            f'{row["Bandwidth_mean"]:.1f}',
            (row["Matrix_Size"], row["Bandwidth_mean"]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_ylabel("Memory Bandwidth (GB/s)", fontsize=12)
    ax.set_title("Memory Bandwidth vs Matrix Size", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    bandwidth_plot_path = os.path.join(plots_dir, "bandwidth_comparison.png")
    plt.savefig(bandwidth_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved bandwidth plot to: {bandwidth_plot_path}")
    plt.show()


def print_statistics_table(stats):
    """Print a formatted table of statistics"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON STATISTICS")
    print("=" * 80)

    for method in stats["Method"].unique():
        method_data = stats[stats["Method"] == method]
        print(f"\n{method.upper()} METHOD:")
        print("-" * 60)
        print(f"{'Size':<10} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'Error':<15}")
        print(f"{'':.<10} {'Mean ± Std':<15} {'Mean ± Std':<20} {'Mean ± Std':<15}")
        print("-" * 60)

        for _, row in method_data.iterrows():
            size = f"{int(row['Matrix_Size'])}x{int(row['Matrix_Size'])}"
            time_str = f"{row['Time_mean']:.2f} ± {row['Time_std']:.2f}"
            bandwidth_str = f"{row['Bandwidth_mean']:.2f} ± {row['Bandwidth_std']:.2f}"
            error_str = f"{row['Error_mean']:.2e} ± {row['Error_std']:.2e}"

            print(f"{size:<10} {time_str:<15} {bandwidth_str:<20} {error_str:<15}")


def calculate_speedup(stats):
    """Calculate speedup of CUDA samples vs simplest"""
    simplest_data = stats[stats["Method"] == "simplest"].set_index("Matrix_Size")
    cuda_data = stats[stats["Method"] == "cuda_samples"].set_index("Matrix_Size")

    print("\n" + "=" * 50)
    print("SPEEDUP ANALYSIS (CUDA Samples vs Simplest)")
    print("=" * 50)
    print(f"{'Matrix Size':<15} {'Speedup':<10} {'Bandwidth Ratio':<15}")
    print("-" * 40)

    for size in simplest_data.index:
        if size in cuda_data.index:
            speedup = (
                simplest_data.loc[size, "Time_mean"] / cuda_data.loc[size, "Time_mean"]
            )
            bandwidth_ratio = (
                cuda_data.loc[size, "Bandwidth_mean"]
                / simplest_data.loc[size, "Bandwidth_mean"]
            )

            size_str = f"{int(size)}x{int(size)}"
            print(f"{size_str:<15} {speedup:.2f}x{'':<5} {bandwidth_ratio:.2f}x")


def main():
    data_dir = "data"
    plots_dir = "plots"

    print("Loading performance data...")
    data = load_and_process_data(data_dir)

    if data is None:
        print("No data loaded. Please check the data directory and CSV files.")
        return

    print(f"Total records loaded: {len(data)}")
    print(f"Methods found: {data['Method'].unique()}")
    print(f"Matrix sizes: {sorted(data['Matrix_Size'].unique())}")

    print("\nCalculating statistics...")
    stats = calculate_statistics(data)

    print_statistics_table(stats)

    calculate_speedup(stats)

    os.makedirs(plots_dir, exist_ok=True)

    print("\nCreating time comparison plot...")
    create_time_plot(stats)

    print("\nCreating bandwidth comparison plot...")
    create_bandwidth_plot(stats)

    print(f"\nAnalysis complete! Plots saved to: {plots_dir}")
    print("- execution_time_comparison.png")
    print("- bandwidth_comparison.png")


if __name__ == "__main__":
    main()
