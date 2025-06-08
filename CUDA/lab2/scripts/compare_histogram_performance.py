import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


base_data_path = "data/5/"


def process_histogram_files(file_list):
    """
    Reads a list of CSV files, concatenates them, and calculates mean and std of 'Time_ms'
    grouped by 'KernelName', 'ArraySize', and 'BinCount'.
    """
    all_data = []
    for f_path in file_list:
        try:
            df = pd.read_csv(f_path)

            if not {"KernelName", "ArraySize", "BinCount", "Time_ms"}.issubset(
                df.columns
            ):
                print(f"Warning: Missing required columns in {f_path}. Skipping.")
                continue
            all_data.append(df)
        except Exception as e:
            print(f"Error reading or processing {f_path}: {e}")
            continue

    if not all_data:
        print("No data processed.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df["ArraySize"] = pd.to_numeric(combined_df["ArraySize"], errors="coerce")
    combined_df["BinCount"] = pd.to_numeric(combined_df["BinCount"], errors="coerce")
    combined_df["Time_ms"] = pd.to_numeric(combined_df["Time_ms"], errors="coerce")

    combined_df.dropna(
        subset=["KernelName", "ArraySize", "BinCount", "Time_ms"], inplace=True
    )

    summary_df = (
        combined_df.groupby(["KernelName", "ArraySize", "BinCount"])
        .agg(mean_time_ms=("Time_ms", "mean"))
        .reset_index()
    )

    return summary_df


histogram_files_pattern = os.path.join(base_data_path, "histogram_benchmark_*.csv")
histogram_files = sorted(glob.glob(histogram_files_pattern))

if not histogram_files:
    print(f"No files found matching pattern: {histogram_files_pattern}")
else:
    print(f"Found files: {histogram_files}")


summary_data = process_histogram_files(histogram_files)

if summary_data.empty:
    print("No summary data to plot after processing.")
else:
    unique_array_sizes = summary_data["ArraySize"].unique()

    markers = ["o", "x", "^", "s", "P", "*", "D"]

    for array_size in unique_array_sizes:
        plt.figure(figsize=(10, 6))
        data_for_size = summary_data[summary_data["ArraySize"] == array_size]

        kernel_names = data_for_size["KernelName"].unique()

        for i, kernel_name in enumerate(kernel_names):
            kernel_data = data_for_size[
                data_for_size["KernelName"] == kernel_name
            ].sort_values(by="BinCount")
            if not kernel_data.empty:
                marker_style = markers[i % len(markers)]
                plt.plot(
                    kernel_data["BinCount"],
                    kernel_data["mean_time_ms"],
                    marker=marker_style,
                    linestyle="",
                    label=f"{kernel_name}",
                )

        plt.xlabel("BinCount")
        plt.ylabel("Time (ms)")
        plt.title(f"Histogram Performance for ArraySize: {int(array_size)}")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)

        safe_array_size_str = (
            str(int(array_size)) if float(array_size).is_integer() else str(array_size)
        )

        output_plot_path = os.path.join(
            "plots", f"histogram_performance_arraysize_{safe_array_size_str}.png"
        )
        plt.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")


print("Script finished.")
