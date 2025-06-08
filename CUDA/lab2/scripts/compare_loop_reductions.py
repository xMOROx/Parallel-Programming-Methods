import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


base_data_path = "data/3/"


cg_files_pattern = os.path.join(base_data_path, "reduction_loop_cg_benchmark_*.csv")
cw_files_pattern = os.path.join(
    base_data_path,
    "reduction_loop_cw_benchmark_*.csv",
)


cg_files = sorted(glob.glob(cg_files_pattern))
cw_files = sorted(glob.glob(cw_files_pattern))


def process_files(file_list, label):
    """
    Reads a list of CSV files, concatenates them, and calculates mean and std of 'time' and 'bandwidth_GBs' grouped by 'size'.
    Assumes CSVs have 'size', 'time_ms', and 'bandwidth_GBs' columns.
    """
    all_data = []
    for f_path in file_list:
        try:
            df = pd.read_csv(f_path)

            if "time_ms" in df.columns:
                df = df.rename(columns={"time_ms": "time"})
            elif "Time (ms)" in df.columns:
                df = df.rename(columns={"Time (ms)": "time"})
            elif "time" not in df.columns:
                print(
                    f"Warning: 'time' or 'time_ms' column not found in {f_path}. Skipping this file for {label} time data."
                )

            if "size" not in df.columns and "Size" in df.columns:
                df = df.rename(columns={"Size": "size"})
            elif "size" not in df.columns:
                print(
                    f"Warning: 'size' column not found in {f_path}. Skipping this file for {label}."
                )
                continue

            if "bandwidth_GBs" not in df.columns:
                print(
                    f"Warning: 'bandwidth_GBs' column not found in {f_path}. Skipping bandwidth data for this file in {label}."
                )
                df["bandwidth_GBs"] = pd.NA

            cols_to_select = ["size"]
            if "time" in df.columns:
                cols_to_select.append("time")
            if "bandwidth_GBs" in df.columns:
                cols_to_select.append("bandwidth_GBs")

            all_data.append(df[cols_to_select])
        except Exception as e:
            print(f"Error reading or processing {f_path}: {e}")
            continue

    if not all_data:
        print(f"No data processed for {label}.")
        return pd.DataFrame(
            columns=["size", "mean_time", "std_time", "mean_bandwidth", "std_bandwidth"]
        )

    combined_df = pd.concat(all_data)

    combined_df["size"] = pd.to_numeric(combined_df["size"], errors="coerce")
    if "time" in combined_df.columns:
        combined_df["time"] = pd.to_numeric(combined_df["time"], errors="coerce")
    if "bandwidth_GBs" in combined_df.columns:
        combined_df["bandwidth_GBs"] = pd.to_numeric(
            combined_df["bandwidth_GBs"], errors="coerce"
        )

    combined_df.dropna(subset=["size"], inplace=True)

    agg_dict = {}
    if "time" in combined_df.columns:
        agg_dict["time"] = ["mean", "std"]
    if "bandwidth_GBs" in combined_df.columns:
        agg_dict["bandwidth_GBs"] = ["mean", "std"]

    if not agg_dict:
        print(f"No columns to aggregate for {label}.")
        return pd.DataFrame(
            columns=["size", "mean_time", "std_time", "mean_bandwidth", "std_bandwidth"]
        )

    summary_df = combined_df.groupby("size").agg(agg_dict).reset_index()

    new_cols = ["size"]
    if "time" in agg_dict:
        new_cols.extend(["mean_time", "std_time"])
    if "bandwidth_GBs" in agg_dict:
        new_cols.extend(["mean_bandwidth", "std_bandwidth"])
    summary_df.columns = new_cols

    expected_cols = ["size", "mean_time", "std_time", "mean_bandwidth", "std_bandwidth"]
    for col in expected_cols:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA

    return summary_df


cg_summary = process_files(cg_files, "Cooperative Groups")


cw_summary = process_files(cw_files, "Standard Loop Unrolling")


plt.figure(figsize=(10, 6))

if (
    not cg_summary.empty
    and "mean_time" in cg_summary.columns
    and cg_summary["mean_time"].notna().any()
):
    plt.plot(
        cg_summary["size"],
        cg_summary["mean_time"],
        marker="o",
        linestyle="-",
        label="Cooperative Groups Mean Time",
    )
    if "std_time" in cg_summary.columns and cg_summary["std_time"].notna().any():
        plt.fill_between(
            cg_summary["size"],
            cg_summary["mean_time"] - cg_summary["std_time"],
            cg_summary["mean_time"] + cg_summary["std_time"],
            alpha=0.2,
            label="Cooperative Groups Time STD",
        )

if (
    not cw_summary.empty
    and "mean_time" in cw_summary.columns
    and cw_summary["mean_time"].notna().any()
):
    plt.plot(
        cw_summary["size"],
        cw_summary["mean_time"],
        marker="x",
        linestyle="--",
        label="Standard Loop Unrolling Mean Time",
    )
    if "std_time" in cw_summary.columns and cw_summary["std_time"].notna().any():
        plt.fill_between(
            cw_summary["size"],
            cw_summary["mean_time"] - cw_summary["std_time"],
            cw_summary["mean_time"] + cw_summary["std_time"],
            alpha=0.2,
            label="Standard Loop Unrolling Time STD",
        )

plt.xlabel("Size")
plt.ylabel("Time (ms)")
plt.title("Comparison of Loop Unrolling Strategies: Time Performance")
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)


output_time_plot_path = "plots/loop_reduction_time_comparison_plot.png"
plt.savefig(output_time_plot_path)
print(f"Time plot saved to {output_time_plot_path}")


plt.figure(figsize=(10, 6))

if (
    not cg_summary.empty
    and "mean_bandwidth" in cg_summary.columns
    and cg_summary["mean_bandwidth"].notna().any()
):
    plt.plot(
        cg_summary["size"],
        cg_summary["mean_bandwidth"],
        marker="o",
        linestyle="-",
        label="Cooperative Groups Mean Bandwidth",
    )
    if (
        "std_bandwidth" in cg_summary.columns
        and cg_summary["std_bandwidth"].notna().any()
    ):
        plt.fill_between(
            cg_summary["size"],
            cg_summary["mean_bandwidth"] - cg_summary["std_bandwidth"],
            cg_summary["mean_bandwidth"] + cg_summary["std_bandwidth"],
            alpha=0.2,
            label="Cooperative Groups Bandwidth STD",
        )


if (
    not cw_summary.empty
    and "mean_bandwidth" in cw_summary.columns
    and cw_summary["mean_bandwidth"].notna().any()
):
    plt.plot(
        cw_summary["size"],
        cw_summary["mean_bandwidth"],
        marker="x",
        linestyle="--",
        label="Standard Loop Unrolling Mean Bandwidth",
    )
    if (
        "std_bandwidth" in cw_summary.columns
        and cw_summary["std_bandwidth"].notna().any()
    ):
        plt.fill_between(
            cw_summary["size"],
            cw_summary["mean_bandwidth"] - cw_summary["std_bandwidth"],
            cw_summary["mean_bandwidth"] + cw_summary["std_bandwidth"],
            alpha=0.2,
            label="Standard Loop Unrolling Bandwidth STD",
        )

plt.xlabel("Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Comparison of Loop Unrolling Strategies: Bandwidth Performance")
plt.legend()
plt.grid(True)
plt.xscale("log")

plt.tight_layout()


output_bandwidth_plot_path = "plots/loop_reduction_bandwidth_comparison_plot.png"
plt.savefig(output_bandwidth_plot_path)
print(f"Bandwidth plot saved to {output_bandwidth_plot_path}")


print("Script finished.")
