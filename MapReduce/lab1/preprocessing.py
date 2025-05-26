import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_DIR = "plots"
CSV_FILE = "results.csv"


def create_plots_and_calculate_cost(df):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df["nCores"] = pd.to_numeric(df["nCores"])
    df["time[s]"] = pd.to_numeric(df["time[s]"])

    df_agg = df.groupby(["nCores", "confId", "dataSize"], as_index=False)[
        "time[s]"
    ].mean()
    df_agg = df_agg.sort_values(by=["dataSize", "confId", "nCores"])

    baselines = {}
    print("Determining baselines (time at nCores=1 for each confId and dataSize)...")
    for group_key, group_df in df_agg.groupby(["dataSize", "confId"]):
        data_size, conf_id = group_key
        baseline_run = group_df[group_df["nCores"] == 1]
        if not baseline_run.empty:
            baseline_time = baseline_run["time[s]"].iloc[0]
            baselines[(data_size, conf_id)] = baseline_time
            print(f"  Baseline for {data_size}, {conf_id}: {baseline_time:.3f}s")
        else:
            print(
                f"  Warning: No 1-core baseline found for {data_size}, {conf_id}. Speedup and COST will be affected for this configuration."
            )
            baselines[(data_size, conf_id)] = np.inf

    def calculate_speedup(row):
        baseline_time = baselines.get((row["dataSize"], row["confId"]), np.inf)
        if baseline_time == np.inf or baseline_time == 0:
            return np.nan
        if row["time[s]"] == 0:
            return np.inf
        return baseline_time / row["time[s]"]

    df_agg["speedup"] = df_agg.apply(calculate_speedup, axis=1)

    for index, row in df_agg.iterrows():
        if row["nCores"] == 1:
            baseline_time = baselines.get((row["dataSize"], row["confId"]), np.inf)

            if row["time[s]"] == baseline_time and baseline_time != np.inf:
                df_agg.loc[index, "speedup"] = 1.0

            elif baseline_time != np.inf and baseline_time != 0 and row["time[s]"] != 0:
                df_agg.loc[index, "speedup"] = baseline_time / row["time[s]"]

    cost_metrics = []

    for data_size in sorted(df_agg["dataSize"].unique()):
        subset_data_size = df_agg[df_agg["dataSize"] == data_size].copy()

        plt.figure(figsize=(12, 7))

        unique_conf_ids_in_subset = sorted(subset_data_size["confId"].unique())

        for i, conf_id in enumerate(unique_conf_ids_in_subset):
            config_data = subset_data_size[
                subset_data_size["confId"] == conf_id
            ].sort_values("nCores")
            plt.plot(
                config_data["nCores"],
                config_data["time[s]"],
                marker="o",
                linestyle="-",
                label=f"System {conf_id}",
            )

            current_conf_baseline_time = baselines.get((data_size, conf_id), np.inf)

            if (
                current_conf_baseline_time != np.inf
                and 1 in config_data["nCores"].values
            ):
                label_marker = f"Baseline (1-core times)" if i == 0 else None
                plt.scatter(
                    [1],
                    [current_conf_baseline_time],
                    color="black",
                    marker="x",
                    s=100,
                    zorder=5,
                    label=label_marker,
                )

        unique_cores = sorted(subset_data_size["nCores"].unique())
        plt.xlabel("Number of Cores")
        plt.ylabel("Average Time (seconds)")
        plt.title(f"Average Execution Time for Data Size: {data_size}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if unique_cores:
            plt.xticks(unique_cores, labels=[str(c) for c in unique_cores])
        plt.minorticks_off()

        safe_data_size_fn = str(data_size).replace(".", "_").replace("/", "_")
        plot_filename = os.path.join(
            OUTPUT_DIR, f"avg_time_vs_cores_{safe_data_size_fn}.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved time plot: {plot_filename}")

        plt.figure(figsize=(12, 7))
        for conf_id in sorted(subset_data_size["confId"].unique()):
            config_data = subset_data_size[
                subset_data_size["confId"] == conf_id
            ].sort_values("nCores")

            if not config_data["speedup"].isnull().all():
                plt.plot(
                    config_data["nCores"],
                    config_data["speedup"],
                    marker="o",
                    linestyle="-",
                    label=f"System {conf_id} (vs own 1-core baseline)",
                )

        plt.xlabel("Number of Cores")
        plt.ylabel("Speedup (vs own 1-core baseline)")
        plt.title(f"Speedup for Data Size: {data_size}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        if unique_cores:
            plt.xticks(unique_cores, labels=[str(c) for c in unique_cores])
        plt.minorticks_off()
        plot_filename = os.path.join(
            OUTPUT_DIR, f"speedup_vs_cores_{safe_data_size_fn}.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved speedup plot: {plot_filename}")

        print(f"\nCOST Metric for Data Size: {data_size}")
        for conf_id in sorted(subset_data_size["confId"].unique()):
            current_baseline_time = baselines.get((data_size, conf_id), np.inf)

            if current_baseline_time == np.inf:
                print(f"  System {conf_id}: COST = N/A (No 1-core baseline)")
                cost_metrics.append(
                    {
                        "dataSize": data_size,
                        "confId": conf_id,
                        "COST_nCores": "N/A (No Baseline)",
                        "BaselineTime[s]": "N/A",
                    }
                )
                continue

            config_data_for_cost = subset_data_size[
                subset_data_size["confId"] == conf_id
            ].sort_values("nCores")

            outperforming_configs = config_data_for_cost[
                (config_data_for_cost["time[s]"] < current_baseline_time)
                & (config_data_for_cost["nCores"] > 1)
            ]

            cost = "Unbounded"
            if not outperforming_configs.empty:
                cost = outperforming_configs["nCores"].iloc[0]

            cost_metrics.append(
                {
                    "dataSize": data_size,
                    "confId": conf_id,
                    "COST_nCores": cost,
                    "BaselineTime[s]": f"{current_baseline_time:.3f}"
                    if current_baseline_time != np.inf
                    else "N/A",
                }
            )
            print(
                f"  System {conf_id}: Baseline Time = {current_baseline_time:.3f}s, COST = {cost} cores"
            )

    print("\n--- COST Metric Summary ---")
    cost_df = pd.DataFrame(cost_metrics)

    cost_df = cost_df[["dataSize", "confId", "BaselineTime[s]", "COST_nCores"]]
    print(cost_df.to_string())
    cost_df.to_csv(os.path.join(OUTPUT_DIR, "cost_summary.csv"), index=False)
    print(f"Saved COST summary to: {os.path.join(OUTPUT_DIR, 'cost_summary.csv')}")


if __name__ == "__main__":
    try:
        main_df = pd.read_csv(CSV_FILE)
        print(f"Successfully loaded data from {CSV_FILE}")
        print("Initial data sample:")
        print(main_df.head())

        main_df["dataSize"] = main_df["dataSize"].astype(str)
        main_df["confId"] = main_df["confId"].astype(str)

        create_plots_and_calculate_cost(main_df)
        print(f"\nProcessing complete. Plots are saved in '{OUTPUT_DIR}' directory.")
    except FileNotFoundError:
        print(
            f"Error: The file {CSV_FILE} was not found. Please ensure it's in the same directory as the script."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        print(traceback.format_exc())
