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
    print("Determining baselines (best single-thread STRONG configuration)...")
    for data_size in df_agg["dataSize"].unique():
        baseline_run = df_agg[
            (df_agg["nCores"] == 1)
            & (df_agg["confId"] == "STRONG")
            & (df_agg["dataSize"] == data_size)
        ]
        if not baseline_run.empty:
            baselines[data_size] = baseline_run["time[s]"].iloc[0]
            print(f"  Baseline for {data_size}: {baselines[data_size]:.3f}s")
        else:
            fallback_baseline = df_agg[
                (df_agg["nCores"] == 1) & (df_agg["dataSize"] == data_size)
            ]
            if not fallback_baseline.empty:
                baselines[data_size] = fallback_baseline["time[s]"].min()
                print(
                    f"  Warning: STRONG 1-core baseline not found for {data_size}. Using best available 1-core: {baselines[data_size]:.3f}s"
                )
            else:
                print(
                    f"  Error: No 1-core baseline found for {data_size}. COST calculation and speedup plots will be affected."
                )
                baselines[data_size] = np.inf

    df_agg["baseline_time"] = df_agg["dataSize"].map(baselines)
    df_agg["speedup"] = df_agg["baseline_time"] / df_agg["time[s]"]

    for index, row in df_agg.iterrows():
        if row["nCores"] == 1 and row["time[s]"] == row["baseline_time"]:
            df_agg.loc[index, "speedup"] = 1.0
        elif row["nCores"] == 1 and row["time[s]"] != row["baseline_time"]:
            df_agg.loc[index, "speedup"] = row["baseline_time"] / row["time[s]"]

    cost_metrics = []

    for data_size in sorted(df_agg["dataSize"].unique()):
        subset_data_size = df_agg[df_agg["dataSize"] == data_size]
        current_baseline_time = baselines.get(data_size, np.inf)

        plt.figure(figsize=(10, 7))
        for conf_id in sorted(subset_data_size["confId"].unique()):
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

        if current_baseline_time != np.inf:
            plt.axhline(
                y=current_baseline_time,
                color="black",
                linestyle="--",
                label=f"Baseline (STRONG 1-core): {current_baseline_time:.2f}s",
            )

        unique_cores = sorted(subset_data_size["nCores"].unique())
        plt.xlabel("Number of Cores")
        plt.ylabel("Average Time (seconds)")
        plt.title(f"Average Execution Time for Data Size: {data_size}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        # plt.xscale("log", base=2)
        # plt.yscale("log")
        plt.xticks(unique_cores, labels=[str(c) for c in unique_cores])
        plt.minorticks_off()
        plot_filename = os.path.join(
            OUTPUT_DIR, f"avg_time_vs_cores_{data_size.replace('G', '')}G.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved time plot: {plot_filename}")

        plt.figure(figsize=(10, 7))
        for conf_id in sorted(subset_data_size["confId"].unique()):
            config_data = subset_data_size[
                subset_data_size["confId"] == conf_id
            ].sort_values("nCores")
            if current_baseline_time != np.inf:
                plt.plot(
                    config_data["nCores"],
                    config_data["speedup"],
                    marker="o",
                    linestyle="-",
                    label=f"System {conf_id}",
                )

        if current_baseline_time != np.inf:
            pass

        plt.xlabel("Number of Cores")
        plt.ylabel("Speedup (vs STRONG 1-core baseline)")
        plt.title(f"Speedup for Data Size: {data_size}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xticks(unique_cores, labels=[str(c) for c in unique_cores])
        plt.minorticks_off()
        plot_filename = os.path.join(
            OUTPUT_DIR, f"speedup_vs_cores_{data_size.replace('G', '')}G.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved speedup plot: {plot_filename}")

        print(f"\nCOST Metric for Data Size: {data_size}")
        if current_baseline_time == np.inf:
            print("  Cannot determine COST as baseline is not available.")
            for conf_id in sorted(subset_data_size["confId"].unique()):
                cost_metrics.append(
                    {
                        "dataSize": data_size,
                        "confId": conf_id,
                        "COST_nCores": "N/A (No Baseline)",
                    }
                )
            continue

        for conf_id in sorted(subset_data_size["confId"].unique()):
            config_data = subset_data_size[
                subset_data_size["confId"] == conf_id
            ].sort_values("nCores")

            outperforming_configs = config_data[
                config_data["time[s]"] < current_baseline_time
            ]

            cost = "Unbounded"
            if not outperforming_configs.empty:
                cost = outperforming_configs["nCores"].iloc[0]

            cost_metrics.append(
                {"dataSize": data_size, "confId": conf_id, "COST_nCores": cost}
            )
            print(f"  System {conf_id}: COST = {cost} cores")

    print("\n--- COST Metric Summary ---")
    cost_df = pd.DataFrame(cost_metrics)
    print(cost_df.to_string())
    cost_df.to_csv(os.path.join(OUTPUT_DIR, "cost_summary.csv"), index=False)
    print(f"Saved COST summary to: {os.path.join(OUTPUT_DIR, 'cost_summary.csv')}")


if __name__ == "__main__":
    try:
        main_df = pd.read_csv(CSV_FILE)
        print(f"Successfully loaded data from {CSV_FILE}")
        print("Initial data sample:")
        print(main_df.head())
        create_plots_and_calculate_cost(main_df)
        print(f"\nProcessing complete. Plots are saved in '{OUTPUT_DIR}' directory.")
    except FileNotFoundError:
        print(
            f"Error: The file {CSV_FILE} was not found. Please ensure it's in the same directory as the script."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
