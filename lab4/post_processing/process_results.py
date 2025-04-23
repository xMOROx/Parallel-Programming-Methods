import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import math


BASE_DIR = "."
NUM_RUNS = 10
RESULTS_SUBDIR = "results"

TARGET_CONFIGURATIONS = [
    {"BUCKETS_NUMBER": 2048},
    {"BUCKETS_NUMBER": 4096},
    {"BUCKETS_NUMBER": 8192},
    {"BUCKETS_NUMBER": 16384},
]


def create_config_id(params):
    return f"BN={params.get('BUCKETS_NUMBER', 'N/A')}"


OUTPUT_PLOT_TIME = "execution_time_subplots_multi_config_multi_run.png"
OUTPUT_PLOT_SPEEDUP = "strong_scaling_speedup_subplots_multi_config_multi_run.png"


COLUMN_MAP = {
    "GenTime": "time_randomization",
    "DistTime": "time_split",
    "SortTime": "time_sort",
    "CopyTime": "time_rewrite",
    "TotalTime": "time_global",
}
TIME_COLUMNS = list(COLUMN_MAP.keys())
PLOT_COLUMNS = list(COLUMN_MAP.values())


def parse_filename(filename):
    basename = os.path.basename(filename)
    match = re.match(
        r"results_variant2_"
        r"(?:CHUNK_(\d+)_)?"
        r"BUCKETS_NUMBER_(\d+)_"
        r"(?:CAPACITY_MULTIPLIER_([\d.]+)p([\d.]+)_)"
        r"(\d+)\.csv$",
        basename,
    )
    if match:
        params = {}
        chunk, buckets, cap_mult_int, cap_mult_dec, threads_fn = match.groups()
        if chunk:
            params["CHUNK"] = int(chunk)
        if buckets:
            params["BUCKETS_NUMBER"] = int(buckets)
        if cap_mult_int is not None and cap_mult_dec is not None:
            params["CAPACITY_MULTIPLIER"] = float(f"{cap_mult_int}.{cap_mult_dec}")

        return params
    else:
        return None


all_data = []
print(
    f"Looking for runs in subdirectories 'run1' to 'run{NUM_RUNS}' within '{os.path.abspath(BASE_DIR)}'"
)
processed_files_count = 0
skipped_files_count = 0

for i in range(1, NUM_RUNS + 1):
    run_dir = os.path.join(BASE_DIR, f"run{i}")
    results_dir = os.path.join(run_dir, RESULTS_SUBDIR)
    if not os.path.isdir(results_dir):
        continue
    search_path = os.path.join(results_dir, "results_variant2_*.csv")
    files_in_run = glob.glob(search_path)
    if not files_in_run:
        continue

    for f in files_in_run:
        file_params = parse_filename(f)
        if not file_params:
            skipped_files_count += 1
            continue

        is_target = False
        config_id = None
        for target_config in TARGET_CONFIGURATIONS:
            if all(file_params.get(k) == v for k, v in target_config.items()):
                is_target = True
                config_id = create_config_id(target_config)
                break
        if not is_target:
            skipped_files_count += 1
            continue

        try:
            df_temp = pd.read_csv(f)
            required_cols = ["Threads"] + TIME_COLUMNS
            if not all(col in df_temp.columns for col in required_cols):
                skipped_files_count += 1
                continue

            df_temp = df_temp[required_cols]
            df_temp["config_id"] = config_id
            df_temp["run"] = i
            all_data.append(df_temp)
            processed_files_count += 1
        except Exception as e:
            print(f"Error reading/processing file {f}: {e}")
            skipped_files_count += 1

if not all_data:
    print("Error: No data loaded for target configurations.")
    exit()

print(f"\nLoaded data from {processed_files_count} files.")
if skipped_files_count > 0:
    print(f"Skipped {skipped_files_count} other files.")

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.rename(columns=COLUMN_MAP, inplace=True)


aggregated_stats = combined_df.groupby(["config_id", "Threads"])[PLOT_COLUMNS].agg(
    ["mean", "std"]
)
mean_times = aggregated_stats.loc[:, pd.IndexSlice[:, "mean"]]
mean_times.columns = PLOT_COLUMNS
std_times = aggregated_stats.loc[:, pd.IndexSlice[:, "std"]]
std_times.columns = PLOT_COLUMNS
std_times.fillna(0, inplace=True)

print("\n--- Mean Execution Times (seconds, averaged over runs) ---")


speedup_dfs = {}
config_ids = mean_times.index.get_level_values("config_id").unique()
print("\n--- Calculating Strong Scaling Speedup (T(1)/T(p)) ---")
for config_id in config_ids:
    config_mean_times = mean_times.loc[config_id]
    try:
        baseline_times = config_mean_times.loc[1]
        speedup_df = baseline_times / config_mean_times
        speedup_dfs[config_id] = speedup_df

    except KeyError:
        print(
            f"Warning: Mean data for 1 thread not found for config '{config_id}'. Cannot calculate speedup."
        )
        speedup_dfs[config_id] = None


markers = ["o", "s", "^", "D", "v", "p", "*", "h", "+", "x"]
linestyles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 10)),
    (0, (1, 1)),
    (0, (5, 1)),
]
colors = plt.cm.get_cmap("tab10", len(config_ids))
num_plots = len(PLOT_COLUMNS)

ncols = 2
nrows = math.ceil(num_plots / ncols)

unique_threads = sorted(list(combined_df["Threads"].unique()))
int_ticks = all(t == int(t) for t in unique_threads)


print(f"\nGenerating execution time plot: {OUTPUT_PLOT_TIME}")
fig_time, axs_time = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7 * ncols, 5 * nrows),
    sharex=True,
)
axs_time = axs_time.flatten()

for plot_idx, col_to_plot in enumerate(PLOT_COLUMNS):
    ax = axs_time[plot_idx]

    for i, config_id in enumerate(config_ids):
        if config_id not in mean_times.index:
            continue
        config_mean = mean_times.loc[config_id]
        config_std = std_times.loc[config_id]
        threads = config_mean.index

        if col_to_plot in config_mean.columns:
            ax.errorbar(
                threads,
                config_mean[col_to_plot],
                yerr=config_std[col_to_plot]
                if col_to_plot in config_std.columns
                else None,
                label=f"{config_id}",
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                color=colors(i),
                linewidth=1.2,
                capsize=2,
                alpha=0.9,
            )

            ax.fill_between(
                threads,
                (config_mean[col_to_plot] - config_std[col_to_plot]).clip(lower=1e-9),
                config_mean[col_to_plot] + config_std[col_to_plot],
                color=colors(i),
                alpha=0.10,
            )

    ax.set_title(col_to_plot.replace("_", " ").title())
    ax.set_ylabel("Mean Time (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if int_ticks:
        ax.set_xticks(unique_threads)
    ax.tick_params(axis="x", rotation=45)

    ax.legend(title="Configuration", fontsize="small")


for ax_idx in range(ncols * (nrows - 1), ncols * nrows):
    if ax_idx < len(axs_time):
        axs_time[ax_idx].set_xlabel("Number of Threads")


for ax_idx in range(num_plots, nrows * ncols):
    axs_time[ax_idx].axis("off")

fig_time.suptitle("Execution Time vs. Threads (Averaged over Runs)", fontsize=16)
fig_time.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_PLOT_TIME)
print(f"Saved execution time plot to {OUTPUT_PLOT_TIME}")
plt.close(fig_time)


print(f"\nGenerating speedup plot: {OUTPUT_PLOT_SPEEDUP}")
fig_speedup, axs_speedup = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7 * ncols, 5 * nrows),
    sharex=True,
    sharey=True,
)
axs_speedup = axs_speedup.flatten()
any_speedup_plotted = False

for plot_idx, col_to_plot in enumerate(PLOT_COLUMNS):
    ax = axs_speedup[plot_idx]
    plot_ideal_on_ax = True

    for i, config_id in enumerate(config_ids):
        speedup_df = speedup_dfs.get(config_id)

        if (
            speedup_df is not None
            and not speedup_df.empty
            and col_to_plot in speedup_df.columns
        ):
            threads = speedup_df.index
            ax.plot(
                threads,
                speedup_df[col_to_plot],
                label=f"{config_id}",
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                color=colors(i),
                linewidth=1.2,
            )
            any_speedup_plotted = True

            if plot_ideal_on_ax and len(threads) > 0:
                ax.plot(
                    unique_threads,
                    unique_threads,
                    label="Ideal",
                    color="black",
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.8,
                )
                plot_ideal_on_ax = False

    ax.set_title(col_to_plot.replace("_", " ").title())
    ax.set_ylabel("Speedup")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if int_ticks:
        ax.set_xticks(unique_threads)
    ax.tick_params(axis="x", rotation=45)

    if not plot_ideal_on_ax:
        ax.legend(title="Configuration", fontsize="small")


for ax_idx in range(ncols * (nrows - 1), ncols * nrows):
    if ax_idx < len(axs_speedup):
        axs_speedup[ax_idx].set_xlabel("Number of Threads")


for ax_idx in range(num_plots, nrows * ncols):
    axs_speedup[ax_idx].axis("off")

fig_speedup.suptitle(
    "Strong Scaling Speedup vs. Threads (Based on Mean Times)", fontsize=16
)
fig_speedup.tight_layout(rect=[0, 0.03, 1, 0.95])

if any_speedup_plotted:
    plt.savefig(OUTPUT_PLOT_SPEEDUP)
    print(f"Saved speedup plot to {OUTPUT_PLOT_SPEEDUP}")
else:
    print("Skipping speedup plot generation as no valid speedup data was found.")
plt.close(fig_speedup)

print("\nScript finished.")
