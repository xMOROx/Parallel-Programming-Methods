import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

BASE_DIR = "."
NUM_RUNS = 10
RESULTS_SUBDIR = "results"
FILE_PATTERN = "results_variant2_*.csv"
OUTPUT_PLOT_TIME = "execution_time_vs_threads.png"
OUTPUT_PLOT_SPEEDUP = "strong_scaling_speedup.png"

COLUMN_MAP = {
    "GenTime": "time_randomization",
    "DistTime": "time_split",
    "SortTime": "time_sort",
    "CopyTime": "time_rewrite",
    "TotalTime": "time_global",
}
TIME_COLUMNS = list(COLUMN_MAP.keys())
PLOT_COLUMNS = list(COLUMN_MAP.values())

all_data = []

print(
    f"Looking for runs in subdirectories 'run1' to 'run{NUM_RUNS}' within '{os.path.abspath(BASE_DIR)}'"
)

for i in range(1, NUM_RUNS + 1):
    run_dir = os.path.join(BASE_DIR, f"run{i}")
    results_dir = os.path.join(run_dir, RESULTS_SUBDIR)

    if not os.path.isdir(results_dir):
        print(f"Warning: Directory not found, skipping: {results_dir}")
        continue

    search_path = os.path.join(results_dir, FILE_PATTERN)
    files = glob.glob(search_path)

    if not files:
        print(f"Warning: No files matching '{FILE_PATTERN}' found in {results_dir}")
        continue

    print(f"Processing run {i}: Found {len(files)} files in {results_dir}")

    for f in files:
        try:
            df_temp = pd.read_csv(f)
            df_temp = df_temp[["Threads"] + TIME_COLUMNS]
            df_temp["run"] = i
            all_data.append(df_temp)
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {f}")
        except Exception as e:
            print(f"Error reading or processing file {f}: {e}")

if not all_data:
    print("Error: No data loaded. Please check BASE_DIR, NUM_RUNS, and file paths.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

combined_df.rename(columns=COLUMN_MAP, inplace=True)


aggregated_stats = combined_df.groupby("Threads")[PLOT_COLUMNS].agg(["mean", "std"])

mean_times = aggregated_stats.loc[:, pd.IndexSlice[:, "mean"]]
mean_times.columns = PLOT_COLUMNS

std_times = aggregated_stats.loc[:, pd.IndexSlice[:, "std"]]
std_times.columns = PLOT_COLUMNS

std_times.fillna(0, inplace=True)

print("\n--- Mean Execution Times (seconds) ---")
print(mean_times)
print("\n--- Standard Deviation of Execution Times (seconds) ---")
print(std_times)

try:
    baseline_times = mean_times.loc[1]
except KeyError:
    print("Error: Data for 1 thread not found. Cannot calculate speedup.")
    speedup_df = None
else:
    speedup_df = baseline_times / mean_times

    print("\n--- Strong Scaling Speedup (T(1)/T(p)) ---")
    print(speedup_df)

markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
linestyles = ["--", "-.", ":", "--", "-.", ":", "--"]
colors = plt.cm.get_cmap("tab10", len(PLOT_COLUMNS))


plt.figure(figsize=(10, 6))
for i, col in enumerate(PLOT_COLUMNS):
    if col in mean_times.columns:
        plt.errorbar(
            mean_times.index,
            mean_times[col],
            yerr=std_times[col] if col in std_times.columns else None,
            label=col,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.5,
            capsize=3,
        )
        plt.fill_between(
            mean_times.index,
            mean_times[col] - std_times[col],
            mean_times[col] + std_times[col],
            alpha=0.2,
        )


plt.xlabel("Number of Threads")
plt.ylabel("Global Execution Time (seconds)")
plt.title("Global Execution Time vs. Number of Threads")
plt.legend(title="Time Components")
plt.grid(True)
plt.xticks(mean_times.index)
plt.yscale("linear")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_TIME)
print(f"\nSaved execution time plot to {OUTPUT_PLOT_TIME}")


if speedup_df is not None and not speedup_df.empty:
    plt.figure(figsize=(10, 6))
    threads = speedup_df.index

    for i, col in enumerate(PLOT_COLUMNS):
        if col in speedup_df.columns:
            plt.plot(
                threads,
                speedup_df[col],
                label=col,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
            )

    plt.plot(
        threads,
        threads,
        label="Ideal speedup",
        color="black",
        linestyle="-",
        linewidth=2,
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Strong Scaling Speedup")
    plt.title("Strong Scaling Speedup vs. Number of Threads")
    plt.legend(title="Components")
    plt.grid(True)
    plt.xticks(threads)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_SPEEDUP)
    print(f"Saved speedup plot to {OUTPUT_PLOT_SPEEDUP}")
else:
    print("Skipping speedup plot because data for 1 thread was missing.")
