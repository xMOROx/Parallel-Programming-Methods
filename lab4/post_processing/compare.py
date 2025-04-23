import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import os
from pathlib import Path


variant3_csv = "variant_3_merged.csv"
variant4_csv = "variant_4_merged.csv"
output_plot_directory = "scaling_plots_by_bucket"
output_plot_filename_prefix = "strong_scaling_speedup_"
cols_to_use = ["n", "p", "time_global", "n_buckets", "run"]
preferred_styles = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]
fallback_style = "default"


def get_available_style(styles_to_try, default_style):
    """Tries to find an available style from a list, otherwise returns default."""
    available = mplstyle.available
    for style in styles_to_try:
        if style in available:
            print(f"Using Matplotlib style: {style}")
            return style
    print(f"Warning: Preferred styles not found. Using fallback style: {default_style}")
    return default_style


plot_style_to_use = get_available_style(preferred_styles, fallback_style)


def load_and_process_data(filepath, variant_name):
    """Loads CSV, adds variant name, converts types, and validates."""
    print(f"\n--- Processing {filepath} ---")
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None
    try:
        print(f"Attempting to read CSV: {filepath}")
        try:
            df_check = pd.read_csv(filepath, nrows=1)
            print(f"Header check. Found: {list(df_check.columns)}")
            if not all(col in df_check.columns for col in cols_to_use):
                print(f"Error: Missing one or more required columns in {filepath}.")
                print(f"Required: {cols_to_use}")
                print(f"Found: {list(df_check.columns)}")
                return None
        except Exception as e:
            print(f"Error during initial header check of {filepath}: {e}")
            return None

        df = pd.read_csv(filepath, usecols=cols_to_use)
        initial_rows = len(df)
        print(f"Loaded {initial_rows} rows using columns: {cols_to_use}")
        if initial_rows == 0:
            print(f"Warning: Loaded 0 rows after selecting columns for {filepath}.")
            return df

        df["variant"] = variant_name
        print("Converting data types...")

        numeric_cols = ["n", "p", "time_global", "n_buckets", "run"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        nan_counts_after_coerce = df[numeric_cols].isna().sum()
        if nan_counts_after_coerce.sum() > 0:
            print(
                f"NaN counts after to_numeric(errors='coerce'):\n{nan_counts_after_coerce}"
            )

        rows_before_dropna = len(df)
        required_for_calc_and_grouping = [
            "p",
            "time_global",
            "n_buckets",
            "run",
            "variant",
        ]
        df.dropna(subset=required_for_calc_and_grouping, inplace=True)
        rows_after_dropna = len(df)
        print(
            f"Rows before dropna (on essential cols): {rows_before_dropna}, Rows after dropna: {rows_after_dropna}"
        )

        if rows_after_dropna > 0:
            try:
                df[["p", "n_buckets", "run"]] = df[["p", "n_buckets", "run"]].astype(
                    int
                )
                if "n" in df.columns:
                    if df["n"].notna().all():
                        df["n"] = df["n"].astype(np.int64)
                    else:
                        print(
                            "Warning: NaN values found in 'n' column after coerce, cannot convert to int64. Keeping as float."
                        )
            except Exception as e:
                print(
                    f"Warning: Could not convert columns to int/int64 after dropna: {e}"
                )

        final_rows = len(df)
        print(
            f"Successfully processed. Final row count for {variant_name}: {final_rows}"
        )
        if final_rows == 0 and initial_rows > 0:
            print(
                f"Warning: All rows were removed during processing for {filepath}. Check NaN counts and original data."
            )
        return df

    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None


df3 = load_and_process_data(variant3_csv, "variant_3")
df4 = load_and_process_data(variant4_csv, "variant_4")


valid_dfs = []
if df3 is not None and not df3.empty:
    valid_dfs.append(df3)
    print(f"\nUsing {len(df3)} rows from variant_3.")
else:
    print(f"\nWarning: No valid data loaded for variant_3 from {variant3_csv}.")

if df4 is not None and not df4.empty:
    valid_dfs.append(df4)
    print(f"Using {len(df4)} rows from variant_4.")
else:
    print(f"Warning: No valid data loaded for variant_4 from {variant4_csv}.")

if not valid_dfs:
    print("Error: No valid data loaded from any variant file. Cannot proceed.")
    exit()

df_combined = pd.concat(valid_dfs, ignore_index=True)
print(f"Total rows combined: {len(df_combined)}")


print("\nCalculating speedup (relative to p=1)...")

problem_size_varies = False
problem_sizes = []
valid_n_values = []
if "n" in df_combined.columns:
    valid_n_values = df_combined["n"].dropna().unique()
    if len(valid_n_values) > 1:
        problem_size_varies = True
        problem_sizes = sorted(list(valid_n_values))
        print(
            f"Warning: Multiple problem sizes (n) found: {problem_sizes}. Speedup calculation assumes strong scaling (fixed n)."
        )
        baseline_group_cols = ["variant", "n_buckets", "run", "n"]
        print("Calculating baseline T(p=1) separately for each problem size 'n'.")
    elif len(valid_n_values) == 1:
        problem_sizes = [valid_n_values[0]]
        print(
            f"Single problem size (n) found: {problem_sizes[0]}. Proceeding with standard strong scaling."
        )
        baseline_group_cols = ["variant", "n_buckets", "run"]
    else:
        print(
            "Warning: No valid problem size 'n' values found in data after dropping NaNs."
        )
        baseline_group_cols = ["variant", "n_buckets", "run"]
else:
    print(
        "Column 'n' (problem size) not found. Proceeding without considering problem size variation."
    )
    baseline_group_cols = ["variant", "n_buckets", "run"]


baseline = df_combined[df_combined["p"] == 1].copy()

if baseline.empty:
    print(
        "Error: No data found for p=1 (single thread) in the combined data. Cannot calculate speedup."
    )
    exit()
print(f"Found {len(baseline)} baseline (p=1) rows.")

if problem_size_varies and "n" not in baseline.columns:
    print("Error: Problem size 'n' varies, but 'n' column missing from baseline data.")
    exit()


baseline = baseline.rename(columns={"time_global": "time_baseline"})

select_cols_for_baseline = baseline_group_cols + ["time_baseline"]

baseline = baseline[select_cols_for_baseline]


duplicates = baseline[baseline.duplicated(subset=baseline_group_cols, keep=False)]
if not duplicates.empty:
    print(
        f"Warning: Duplicate baseline (p=1) entries found for the same group ({', '.join(baseline_group_cols)}). Keeping first occurrence."
    )
    baseline = baseline.drop_duplicates(subset=baseline_group_cols, keep="first")


df_merged = pd.merge(df_combined, baseline, on=baseline_group_cols, how="left")

missing_baseline_count = df_merged["time_baseline"].isna().sum()
if missing_baseline_count > 0:
    print(
        f"Warning: {missing_baseline_count} rows could not find a matching baseline T(p=1). Excluding these rows."
    )
    df_merged.dropna(subset=["time_baseline"], inplace=True)
    if df_merged.empty:
        print("Error: After removing rows without baselines, no data remains.")
        exit()


df_merged["speedup"] = df_merged["time_baseline"] / df_merged["time_global"]


aggregation_group_cols = ["variant", "n_buckets", "p"]
if problem_size_varies:
    aggregation_group_cols.append("n")

results = (
    df_merged.groupby(aggregation_group_cols)["speedup"]
    .agg(["mean", "std"])
    .reset_index()
)
results["std"].fillna(0, inplace=True)
print("Speedup calculation complete.")


results_to_plot_list = []
target_n_values = []

if problem_size_varies:
    target_n = problem_sizes[0]
    target_n_values.append(target_n)
    print(
        f"\nWarning: Multiple problem sizes exist. Generating plots ONLY for n={target_n}."
    )
    results_to_plot = results[results["n"] == target_n].copy()
    if results_to_plot.empty:
        print(
            f"Error: No aggregated results found for the selected problem size n={target_n}. Cannot plot."
        )
        exit()
    results_to_plot_list.append(results_to_plot)
else:
    results_to_plot_list.append(results)
    target_n_values.append("N/A" if not problem_sizes else problem_sizes[0])

output_dir = Path(output_plot_directory)
output_dir.mkdir(parents=True, exist_ok=True)

for i, results_to_plot in enumerate(results_to_plot_list):
    target_n = target_n_values[i]
    print(f"\nSaving plots for n={target_n} to directory: {output_dir.resolve()}")

    all_thread_counts = sorted(results_to_plot["p"].unique())
    all_bucket_counts = sorted(results_to_plot["n_buckets"].unique())

    if not all_thread_counts:
        print(
            f"Error: No thread counts ('p') found in the results for n={target_n}. Skipping plots for this 'n'."
        )
        continue

    markers = {"variant_3": "o", "variant_4": "s"}
    linestyles = {"variant_3": "--", "variant_4": "-"}
    colors = {"variant_3": "tab:blue", "variant_4": "tab:orange"}

    for n_buckets in all_bucket_counts:
        print(f"  Generating plot for n_buckets = {n_buckets}...")
        results_subset = results_to_plot[results_to_plot["n_buckets"] == n_buckets]

        if results_subset.empty:
            print(
                f"    Skipping n_buckets = {n_buckets} (no aggregated data found for n={target_n})."
            )
            continue

        current_thread_counts = sorted(results_subset["p"].unique())
        if not current_thread_counts:
            print(
                f"    Skipping n_buckets = {n_buckets} (no thread counts 'p' found in subset for n={target_n})."
            )
            continue
        max_threads_in_subset = max(current_thread_counts)

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            plt.style.use(plot_style_to_use)
        except Exception as e:
            print(f"Error applying style '{plot_style_to_use}': {e}. Using default.")
            plt.style.use("default")

        ideal_threads = [t for t in all_thread_counts if t <= max_threads_in_subset]
        if not ideal_threads:
            ideal_threads = [1]
        ax.plot(
            ideal_threads,
            ideal_threads,
            label="Ideal speedup",
            color="black",
            linestyle=":",
            marker="",
            zorder=1,
        )

        data_v3 = results_subset[results_subset["variant"] == "variant_3"].sort_values(
            "p"
        )
        if not data_v3.empty:
            ax.errorbar(
                data_v3["p"],
                data_v3["mean"],
                yerr=data_v3["std"],
                label=f"Variant 3 (B={n_buckets})",
                marker=markers["variant_3"],
                linestyle=linestyles["variant_3"],
                color=colors["variant_3"],
                capsize=3,
                alpha=0.8,
                zorder=2,
            )

        data_v4 = results_subset[results_subset["variant"] == "variant_4"].sort_values(
            "p"
        )
        if not data_v4.empty:
            ax.errorbar(
                data_v4["p"],
                data_v4["mean"],
                yerr=data_v4["std"],
                label=f"Variant 4 (B={n_buckets})",
                marker=markers["variant_4"],
                linestyle=linestyles["variant_4"],
                color=colors["variant_4"],
                capsize=3,
                alpha=0.8,
                zorder=2,
            )

        ax.set_xlabel("Number of Threads (p)")
        ax.set_ylabel("Strong Scaling Speedup")
        title = f"Strong Scaling Speedup (time_global) for {n_buckets} Buckets"
        ax.set_title(title)

        ax.set_xticks(all_thread_counts)
        try:
            from matplotlib.ticker import ScalarFormatter

            ax.get_xaxis().set_major_formatter(ScalarFormatter())
        except ImportError:
            pass

        max_ideal_speedup = max(ideal_threads) if ideal_threads else 1
        max_observed_speedup = (
            results_subset["mean"].max() if not results_subset.empty else 1
        )
        ax.set_ylim(bottom=0, top=max(max_ideal_speedup, max_observed_speedup) * 1.1)

        ax.grid(True, which="both", linestyle="-", linewidth=0.5)
        ax.legend(loc="upper left")
        plt.tight_layout()

        plot_filename_suffix = f"buckets_{n_buckets}"
        if "n" in df_combined.columns and len(valid_n_values) > 0:
            plot_filename_suffix += f"_n_{target_n}"
        plot_filename = (
            output_dir / f"{output_plot_filename_prefix}{plot_filename_suffix}.png"
        )

        try:
            plt.savefig(plot_filename, dpi=150)
            print(f"    Plot saved: {plot_filename}")
        except Exception as e:
            print(f"    Error saving plot {plot_filename}: {e}")

        plt.close(fig)

print("\nScript finished.")
