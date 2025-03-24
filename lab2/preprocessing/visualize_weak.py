import pandas as pd
import matplotlib.pyplot as plt
import os
import math

small_m = 10_000_000
huge_m = 10_000_000_000
medium_m = int(math.sqrt(small_m * huge_m))
size_mapping = {"small": small_m, "medium": medium_m, "huge": huge_m}
colors = {"small": "blue", "medium": "green", "huge": "red"}

folders = [f"experiment_{i}" for i in range(1, 31)]
df_list = []

for folder in folders:
    csv_path = os.path.join(folder, "experiment_results.csv")
    if os.path.isfile(csv_path):
        df_list.append(pd.read_csv(csv_path))

if not df_list:
    print("Brak plików z danymi. Zakończono.")
    exit(0)

data = pd.concat(df_list, ignore_index=True)
data = data[["variant", "size", "num_procs", "execution_time"]]

grouped = (
    data.groupby(["variant", "size", "num_procs"])["execution_time"]
    .agg(["mean", "std"])
    .reset_index()
)
grouped.columns = ["variant", "size", "num_procs", "mean", "std"]
weak = grouped[grouped["variant"] == "weak"]

metrics = {
    "execution_time": ("Czas wykonania (s)", "weak_scaling.png"),
    "speedup": ("Speedup", "speedup.png"),
    "efficiency": ("Efficiency", "efficiency.png"),
    "serial_fraction": ("Serial Fraction", "serial_fraction.png"),
}

for metric, (ylabel, filename) in metrics.items():
    plt.figure(figsize=(12, 8))
    for i, (size, M) in enumerate(size_mapping.items(), 1):
        subset = weak[weak["size"] == size]
        t1_values = subset[subset["num_procs"] == 1]["mean"].values
        if len(t1_values) == 0:
            continue

        t1 = t1_values[0]
        num_procs_list = subset["num_procs"].values
        speedup_list, efficiency_list, serial_fraction_list = [], [], []

        for mean, p in zip(subset["mean"], subset["num_procs"]):
            n = p * M
            S = (n / M) * (t1 / mean)
            E = S / p
            if S == 0 or p == 1:
                sf = float("nan")
            else:
                sf = ((1 / S) - 1 / p) / (1 - 1 / p)
            speedup_list.append(S)
            efficiency_list.append(E)
            serial_fraction_list.append(sf)

        if metric == "execution_time":
            values = subset["mean"]
            errors = subset["std"]
        elif metric == "speedup":
            values = speedup_list
        elif metric == "efficiency":
            values = efficiency_list
        elif metric == "serial_fraction":
            values = [sf if not math.isnan(sf) else None for sf in serial_fraction_list]

        plt.subplot(1, 3, i)

        if metric == "execution_time":
            plt.errorbar(
                num_procs_list,
                values,
                yerr=errors,
                marker="o",
                linestyle="-",
                color=colors[size],
                label=size,
                capsize=4,
                elinewidth=1,
                markeredgewidth=1,
            )
        else:
            plt.plot(
                num_procs_list,
                values,
                marker="o",
                linestyle="-",
                color=colors[size],
                label=size,
            )

        num_procs_range = weak["num_procs"].unique()
        if metric == "speedup":
            plt.plot(
                num_procs_range,
                num_procs_range,
                linestyle="--",
                color="gray",
                label="Idealny (S = P)",
            )
        elif metric == "efficiency":
            plt.axhline(y=1, color="gray", linestyle="--", label="Idealny (E = 1)")
        elif metric == "serial_fraction":
            plt.axhline(y=0, color="gray", linestyle="--", label="Idealny (F = 0)")

        plt.xlabel("Liczba procesorów (P)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} - {size}")

        if metric == "efficiency":
            plt.legend(loc="lower left")
        elif metric == "serial_fraction":
            plt.legend(loc="upper right")
        else:
            plt.legend()

        plt.grid(True)

    plt.tight_layout()
    plt.savefig("weak_" + filename)

plt.show()
