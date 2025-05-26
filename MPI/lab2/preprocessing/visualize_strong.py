import pandas as pd
import matplotlib.pyplot as plt
import os

size_mapping = {"small": 1, "medium": 2, "huge": 3}
colors = {
    "small": "blue",
    "medium": "green",
    "huge": "red",
}

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
strong = grouped[grouped["variant"] == "strong"]

metrics = {
    "execution_time": ("Czas wykonania (s)", "strong_scaling.png"),
    "speedup": ("Speedup", "speedup.png"),
    "efficiency": ("Efficiency", "efficiency.png"),
    "serial_fraction": ("Serial Fraction", "serial_fraction.png"),
}

for metric, (ylabel, filename) in metrics.items():
    plt.figure(figsize=(12, 8))
    for i, size in enumerate(size_mapping.keys(), 1):
        subset = strong[strong["size"] == size]
        t1 = subset[subset["num_procs"] == 1]["mean"].values[0]

        if metric == "execution_time":
            values = subset["mean"]
            errors = subset["std"]
        elif metric == "speedup":
            values = t1 / subset["mean"]
        elif metric == "efficiency":
            values = (t1 / subset["mean"]) / subset["num_procs"]
        elif metric == "serial_fraction":
            values = ((1 / (t1 / subset["mean"])) - (1 / subset["num_procs"])) / (
                1 - (1 / subset["num_procs"])
            )

        plt.subplot(1, 3, i)

        if metric == "execution_time":
            plt.errorbar(
                subset["num_procs"],
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
                subset["num_procs"],
                values,
                marker="o",
                linestyle="-",
                color=colors[size],
                label=size,
            )

        num_procs_range = strong["num_procs"].unique()
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
            plt.axhline(
                y=0,
                color="gray",
                linestyle="--",
                label="Idealny (F = 0)",
            )

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
    plt.savefig("strong_" + filename)
