import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("merged_results.csv")
df2 = pd.read_csv("merged_results2.csv")


df = pd.concat([df1, df2], ignore_index=True)


summary_df = (
    df.groupby(["p", "type", "size", "wait"])["time"].agg(["mean", "std"]).reset_index()
)

summary_df["speedup"] = summary_df.groupby(["type", "size", "wait"])["mean"].transform(
    lambda x: x.iloc[0] / x
)


unique_types = summary_df["type"].unique()
for type_ in unique_types:
    plt.figure(figsize=(12, 6))
    subset = summary_df[summary_df["type"] == type_]
    for (size, wait), group in subset.groupby(["size", "wait"]):
        plt.errorbar(
            group["p"],
            group["mean"],
            yerr=group["std"],
            label=f"{size}, {wait}",
            capsize=3,
            marker="o",
            linestyle="--",
        )

    plt.xlabel("Cores")
    plt.ylabel("Time (s)")
    plt.title(f"Execution Time for {type_}")
    plt.legend()
    plt.grid()
    plt.savefig(f"time_plot_{type_}.png")


unique_sizes = summary_df["size"].unique()
for size in unique_sizes:
    plt.figure(figsize=(12, 6))
    subset = summary_df[summary_df["size"] == size]
    for (typee, wait), group in subset.groupby(["type", "wait"]):
        plt.errorbar(
            group["p"],
            group["mean"],
            yerr=group["std"],
            label=f"{typee}, {wait}",
            capsize=3,
            marker="o",
            linestyle="--",
        )

    plt.xlabel("Cores")
    plt.ylabel("Time (s)")
    plt.title(f"Execution Time for size {size}")
    plt.legend()
    plt.grid()
    plt.savefig(f"time_plot_{size}.png")


for type_ in unique_types:
    plt.figure(figsize=(12, 6))
    subset = summary_df[summary_df["type"] == type_]
    for (size, wait), group in subset.groupby(["size", "wait"]):
        plt.plot(
            group["p"],
            group["speedup"],
            label=f"{size}, {wait}",
            marker="o",
            linestyle="--",
        )

    plt.plot(
        group["p"], group["p"], linestyle="dashed", color="black", label="Ideal speedup"
    )

    plt.xlabel("Cores")
    plt.ylabel("Speedup")
    plt.title(f"Speedup for {type_}")
    plt.legend()
    plt.grid()
    plt.savefig(f"speedup_plot_{type_}.png")
