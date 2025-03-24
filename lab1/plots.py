import pandas as pd
import matplotlib.pyplot as plt

df_send_recv_same = pd.read_csv("results_1.csv")
df_isend_irecv_same = pd.read_csv("results_3.csv")
df_send_recv_diff = pd.read_csv("results_2.csv")
df_isend_irecv_diff = pd.read_csv("results_4.csv")

x_same = df_send_recv_same["Size(B)"]
x_diff = df_send_recv_diff["Size(B)"]

plt.figure(figsize=(10, 6))
plt.semilogx(
    x_same,
    df_send_recv_same["Throughput(Mbit/s)"],
    marker="o",
    label="send_recv (MPI_Send/MPI_Recv)",
)
plt.semilogx(
    x_same,
    df_isend_irecv_same["Throughput(Mbit/s)"],
    marker="o",
    label="isend_irecv (MPI_Isend/MPI_Irecv)",
)

plt.xticks(x_same, x_same, rotation=45, ha="right")
plt.grid(True, linestyle="--", alpha=0.7, which="both")
plt.grid(True, which="minor", alpha=0.2)
plt.title("MPI Point-to-Point (2 procesory na jednym nodzie)")
plt.xlabel("Rozmiar (B)")
plt.ylabel("Przepustowość (Mbit/s)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_same_node.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogx(
    x_diff,
    df_send_recv_diff["Throughput(Mbit/s)"],
    marker="o",
    label="send_recv (MPI_Send/MPI_Recv)",
)
plt.semilogx(
    x_diff,
    df_isend_irecv_diff["Throughput(Mbit/s)"],
    marker="o",
    label="isend_irecv (MPI_Isend/MPI_Irecv)",
)

plt.xticks(x_diff, x_diff, rotation=45, ha="right")
plt.grid(True, linestyle="--", alpha=0.7, which="both")
plt.grid(True, which="minor", alpha=0.2)
plt.title("MPI Point-to-Point (2 procesory na różnych nodach)")
plt.xlabel("Rozmiar (B)")
plt.ylabel("Przepustowość (Mbit/s)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_diff_node.png")
plt.show()
