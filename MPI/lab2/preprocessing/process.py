import os
import pandas as pd

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for i in range(1, 31):
    input_file = os.path.join(f"experiment_{i}", "experiment_results.csv")
    output_file = os.path.join(output_folder, f"experiment_results_{i}.csv")

    df = pd.read_csv(input_file)

    df = df.drop(columns=["speedup", "efficiency", "serial_fraction"])

    df.to_csv(output_file, index=False)

    print(f"Processed {input_file} -> {output_file}")
