import csv
from pathlib import Path

base_dir = Path(".")
run_folders = [f"run{i}" for i in range(1, 7)]
source_results_subdir = "results"
csv_file_pattern = "results_variant2_*.csv"
output_filename = "variant_4_merged.csv"
output_header = [
    "n",
    "p",
    "time_randomization",
    "time_split",
    "time_merge",
    "time_sort",
    "time_rewrite",
    "time_global",
    "n_buckets",
    "run",
]
time_merge_placeholder = 0

print("Starting merge process...")
print(f"Looking for folders: {', '.join(run_folders)}")
print(f"Output file: {output_filename}")

all_data_rows = []

for run_folder_name in run_folders:
    run_path = base_dir / run_folder_name
    results_path = run_path / source_results_subdir
    try:
        run_number = int(run_folder_name.replace("run", ""))
    except ValueError:
        print(
            f"Warning: Could not parse run number from folder name '{run_folder_name}'. Skipping."
        )
        continue

    if not results_path.is_dir():
        print(
            f"Warning: Directory not found: {results_path}. Skipping run {run_number}."
        )
        continue

    print(f"Processing run {run_number} in '{results_path}'...")

    found_files = list(results_path.glob(csv_file_pattern))

    if not found_files:
        print(f"  No CSV files matching '{csv_file_pattern}' found in {results_path}.")
        continue

    print(f"  Found {len(found_files)} matching CSV files.")

    for csv_file_path in found_files:
        try:
            with open(csv_file_path, "r", newline="") as infile:
                reader = csv.reader(infile)
                source_header = next(reader)
                try:
                    data_row = next(reader)
                except StopIteration:
                    print(
                        f"  Warning: Skipping empty data file (no data row): {csv_file_path.name}"
                    )
                    continue

                if len(data_row) < 8:
                    print(
                        f"  Warning: Skipping row in {csv_file_path.name} due to insufficient columns: {data_row}"
                    )
                    continue

                output_row = [
                    data_row[1],  # n (Elements)
                    data_row[0],  # p (Threads)
                    data_row[3],  # time_randomization (GenTime)
                    data_row[4],  # time_split (DistTime)
                    time_merge_placeholder,  # time_merge (Placeholder)
                    data_row[5],  # time_sort (SortTime)
                    data_row[6],  # time_rewrite (CopyTime)
                    data_row[7],  # time_global (TotalTime)
                    data_row[2],  # n_buckets (Buckets)
                    run_number,  # run (derived from folder name)
                ]
                all_data_rows.append(output_row)

        except FileNotFoundError:
            print(f"  Error: File not found during processing {csv_file_path}")
        except Exception as e:
            print(f"  Error processing file {csv_file_path}: {e}")

if not all_data_rows:
    print("\nNo data rows were collected. Output file will not be created.")
else:
    try:
        with open(base_dir / output_filename, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(output_header)
            writer.writerows(all_data_rows)
        print(f"\nSuccessfully merged {len(all_data_rows)} rows into {output_filename}")
    except IOError as e:
        print(f"\nError writing output file {output_filename}: {e}")

print("Merge process finished.")
