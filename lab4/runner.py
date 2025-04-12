#!/usr/bin/env python3
import subprocess
import sys
import os

NUM_THREADS = [1, 2, 4, 8, 16, 32, 48]
PROBLEM_SIZE = 10_000_000

CONFIGS = [
    [("CHUNK", 40000), ("BUCKETS_NUMBER", 2**10), ("CAPACITY_MULTIPLIER", 2.0)],
    # [],  # domyślne makra z programu C
]

FILE_NAME = "variant2.c"
BINARY_PREFIX = "variant2"

os.makedirs("results", exist_ok=True)
os.makedirs("compiled", exist_ok=True)


def config_suffix(configs):
    if not configs:
        return "default"
    return "_".join(f"{name}_{str(value).replace('.', 'p')}" for name, value in configs)


def compile_program(configs):
    suffix = config_suffix(configs)
    binary_name = f"{BINARY_PREFIX}_{suffix}"
    binary_path = os.path.join("compiled", binary_name)

    compile_command = ["gcc", FILE_NAME, "-o", binary_path, "-fopenmp"]

    for name, value in configs:
        compile_command.append(f"-D{name}={value}")

    print(f"\nCompiling {binary_name} with command:")
    print(" ".join(compile_command))
    try:
        subprocess.run(compile_command, check=True)
        print(f"-> {binary_name} compiled successfully.")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Compilation failed for: {binary_name}")
        sys.exit(1)

    return binary_path


def run_experiment(binary_path, num_threads, problem_size):
    cmd = [binary_path, str(num_threads), str(problem_size)]
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as e:
        print("Error while executing program:", e)
        print("Program output:", e.stdout.decode("utf-8"))
        print("Error message:", e.stderr.decode("utf-8"))
        sys.exit(1)

    stdout_str = result.stdout.decode("utf-8")
    stderr_str = result.stderr.decode("utf-8")

    print("=== STDOUT ===")
    print(stdout_str)
    print("=== STDERR ===")
    print(stderr_str)

    csv_filename = "results.csv"
    if os.path.exists(csv_filename):
        binary_name = os.path.basename(binary_path)
        new_csv = os.path.join("results", f"results_{binary_name}_{num_threads}.csv")
        os.rename(csv_filename, new_csv)
        print(f"Results saved in file: {new_csv}")
    else:
        print("results.csv file not found after program execution.")


def main():
    for config in CONFIGS:
        binary_path = compile_program(config)
        for num_threads in NUM_THREADS:
            print(
                f"\n[Running config: {config_suffix(config)}] - Threads: {num_threads}"
            )
            run_experiment(binary_path, num_threads, PROBLEM_SIZE)


if __name__ == "__main__":
    main()
