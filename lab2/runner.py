#!/usr/bin/env python3
import subprocess
import re
import math
import sys

NUM_PROCS = list(range(1, 13))

variants = {
    "strong": {
        "binary": "mpi_pi_par",
        "sizes": {
            "small": 10000000,
            "huge": 25000000000,
            "medium": int(math.sqrt(10000000 * 25000000000)),
        },
    },
    "weak": {
        "binary": "mpi_pi_par_per_cpu",
        "sizes": {
            "small": 10000000,
            "huge": 10000000000,
            "medium": int(math.sqrt(10000000 * 10000000000)),
        },
    },
}

results = {
    "strong": {"small": {}, "medium": {}, "huge": {}},
    "weak": {"small": {}, "medium": {}, "huge": {}},
}


def compile_codes():
    compile_commands = [
        (["mpicc", "mpi_pi_par.c", "-o", "mpi_pi_par"], "mpi_pi_par"),
        (
            ["mpicc", "mpi_pi_par_per_cpu.c", "-o", "mpi_pi_par_per_cpu"],
            "mpi_pi_par_per_cpu",
        ),
    ]
    for cmd, binary in compile_commands:
        print(f"Compiling {binary} ...", flush=True)
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully compiled {binary}.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation of {binary}.", flush=True)
            sys.exit(1)


def run_experiment(binary, points, num_procs):
    cmd = [
        "mpirun",
        "-np",
        str(num_procs),
        f"./{binary}",
        str(points),
    ]
    print("Running:", " ".join(cmd), flush=True)
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Execution error:", e, flush=True)
        return None

    stdout_str = result.stdout.decode("utf-8")
    stderr_str = result.stderr.decode("utf-8")

    print("\nCommand Output:", flush=True)
    print("=== STDOUT ===", flush=True)
    print(stdout_str, flush=True)
    print("\n=== STDERR ===", flush=True)
    print(stderr_str, flush=True)
    print("=============", flush=True)

    match = re.search(r"Execution time:\s*([\d\.]+)\s*seconds", stdout_str)
    if match:
        exec_time = float(match.group(1))
        print(f"  Execution time: {exec_time} s", flush=True)
        return exec_time
    else:
        print("Failed to read execution time from output", flush=True)
        return None


def run_all_experiments():
    for variant, config in variants.items():
        binary = config["binary"]
        for size_label, size_value in config["sizes"].items():
            print(
                f"\n[Variant: {variant.upper()}] Size '{size_label}' = {size_value}",
                flush=True,
            )
            for p in NUM_PROCS:
                time_val = run_experiment(binary, size_value, p)
                if time_val is None:
                    print("Error in experiment, aborting.", flush=True)
                    sys.exit(1)
                results[variant][size_label][p] = time_val


def save_results_to_csv(results, filename="experiment_results.csv"):
    print(f"Saving results to file {filename}...", flush=True)
    with open(filename, "w") as f:
        f.write(
            "variant,size,num_procs,execution_time,speedup,efficiency,serial_fraction\n"
        )
        for variant, sizes in results.items():
            for size_label, proc_times in sizes.items():
                baseline = proc_times.get(1)
                for num_proc, time_val in sorted(proc_times.items()):
                    speedup = baseline / time_val if baseline else 0
                    efficiency = speedup / num_proc if num_proc else 0
                    serial_fraction = (
                        0
                        if num_proc == 1
                        else (1 / speedup - 1 / num_proc) / (1 - 1 / num_proc)
                    )
                    f.write(
                        f"{variant},{size_label},{num_proc},{time_val},{speedup},{efficiency},{serial_fraction}\n"
                    )
    print(f"Results successfully saved to {filename}", flush=True)


def main():
    compile_codes()
    run_all_experiments()
    save_results_to_csv(results)


if __name__ == "__main__":
    main()

