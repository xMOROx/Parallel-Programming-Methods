#!/usr/bin/env python3
import subprocess
import sys
import os


NUM_THREADS = [1, 2, 4, 8, 16, 32, 48]

PROBLEM_SIZE = 10_000_000_000

schedule_configs = [
    ("static", 1, False),
    ("static", 1, True),
    ("static", 1000, False),
    ("static", 1000, True),
    ("static", 40000, False),
    ("static", 40000, True),
    ("static", 200000000, False),
    ("static", 200000000, True),
    ("dynamic", 1, False),
    ("dynamic", 1, True),
    ("dynamic", 1000, False),
    ("dynamic", 1000, True),
    ("dynamic", 40000, False),
    ("dynamic", 40000, True),
    ("dynamic", 200000000, False),
    ("dynamic", 200000000, True),
    ("guided", 1000, False),
    ("guided", 1000, True),
    ("guided", 40000, False),
    ("guided", 40000, True),
    ("guided", 200000000, False),
    ("guided", 200000000, True),
]


os.makedirs("results", exist_ok=True)
os.makedirs("compiled", exist_ok=True)


def compile_program(schedule, chunk, nowait):
    binary_name = f"generate_{schedule}_{chunk}_{'nowait' if nowait else 'wait'}"
    binary_path = os.path.join("compiled", binary_name)
    compile_command = [
        "gcc",
        "generate.c",
        "-o",
        binary_path,
        "-fopenmp",
        f"-DCHUNK_SIZE={chunk}",
        f"-DSCHEDULE_TYPE={schedule}",
    ]
    if nowait:
        compile_command.append("-DNOWAIT")

    print(f"\nCompiling {binary_name} with command:")
    print(" ".join(compile_command))
    try:
        subprocess.run(compile_command, check=True)
        print(f"-> {binary_name} compiled successfully.")
    except subprocess.CalledProcessError:
        print(f"Compilation error for {binary_name}.")
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
        print("Results.csv file not found after program execution.")


def main():
    for schedule, chunk, nowait in schedule_configs:
        binary_path = compile_program(schedule, chunk, nowait)
        for num_threads in NUM_THREADS:
            print(
                f"\n[Setting: {schedule} with CHUNK_SIZE={chunk}, {'NOWAIT' if nowait else 'WAIT'}] - number of threads: {num_threads}"
            )
            run_experiment(binary_path, num_threads, PROBLEM_SIZE)


if __name__ == "__main__":
    main()
