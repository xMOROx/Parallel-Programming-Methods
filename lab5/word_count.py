import time
import argparse
from collections import Counter
import sys
import os


def sequential_word_count(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.", file=sys.stderr)
        return None, 0

    word_counts = Counter()

    print(f"Starting processing file: {file_path}...")
    start_time = time.perf_counter()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = line.strip().split()
                word_counts.update(words)

    except Exception as e:
        print(
            f"An error occurred while processing file {file_path}: {e}",
            file=sys.stderr,
        )
        return None, 0

    end_time = time.perf_counter()
    processing_time = end_time - start_time
    print(f"Finished processing file {file_path}. Time: {processing_time:.4f} seconds.")

    return word_counts, processing_time


def save_results_to_file(word_counts, output_file_path):
    print(f"\nSaving results to file: {output_file_path}...")
    try:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            sorted_counts = word_counts.most_common()

            for word, count in sorted_counts:
                outfile.write(f"{word}: {count}\n")
        print(f"Results successfully saved to {output_file_path}")
        return True
    except IOError as e:
        print(
            f"ERROR: Could not save results to file {output_file_path}: {e}",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"ERROR: Unexpected error while writing to file {output_file_path}: {e}",
            file=sys.stderr,
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Word Count algorithm in Python with saving to file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output TXT file where all counted words sorted by frequency will be saved.",
    )

    args = parser.parse_args()

    word_counts, processing_time = sequential_word_count(args.input_file)

    if word_counts is not None:
        print("\n--- Summary ---")
        print(f"File processed: {args.input_file}")
        print(f"Total processing time: {processing_time:.4f} seconds")

        if args.output:
            save_results_to_file(word_counts, args.output)
        else:
            print(
                "\nNo output file path provided (-o), results were not saved to a file."
            )

    else:
        print(f"\nFile {args.input_file} was not processed successfully.")


if __name__ == "__main__":
    main()
