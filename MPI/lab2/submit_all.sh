#!/bin/bash -l

BATCH_SIZE=5
START=1
END=30

for ((i = START; i <= END; i += BATCH_SIZE)); do
  JOB_IDS=()

  for ((j = i; j < i + BATCH_SIZE && j <= END; j++)); do
    EXP_DIR="experiment_$j"
    mkdir -p "$EXP_DIR"

    cp mpi_pi_par.c mpi_pi_par_per_cpu.c runner.py run.sh "$EXP_DIR/"

    cd "$EXP_DIR" || exit
    JOB_ID=$(sbatch run.sh | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    echo "Submitted job $JOB_ID in $EXP_DIR"
    cd ..
  done

  while true; do
    sleep 120

    RUNNING_JOBS=0
    for JOB_ID in "${JOB_IDS[@]}"; do
      if squeue -j "$JOB_ID" -h &>/dev/null; then
        ((RUNNING_JOBS++))
      fi
    done

    if [ "$RUNNING_JOBS" -eq 0 ]; then
      echo "Batch finished, submitting next batch."
      break
    else
      echo "$RUNNING_JOBS jobs still running, waiting..."
    fi
  done
done
