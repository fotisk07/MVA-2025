#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 <prod10|prod20|prod40|prod80> <command> [args...]"
  exit 1
fi

PARTITION="$1"
shift
CMD=("$@")

# ----------------------------
# Partition → MIG mapping
# ----------------------------
case "$PARTITION" in
  prod10)
    GPU_SPEC="gpu:1g.10gb:1"
    ;;
  prod20)
    GPU_SPEC="gpu:2g.20gb:1"
    ;;
  prod40)
    GPU_SPEC="gpu:3g.40gb:1"
    ;;
  prod80)
    GPU_SPEC="gpu:A100.80gb:1"
    ;;
  *)
    echo "Unknown partition: $PARTITION"
    exit 1
    ;;
esac

JOB_NAME=$(basename "${CMD[0]}")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GPU_SPEC}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -e
echo "Running on partition: ${PARTITION}"
echo "GPU spec: ${GPU_SPEC}"
echo "Command: ${CMD[@]}"

srun ${CMD[@]}
EOF
