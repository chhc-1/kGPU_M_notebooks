#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --mail-user=hc8g23@soton.ac.uk
##SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --exclude=indigo[51,52,53,54,55,56,57,59,60]

#SBATCH --time=03:00:00

#SBATCH --output=run.err
#SBATCH --error=run.out


module load julia/1.11.6
module load cuda/12.3


NUM_SAMPLES=10000
INTERVAL=1
OUTPUT_FILE="USAGE.log"
##START_TIME = $(date +%s)

(julia --project=./ Benchmark_kGPU_M_isotesting.jl --with --args) &
JOB_PID=$!

while (ps -p $JOB_PID > /dev/null); do
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -n 1)
  echo "$(date +%H:%M:%S) - $GPU_UTIL" >> $OUTPUT_FILE
  sleep $INTERVAL
done