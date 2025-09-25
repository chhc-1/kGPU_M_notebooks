#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --mail-user=hc8g23@soton.ac.uk
##SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --exclude=indigo[51,52,53,54,55,56,57,59,60]

#SBATCH --time=03:00:00

#SBATCH --output=run.out
#SBATCH --error=run.err


module load julia/1.11.6
module load cuda/12.3

##watch -n 1 nvidia-smi --filename=USAGE
##watch -n 1 nvidia-smi --filename=USAGE --loop=2 --display=UTILIZATION --query-gpu=utilization.gpu --format=csv
##watch -n 1 
##nvidia-smi --filename=USAGE --loop=2 --query-gpu=utilization.gpu --format=csv
##watch -n 3 nvidia-smi --filename=USAGE
##nvidia-smi --filename=USAGE --loop=5

NUM_SAMPLES=10000
INTERVAL=5
OUTPUT_FILE="USAGE.log"
##START_TIME = $(date +%s)

(julia --project=./ kGPU_M_GPU_usage.jl --with --args) &
JOB_PID=$!

while (ps -p $JOB_PID > /dev/null); do
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -n 1)
  echo "$(date +%H:%M:%S) - $GPU_UTIL" >> $OUTPUT_FILE
  
  sleep $INTERVAL
done

##nvidia-smi --filename=USAGE --loop=5 --query-gpu=utilization.gpu --format=csv,noheader

##nvidia-smi --filename=USAGE --loop=5
##nvidia-smi --filename=USAGE --loop=5 --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1

#for (( i=0; i<$NUM_SAMPLES; i++ )); do
#    # Get the GPU utilization (extract the first GPU's utilization)
#    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)
#
#    # Log the time and utilization
#    echo "$(date +%H:%M:%S) - $GPU_UTIL" >> $OUTPUT_FILE
#
#    # Check if the job is still running
#    if ! ps -p $JOB_PID > /dev/null; then
#        echo "Job (PID: $JOB_PID) finished early."
#        break
#    fi

#    sleep $INTERVAL
#done