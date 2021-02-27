#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J parse
#SBATCH --exclude=node28
#SBATCH -o logs/CPU-%A_%a.log 
#SBATCH -e logs/CPU-%A_%a.err 
#SBATCH --array=1-60         # Number of job arrays to launch
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 8000              # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=1       # CPU cores per process (default 1)
#SBATCH -p cpu                  # Name of the partition to use


echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_JOBID: " $SLURM_JOBID

# parse parameters
file="$(pwd)/puits.txt"
puit_name=$(grep "$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 2)

python parse.py "$puit_name" 
