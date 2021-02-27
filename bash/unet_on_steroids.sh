#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J unet_on_steroids
#SBATCH --exclude=node28
#SBATCH -o ../logs/GPU-%A.log
#SBATCH -e ../logs/GPU-%A.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 16G              # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=4       # CPU cores per process (default 1)
#SBATCH -p gpu-cbio                  # Name of the partition to use

echo 'unet_on_steroids.sh'
echo
nvidia-smi
echo

# directories
input_directory='/mnt/data3/mphilbert/'
output_directory='/mnt/data3/mphilbert/output/'
log_directory="/mnt/data3/mphilbert/output/log"

# python script
script='../src/unet_on_steroids.py'

python "$script" "$input_directory" "$output_directory" \
        --log_directory "$log_directory"