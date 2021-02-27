#!/bin/bash

#SBATCH --export=ALL            
#SBATCH -J test_cpu             
#SBATCH --exclude=node28        
#SBATCH -o logs/CPU-%A_%a.log   
#SBATCH -e logs/CPU-%A_%a.err   
#SBATCH -t 0-100:00            
#SBATCH --mem 4G                
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=1       
#SBATCH -p cpu                   

python mip.py
