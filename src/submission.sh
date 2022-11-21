#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=230GB       
#SBATCH --time=0-03:40
#SBATCH --mail-user=<yael.demers@mail.mcgill.ca>
#SBATCH --mail-type=ALL 

python run.py 