#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=150GB        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --account=def-acliu
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=yael.demers@mail.mcgill.ca
#SBATCH --mail-type=ALL

echo "Removing outdated checkpoints"
rm ../checkpoints/*

echo "Setting up environment"
module load cuda cudnn 
source ../../tensorflow/bin/activate

echo "Running 'run_upaint_v2.py'"
python run_upaint_v2.py