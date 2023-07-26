#!/bin/bash
#SBATCH --gres=gpu:4         # request GPU "generic resource"
#SBATCH --cpus-per-task=1    # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=100GB          # memory per node -> optimally 250GB
#SBATCH --time=00-00:50      # time (DD-HH:MM) -> optimally 00-03:00
#SBATCH --account=def-acliu
#SBATCH --output=/home/ydemers/projects/def-acliu/ydemers/RFI-Inpainting/outputs/slurm/UPAINT-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=yael.demers@mail.mcgill.ca
#SBATCH --mail-type=ALL

#cat scripts/run_upaint_v2.py

echo "Removing outdated checkpoints"
rm models/upaint/checkpoints/latest_upaint.hdf5

echo "Setting up environment"
module load cuda cudnn 
source ../tensorflow/bin/activate

echo "Running 'run_upaint_v2.py'"
#python -m scripts.run_upaint_v2
python scripts/run_upaint_v2.py