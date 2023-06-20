#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=25GB        # memory per node
#SBATCH --time=0-00:15      # time (DD-HH:MM)
#SBATCH --account=def-acliu
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=yael.demers@mail.mcgill.ca
#SBATCH --mail-type=ALL

module load cuda cudnn 
source ../../tensorflow/bin/activate
python run_upaint_v2.py