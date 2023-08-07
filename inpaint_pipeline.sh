#!/bin/bash
#SBATCH --gres=gpu:v100l:2   # request GPU -> v100l is a cluster with sufficient GPU RAM
#SBATCH --cpus-per-task=1    # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=100GB          # memory per node, v100l can provide at most 100GB
#SBATCH --time=00-01:30      # time (DD-HH:MM)
#SBATCH --account=def-acliu
#SBATCH --output=/home/ydemers/projects/rrg-acliu/ydemers/RFI-Inpainting/outputs/slurm/UPAINT-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=yael.demers@mail.mcgill.ca
#SBATCH --mail-type=ALL

# Define variables for training parameters
MODEL_PATH="checkpoints/latest_upaint.hdf5"
MODEL_TYPE="UPAINT"
EPOCHS=100
BATCH_SIZE=4

# Print the training parameters header to the slurm output
echo "===================================="
echo "Training Parameters:"
echo "===================================="

# Print the individual training parameters to the slurm output with separators
echo "Model Path: $MODEL_PATH"
echo "-----------------------------------"
echo "Model Type: $MODEL_TYPE"
echo "-----------------------------------"
echo "Epochs: $EPOCHS"
echo "-----------------------------------"
echo "Batch Size: $BATCH_SIZE"
echo "==================================="

# Remove outdated checkpoints
echo ">> Removing outdated checkpoints"
rm "$MODEL_PATH"

# Setting up environment
echo ">> Setting up environment"
module load cuda cudnn
source ../tensorflow/bin/activate

# Run the Python script with the variables
echo ">> Running 'run_upaint.py'"
python scripts/run_upaint.py

#python run_upaint.py --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"