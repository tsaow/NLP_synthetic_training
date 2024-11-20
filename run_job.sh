#!/bin/bash
#SBATCH --job-name=my_job_name          # Job name
#SBATCH --output=output_%j.log          # Output log file (%j will be replaced with the job ID)
#SBATCH --error=error_%j.log            # Error log file
#SBATCH --partition=compute             # Partition to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for a single program)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory per node
#SBATCH --time=01:00:00                 # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                    # Request one GPU (if needed)
#SBATCH --account=your_account_name     # Your account name

# Load necessary modules
module load python/3.9  # or the appropriate module for your environment

# Activate your virtual environment
source ~/myenv/bin/activate

# Run your Python script
python3 make_data.py