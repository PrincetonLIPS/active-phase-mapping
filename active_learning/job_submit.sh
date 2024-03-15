#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=100:00:00
#SBATCH --job-name=ternary
#SBATCH --export=ALL
#SBATCH -o ternary_out
#SBATCH -e ternary_err

# load modules
module load apps/python3/2020.02
conda activate apm

cd $SLURM_SUBMIT_DIR
echo "running job"
#srun python base.py -num_y 10 -num_samples 200 -num_curves 200 -entropy_type joint -cores 1 -nodes 1 -directory ../data/base -method BASE
srun python cal.py -num_y 10 -num_samples 200 -num_curves 200 -entropy_type joint -cores 6 -nodes 1 -directory ../data/ternary -method CHAASE
echo "job has finished"
