#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=test_distillbert
#SBATCH --gres=gpu:1
#SBATCH --mem=90000
#SBATCH --time=10:00:00
#SBATCH -o /home/ma/ma_ma/ma_mbui/clkd/slurm/output/slurm-%j.out

now=$(date +"%T")
echo "Program starts:  $now"

# Activate conda env
# srun $1
# Run script
# conda activate clkd_fairness
export HYDRA_FULL_ERROR=1
srun python /home/ma/ma_ma/ma_mbui/clkd/run.py +experiment=jigsaw_distillbert


end=$(date +"%T")
echo "Completed: $end"