#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=distill_warmup_randomseed0
#SBATCH --gres=gpu:1
#SBATCH --mem=90000
#SBATCH --time=15:00:00
#SBATCH -o /home/ma/ma_ma/ma_mbui/clkd/slurm/output/slurm-%j.out

now=$(date +"%T")
echo "Program starts:  $now"

# Activate conda env
# srun $1
# Run script
# source /pfs/data5/home/ma/ma_ma/ma_mbui/.conda/envs/adapters_fairness/bin adapters_fairness
export HYDRA_FULL_ERROR=1
srun python /home/ma/ma_ma/ma_mbui/clkd/run.py +experiment=jigsaw_distillbert_warmup seed=0 datamodule.split_seed=6


end=$(date +"%T")
echo "Completed: $end"