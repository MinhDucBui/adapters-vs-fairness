#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=test_bert
#SBATCH --gres=gpu:1
#SBATCH --mem=90000
#SBATCH --time=12:00:00
#SBATCH -o /home/ma/ma_ma/ma_mbui/clkd/slurm/output/slurm-%j.out

now=$(date +"%T")
echo "Program starts:  $now"

# Activate conda env
source $HOME/.bashrc
conda activate adapters_vs_fairness
# srun $1
# Run script
# conda activate clkd_fairness
export HYDRA_FULL_ERROR=1
srun python /pfs/work7/workspace/scratch/ma_mbui-minbui/clkd/run.py +experiment=bert_lora seed=1


end=$(date +"%T")
echo "Completed: $end"