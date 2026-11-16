#!/bin/bash
### Test scenarios: one virtual patient per array task (30 tasks).
### Adapt partition, account and environment to your cluster, then, from
### experiments/lnns2026:   mkdir -p logs && sbatch slurm/run_eval.slurm
#SBATCH -J eval_scen
#SBATCH -p cpu
##SBATCH -A <your_account>
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

# source <conda.sh> && conda activate <env>

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python eval_scenarios.py --patient-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK" --gains gains_pid_v2.csv --outdir results_eval
