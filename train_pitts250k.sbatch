#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=14:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=tp250_dt_e4-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gj2148@nyu.edu
#SBATCH --output=/scratch/gj2148/VPR4LQQ/slurm_outputs/pitts250k_distill_triplet/180p_-1_1e-06_epoch4-9.out

module purge

singularity exec --nv --overlay /scratch/gj2148/VPR4LQQ/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate vpr4lqq; cd /scratch/lg3490/VPR4LQQ; source shell/trainer_pitts250.sh; conda deactivate"

# time per epoch ~100 minutes
# time per validation ~40 minutes
# 43403952: pitts250k_distill/180p_-1_1e-06_epoch1-3.out
# 43404318: pitts250k_triplet/180p_-1_1e-06_epoch1-3.out
