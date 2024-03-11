#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=svox
#SBATCH --mail-type=NONE
#SBATCH --output=/scratch/lg3490/VPR4LQQ/testing_data_svox.out


module purge

singularity exec --nv --overlay /scratch/gj2148/VPR4LQQ/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate vpr4lqq; cd /scratch/lg3490/VPR4LQQ/src; python test_trained_model.py --distill 1 --vlad 1 --triplet 1; conda deactivate"

# tokyo247 runtimes (-c10 --mem=80GB no gpus = --time=11:00) dataset creation: 3 minutes; every vpr evaluation: 1:10
# msls runtimes (-c10 --mem=25GB --gres=gpu:1 = --time=1:16:00) dataset creation: 1 minute; every vpr evaluation: 15:00
# amstertime runtimes (-c10 --mem=10GB --gres=gpu:1 = --time=00:20:00) dataset creation: <1 minute; every vpr evaluation: 3:00

