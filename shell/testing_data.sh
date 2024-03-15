#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=td_svox
#SBATCH --mail-type=NONE
#SBATCH --output=/scratch/lg3490/VPR4LQQ/testing_data_svox_generation_new.out

module purge

singularity exec --nv --overlay /scratch/gj2148/VPR4LQQ/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate vpr4lqq; cd /scratch/lg3490/VPR4LQQ/src/dataset; python testing_data.py; conda deactivate"

# empty raw folder command (in case you would like to restore dataset state)
# ls -1 tokyo247/images/test/database/raw/ | while read name; do mv tokyo247/images/test/database/raw/$name tokyo247/images/test/database/; done;
