#!/bin/bash

work_path=/scratch/lg3490/VPR4LQQ

script=$work_path/src/trainer_pitts250.py

# conda activate Resolution-Agnostic-VPR
CUDA_VISIBLE_DEVICES=0 python $script -c $work_path/configs/trainer_pitts250.yaml

# python src/trainer_pitts250.py -c configs/trainer_pitts250.yaml
# python src/test_trained_model.py -c configs/test_trained_model.yaml