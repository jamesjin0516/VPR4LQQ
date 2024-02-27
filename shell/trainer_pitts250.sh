#!/bin/bash

work_path=/home/gj2148/VPR4LQQ
data_path=/mnt/data/Resolution_Agnostic_VPR

script=$work_path/src/trainer_pitts250.py

# conda activate Resolution-Agnostic-VPR
CUDA_VISIBLE_DEVICES=0 python $script -c $work_path/configs/trainer_pitts250.yaml
