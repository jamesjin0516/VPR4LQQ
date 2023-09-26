#!/bin/bash

work_path=/home/unav/Desktop/VPR4LQQ
data_path=/mnt/data/VPR4LQQ

script=$work_path/src/trainer_unav.py

conda activate VPR4LQQ
CUDA_VISIBLE_DEVICES=1 python $script -c $work_path/configs/trainer_unav.yaml
