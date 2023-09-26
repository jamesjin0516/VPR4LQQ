#!/bin/bash

work_path=/home/unav/Desktop/Resolution_Agnostic_VPR
data_path=/mnt/data/Resolution_Agnostic_VPR

script=$work_path/src/trainer_pitts250.py

conda activate Resolution-Agnostic-VPR
CUDA_VISIBLE_DEVICES=1 python $script -c $work_path/configs/trainer_pitts250.yaml
