#!/bin/bash

work_path=/home/unav/Desktop/Resolution_Agnostic_VPR

script=$work_path/src/dataset/pitts250k_data.py

CUDA_VISIBLE_DEVICES=1 python $script -c $work_path/configs/trainer_pitts250.yaml