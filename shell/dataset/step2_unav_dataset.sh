#!/bin/bash

work_path=/home/unav/Desktop/VPR4LQQ

script=$work_path/src/dataset/unav_data.py

CUDA_VISIBLE_DEVICES=1 python $script -c $work_path/configs/trainer_unav.yaml

