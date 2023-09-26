#!/bin/bash

work_path=/home/unav/Desktop/Resolution_Agnostic_VPR
data_path=/mnt/data/Resolution_Agnostic_VPR
model_path=$data_path
models='parameters/RA_VPR/unav_triplet/203p_30_1e-05,parameters/RA_VPR/unav_distill_vlad_triplet/203p_30_1e-05'

# model_path=$data_path/parameters/RA_VPR/unav_distill_vlad_triplet/203p_30_1e-06

script=$work_path/src/test.py

conda activate Resolution-Agnostic-VPR
CUDA_VISIBLE_DEVICES=1 python $script -c $work_path/configs/trainer_unav.yaml -p $model_path -m $models
