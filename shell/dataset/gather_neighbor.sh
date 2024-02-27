#!/bin/bash
name=unav

work_path=/home/unav/Desktop/VPR4LQQ
data_path=/mnt/data/VPR4LQQ
# outf=/mnt/data/UNav-Dataset
outf=/mnt/data/rerun

script=$work_path/src/dataset/gather_neighbor.py

conda activate VPR4LQQ
python $script -r $data_path -n $name -o $outf
