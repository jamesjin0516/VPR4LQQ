#!/bin/bash
name=unav

work_path=/home/unav/Desktop/Resolution_Agnostic_VPR
data_path=/mnt/data/Resolution_Agnostic_VPR

script=$work_path/src/dataset/gather_neighbor.py

conda activate Resolution-Agnostic-VPR
python $script -r $data_path -n $name
