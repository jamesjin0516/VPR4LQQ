#!/bin/bash

work_path=/home/unav/Desktop/Resolution_Agnostic_VPR

script=$work_path/src/dataset/unav_data.py

python $script -c $work_path/configs/trainer.yaml

