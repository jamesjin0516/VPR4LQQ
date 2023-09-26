#!/bin/bash

video=Tandon4_0
type=train
work_path=/home/unav/Desktop/Resolution_Agnostic_VPR
openvslam=/home/unav/Desktop/openvslam/openvslam/build/run_image_slam
config=$work_path/configs/equirectangular_slam.yaml
vocab=/home/endeleze/Desktop/openvslam/openvslam/build/orb_vocab/orb_vocab.dbow2

log_folder=$work_path/logs/$type/$video/equirectangular_images

script=$work_path/src/third_party/"Real-ESRGAN"/inference_realesrgan.py
inputs=/mnt/data/Resolution_Agnostic_VPR/logs/$type/$video/low
outputs=/mnt/data/Resolution_Agnostic_VPR/logs/$type/$video/fake

conda activate Resolution-Agnostic-VPR

for p in $inputs/*/**; do
    s="${outputs}${p: -8:-4}"
    d="${outputs}${p: -8}"
    [ ! -d "$s" ] && mkdir -p "$s"
    python $script -n RealESRGAN_x4plus -i $p -o $d--face_enhance
done

