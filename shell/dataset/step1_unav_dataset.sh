#!/bin/bash

video=Ratchasuda2_1
database=unav
work_path=/home/unav/Desktop/VPR4LQQ
data_path=/mnt/data/VPR4LQQ
openvslam=/root/Desktop/openvslam/openvslam/build/run_image_slam
config=$work_path/configs/equirectangular_slam.yaml
vocab=/root/Desktop/openvslam/openvslam/build/orb_vocab/orb_vocab.dbow2
json_name="${video}.json"

src_dir=$data_path/data/"${video}_images"
video_path=$data_path/data/$database/video/$video.mp4
json_path=$data_path/data/$database/utm
log_folder=$data_path/logs/$database/$video
map_path=$json_path/"${video}.msg"

[ ! -d "$src_dir" ] && mkdir -p "$src_dir"

###### Extract frames from data
# ffmpeg -i $video_path -r 29.97 $src_dir/%05d.png 

###### Map the environment
# $openvslam -v $vocab -i $src_dir -c $config --no-sleep --map-db $map_path

# ####### Align the sparse map
# conda activate Resolution-Agnostic-VPR
# python $work_path/src/tools/Aligner.py --maps $map_path --src_dir $src_dir --outf $json_path --plan $data_path/data/floorplan --file_name $json_name

###### Equirec into perspective images
conda activate VPR4LQQ

work_path=/home/unav/Desktop/VPR4LQQ
script=$work_path/src/dataset/Equi2Pers.py

python $script -c $work_path/configs/trainer_unav.yaml -v $video -d $database 

