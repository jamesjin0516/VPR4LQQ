#!/bin/bash
video=ICT1_3
dir=/mnt/data/Resolution_Agnostic_VPR/logs/unav/$video/000
total_count=0
res_list=''203p' '405p' '810p''
# res_list=''135p' '270p' '540p''

echo $video

for res in $res_list
    do
        for folder in 000 020 040 060 080 100 120 140 160 180 200 220 240 260 280 300 320 340
            do 
                # find $dir/$folder/$res -type f | wc -l
                count=$(find $dir/$folder/images/$res -type f | wc -l)
                let "total_count += count"
            done
        echo "  $res: $total_count"
        total_count=0
    done