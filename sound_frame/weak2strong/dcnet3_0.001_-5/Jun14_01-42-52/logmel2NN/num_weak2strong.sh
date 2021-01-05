#!/bin/bash
# FILE=($(ls ./audio/weak2strong_NN/))
# FILE=($(ls /misc/Work20/endohayato/DCASE2019_task4/dataset/audio/train/weak))
cnt=0
for file in ${FILE[@]}
do
    file=($(echo $file | tr -s '_' ' '))
    if [ ${file[-2]} = "0.000" ]; then
        cnt=$((cnt + 1))
    fi
done
echo $cnt
# echo ${FILE[@]}