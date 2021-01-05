#!/bin/bash
weak2strong_path=("Dog" "Cat" "Dishes" "Running_water")
weak2strong_path=("Dog" "Cat" "Dishes" "Running_water" "Alarm_bell_ringing" "Blender" "Vacuum_cleaner" "Electric_shaver_toothbrush" "Frying")
# weak2strong_path=("Dog" "Cat")

weak_path=($(ls | grep ^[A-Z]))
save_path=./audio/weak2strong_NN
save_path_backup=/Work19/endohayato/deepcluster/sound_frame/weak2strong/dcnet3_0.001_-5/Jun14_01-42-52/logmel2NN

for p in ${!weak2strong_path[@]}; do
  for q in ${!weak_path[@]}; do
#   echo ${weak2strong_path[p]}
    if [ "${weak2strong_path[p]}" = "${weak_path[q]}" ]; then
        # unset ${weak_path[q]}
        weak_path[$q]=
    fi
  done
done

echo ${weak2strong_path[@]}
echo ${weak_path[@]}

echo "Old data path : $save_path"
rm $save_path/*
echo "Old data deleted!!"

rm ./metadata/weak2strong_NN.tsv
touch ./metadata/weak2strong_NN.tsv


for weak2strong_NAME in ${weak2strong_path[@]}
do
    echo "Start : $weak2strong_NAME"
    cp ./$weak2strong_NAME/* $save_path

    cat ./metadata/weak2strong_NN_sub.tsv | grep $weak2strong_NAME | grep -v "," >> ./metadata/weak2strong_NN.tsv
    echo "end : $weak2strong_NAME"
done

# for weak_NAME in ${weak_path[@]}
# do
#     echo "Start : $weak_NAME"
#     cp ./weak/$weak_NAME/* $save_path
#     cat /Work20/endohayato/DCASE2019_task4/dataset/metadata/train/weak.tsv | grep $weak_NAME >> ./metadata/weak2strong_NN.tsv
#     echo "end : $weak_NAME"
# done

sort -u ./metadata/weak2strong_NN.tsv -o ./metadata/weak2strong_NN.tsv
sed -i '1s/^/filename\tevent_labels\n/' ./metadata/weak2strong_NN.tsv

echo "### Make backup to Work19 ###"
cp -r metadata $save_path_backup
cp -r audio $save_path_backup