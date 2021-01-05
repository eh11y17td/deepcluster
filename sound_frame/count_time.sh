value=0
cnt=0
# path=`find ./dataset/audio/train -type f | grep "weak2strong/"`
# path=`find .//weak2strong/sound_frame_dcnet_0.001_-5/checkpoints/checkpoint_50.0 -type f`
path=`find ./weak2strong/normal -type f`

for i in $path
do
time=`sox --i -D $i`
value=`echo "$value+$time" | bc`
cnt=$((cnt + 1))
# echo $value 
# echo $cnt
done

echo $value 
echo $cnt