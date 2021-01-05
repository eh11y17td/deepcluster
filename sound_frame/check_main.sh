# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# DIR="./CIFAR10_PNG"
ARCH="dcnet3"
LR=0.001
WD=-5
K=3
WORKERS=4
epoch=100
EXP="./exp/${ARCH}_${LR}_${WD}"
# resume="./exp/dcnet4_0.001_-5/Jul27_21-07-34/checkpoints/checkpoint_25.0.pth.tar"
# resume="./exp/dcnet4_0.001_-5/Jul27_21-07-34/checkpoint.pth.tar"
# resume="./normal"

resume="./exp/${ARCH}_${LR}_${WD}/Jun14_01-42-52/logmel2NN"

# PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=1 python check_clustering_sound_frame.py --arch ${ARCH}\
 --resume ${resume} --verbose --workers ${WORKERS}
