#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Download pre-trained checkpoints for each scene
wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_ckpts.zip
unzip platonerf_ckpts.zip
rm platonerf_ckpts.zip

# Download datasets
wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_chair.zip
unzip platonerf_chair.zip
rm platonerf_chair.zip

wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_bunny.zip
unzip platonerf_bunny.zip
rm platonerf_bunny.zip

wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_dragon.zip
unzip platonerf_dragon.zip
rm platonerf_dragon.zip

wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_occlusion.zip
unzip platonerf_occlusion.zip
rm platonerf_occlusion.zip

wget https://github.com/facebookresearch/PlatoNeRF/releases/download/v0/platonerf_real.zip
unzip platonerf_real.zip
rm platonerf_real.zip

echo "Datasets were downloaded to './data' and checkpoints were downloaded to './pretrained'!"
