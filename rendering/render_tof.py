#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import sys

points = 24
dirs = sys.argv[1]

current = ""
target = ""
for i in range(points):
    illum_src = os.path.join(dirs, "illumination{}.xml".format(str(i).zfill(2)))
    illum_tgt = os.path.join(dirs, "illumination.xml".format(str(i).zfill(2)))
    os.system("cp {} {}".format(illum_src, illum_tgt))
    os.system("mitsuba {}/frame.xml -D samples=32 -D decomposition=transient -D tMin=0 -D tMax=15 -D tRes=0.0384 -D modulation=none -D lambda=200 -D phase=0".format(dirs))
    os.system("mv {}/frame.exr {}/frame{}.exr".format(dirs, dirs, str(i).zfill(2)))
