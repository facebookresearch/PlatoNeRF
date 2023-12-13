#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Implementation is based on original Bounce Flash Lidar repo:
# https://github.com/co24401/BounceFlashLidar

import numpy as np
from math import sqrt, cos, sin, tan, atan, radians
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
from copy import deepcopy
import math
import time
import os
import sys

from ray_tracing import robust_carving_frame

c = 3E8

# === Data Parameters === #
numSpots = 16
num_u = 512
num_v = 512
numBins = 391
fov_x = 90 
fov_y = 90 
bin_width = 0.0384 / c # picoseconds
x_c = np.array([0, 0, 0])
x_l = np.array([0, 0, 0])

output_dir = "logs"
files = ["chair", "bunny", "dragon", "occlusion"]
fname = sys.argv[1]
if fname not in files:
    print(fname, "not in files!")
    exit()
num_vox = int(sys.argv[2])

fin_pt_cloud = np.load(os.path.join(output_dir, "bfpc_" + sys.argv[1] + ".npy"), allow_pickle=True).item()["pc"]
laser_pos = np.load("logs/bflaserpos_{}.npy".format(fname))
shadow_masks = np.load("logs/bfshadowmasks_{}.npy".format(fname))

lit_pts = []; shadow_pts = []
illum_pts = laser_pos

count = 0
for i in range(numSpots):
    lit_i = []
    shadow_i = []
    for x_ii in range(num_u):
        for y_ii in range(num_v):
            point = np.array(fin_pt_cloud[y_ii][x_ii])
            if np.isnan(point).any():
                count += 1
                continue

            binary = shadow_masks[i][y_ii][x_ii]
            if binary:
                lit_i.append(point)
            else:
                shadow_i.append(point)
    lit_i = np.array(lit_i)
    shadow_i = np.array(shadow_i)
    lit_pts.append(lit_i)
    shadow_pts.append(shadow_i)

    print("Total NaNs", count)
    count = 0



# IMPLEMENT BOUNCE_FLASH_ROBUST_CARVING_SCRIPT
class voxelGrid():
    def __init__(self, x_lims, y_lims, z_lims, num_x, num_y, num_z):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims
        self.num_x = num_x
        self.num_y = num_y
        self.num_z = num_z
        self.volume = np.zeros((num_x, num_y, num_z))

# discretize hidden space

x_lims = np.array([-2, 2])
y_lims = np.array([-2, 2])
z_lims = np.array([1.3, 4])

num_x = num_vox
num_y = num_vox
num_z = num_vox
outside_voxel = voxelGrid(x_lims, y_lims, z_lims, num_x, num_y, num_z)
inside_voxel = voxelGrid(x_lims, y_lims, z_lims, num_x, num_y, num_z)

# count number of times lit and shadow pixel intersects voxel
print("Commence Robust Carving")
start_time = time.time()
for i in range(len(lit_pts)):
    # lit_pts = 17 x 1 list (contains pixels not in shadow for each light source)
    # shadow_pts = 17 x 1 list (contains pixels in shadow for each light source)
    # illum_pts = 17 x 3 double (contains locations of light sources)
    outside_voxel, inside_voxel = robust_carving_frame(outside_voxel, inside_voxel, illum_pts[i, :], 
                                                         lit_pts[i], shadow_pts[i])
    print(i)
end_time = time.time()
print(f'Run time: {end_time-start_time} seconds')

eta = 0.05
xi = 0.5 # Probability that an empty voxel is traced to shadow (probability false alarm) --> higher = overcarve
p0 = 0.8 # Prior probability that any voxel is empty
p1 = 0.2 # Prior probability that any voxel is occupied
T = 0.3 # Probability threshold to decide that voxel is occupied --> usually only play with T

m = np.linspace(0, len(lit_pts), len(lit_pts)+1)
n = np.linspace(0, len(lit_pts), len(lit_pts)+1)
[M, N] = np.meshgrid(m, n)
PO = p1 * (eta**M) * ((1-eta)**N)/(p0*((1-xi)**M)*(xi**N) + p1*(eta**M)*((1-eta)**N))

testDets = np.zeros((num_u*num_v, 3))
testLas = np.zeros((numSpots, 3))

save_dict = {"outside_voxel": outside_voxel,
             "inside_voxel": inside_voxel,
             "PO": PO,
             "testDets": testDets,
             "testLas": testLas,
             "params": np.array([num_x,num_y,num_z,eta,xi,p0,p1])}

np.save("logs/bfshadows_{}_{}.npy".format(fname, num_x), save_dict)
