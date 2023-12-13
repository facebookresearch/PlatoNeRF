#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import sys
import mitsuba as mi
import numpy as np
from copy import deepcopy

mi.set_variant("scalar_rgb")

ORIGIN = [0.,0.,0.] # Make sure the rgb.xml file has camera origin at "0.0, 0.0, 0.0" before running!
LOOKAT = np.array([0,-1.5,3])
MAX_DEPTH = 5.6541066 # used for normalizing depth images for visualization

def compute_points_around_circle(origin, radius, num_points, start_angle):
    angles = np.linspace(start_angle, start_angle + 2*np.pi, num_points, endpoint=False)
    x_coords = origin[0] + radius * np.cos(angles)
    y_coords = origin[1] + radius * np.sin(angles)
    points = np.column_stack((x_coords, y_coords))
    return points

""" Compute origins and lookats """

origins = []
lookats = []

Nframes = 40
#Nframes = 5
render_poses = []
for i in range(Nframes):
    origin = deepcopy(ORIGIN)
    origin[2] += 0.06282151815625667 * i
    origin[1] = -1.5

    if i == Nframes-1: 
        continue
    origins.append(origin)

# Compute poses of camera moving in a circle around object at (0,0,-3)
origin = (0, 3)   # Center of the circle
radius = 0.99     # Radius of the circle
num_points = 100  # Number of points around the circle
points = compute_points_around_circle(origin, radius, num_points, -np.pi/2)

y = -1.5
for point in points:
    origin = np.array([point[0], y, point[1]])
    origins.append(origin)

print("Total cameras to process: {}".format(len(origins)))

""" Rendering """
current = ", ".join([str(o) for o in ORIGIN])

for i, origin in enumerate(origins):
    print("Rendering {}".format(i))
    origin = ", ".join([str(o) for o in origin])
    os.system("sed -i \'s/origin=\"{}\"/origin=\"{}\"/g\' {}".format(current, origin, sys.argv[1]))
    current = origin

    scene = mi.load_file(sys.argv[1])
    data = mi.render(scene, spp=256)
    image = data[:,:,:3]
    depth = data[:,:,3]
    mi.util.write_bitmap(os.path.join(sys.argv[2],"{}.png".format(str(i).zfill(4))), image)
    np.save(os.path.join(sys.argv[2], "depth_{}.npy".format(str(i).zfill(4))), depth)
    depth = depth / MAX_DEPTH
    depth_map = np.stack([depth, depth, depth],axis=2)
    depth_map = np.float32(depth_map)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(sys.argv[2], "depth_map_{}.png".format(str(i).zfill(4))),depth_map*255)

current = ", ".join([str(o) for o in ORIGIN])
os.system("sed -i \'s/origin=\"{}\"/origin=\"{}\"/g\' {}".format(origin, current, sys.argv[1]))
