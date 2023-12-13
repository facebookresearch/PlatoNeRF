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

# === Import histogram data === #
output_dir = "logs"
files = ["chair", "bunny", "dragon", "occlusion"]
for fname in files:
    filedir = './data/{}/train/'.format(fname)
    print("Processing {}".format(fname))
    hists = {}
    count = 0
    for i in range(24):
        if i in [0,3,7,10,13,14,19,23]: continue
        filename = f'{filedir}{i:03d}.npy'
        a = np.load(filename)
        hists[count] = a
        count += 1

    # === Import camera matrices === #
    f_x = (num_u/2) / tan(radians(fov_x/2))
    f_y = (num_v/2) / tan(radians(fov_y/2))
    c_x = num_u // 2
    c_y = num_v // 2
    K = np.array([[f_x,  0 ,  c_x , 0],
                  [ 0 , f_y, c_y, 0],
                  [ 0 ,  0 ,  1 , 0]]) # intrinsic matrix
    R  =  np.array([[1 , 0 , 0,  0  ],
                    [0 , 1 , 0,  0.0],
                    [0 , 0 , 1, 0.0],
                    [0 , 0 , 0 , 1  ]])

    cam_matrix = np.matmul(K, R) # mapping from world coordinates to pixel coordinates (in homogeneous coordinates)
    cam_matrix_inv = np.linalg.inv(cam_matrix[:, 0:3]) # mapping from pixel coordinates (in homogeneous coordinates) to world coordinates (in homogeneous coordinates)

    def create_ray_visualizer(rays):
        num_rows = rays.shape[1]
        num_cols = rays.shape[2]
        num_rays = num_rows * num_cols
        rays_vis = np.zeros((100*num_rays, 4))
        i = 0
        for iy in range(num_rows):
            for ix in range(num_cols):
                rays_z = np.linspace(0, 3, 100)
                rays_x = rays[0, iy, ix] * rays_z
                rays_y = rays[1, iy, ix] * rays_z
                rays_c = np.vstack([rays_x, rays_y, rays_z])
                rays_vis[i*100:(i+1)*100, 0] = i + 1
                rays_vis[i*100:(i+1)*100, 1:] = np.transpose(rays_c)
                i += 1
        rays_vis = pd.DataFrame(data=rays_vis, columns=["ray", "x", "y", "z"])
        return rays_vis

    def plot_rays_walls(rays_vis):
        # plot ray and wall
        lines = px.line_3d(rays_vis, x="x", y="y", z="z")
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        camera = dict(eye=dict(x=-1., y=0., z=-2.5), up=dict(x=0, y=1., z=0))
        plot_figure = go.Figure(data=lines, layout=layout)
        plot_figure.update_layout(scene_camera=camera)
        plotly.offline.iplot(plot_figure)

    # === Obtain rays per pixel (normalized to unit length) === #
    x_grid, y_grid = np.meshgrid(np.linspace(1, num_u, num_u), np.linspace(1, num_v, num_v))
    x_grid_flat = np.reshape(x_grid, [num_u*num_v], order='C')
    y_grid_flat = np.reshape(y_grid, [num_u*num_v], order='C')
    pix_coords = np.stack([x_grid_flat, y_grid_flat, np.ones([num_u*num_v])], axis=0)
    rays = np.matmul(cam_matrix_inv, pix_coords)
    rays = rays / np.expand_dims(np.linalg.norm(rays, axis=0), axis=0)
    rays = np.reshape(rays, (3, num_u, num_v), order='C')
    rays = np.flip(rays, axis=1)
    rays = np.flip(rays, axis=2)
    # === Visualize rays for debugging purposes === #
    debug = 0
    if debug:
        rays_vis = create_ray_visualizer(rays[:, 0:-1:32, 0:-1:64])
        plot_rays_walls(rays_vis)

        plt.figure()
        _, (plot1, plot2) = plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        for iy in range(0, rays.shape[1], 512):
            for ix in range(0, rays.shape[2], 100):
                x = np.linspace(0, 5, 100) * rays[0, iy, ix]
                y = np.linspace(0, 5, 100) * rays[1, iy, ix]
                plt.plot(x, y)
        plt.title('X-Y Slice')
        plt.subplot(1, 2, 2)
        for iy in range(0, rays.shape[1], 512):
            for ix in range(0, rays.shape[2], 100):
                x = np.linspace(0, 5, 100) * rays[0, iy, ix]
                z = np.linspace(0, 5, 100) * rays[2, iy, ix]
                plt.plot(x, z)
        plot2.invert_xaxis()
        plt.title('X-Z Slice')

    # === Obtain Intensity Image from Histograms === #
    pkbins = np.zeros((numSpots, num_u, num_v))
    Evals = np.zeros((numSpots, num_u, num_v))
    plt.figure()
    for i in range(numSpots):
        pk_idx = np.argmax(hists[i], axis=-1) + 1
        pkbins[i, :, :] = pk_idx
        E = np.sum(hists[i], axis=-1)
        Evals[i, :] = E
        plt.subplot(4, 4, i+1)

    # === Process single bounce returns === #
    ### NOTE: THIS SECTION ASSUMES CO-LOCATED LASER AND DETECTOR ###
    print('Processing single-bounce returns')
    laser_pos = np.zeros((numSpots, 3))
    pulses_1B = np.zeros((numSpots, numBins))
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(4, 4)
    fig.tight_layout(pad=1.0)
    for ii in range(numSpots):
        # Pixel with max E val assumed to be laser spot
        img = deepcopy(Evals[ii, :, :])
        idx = np.unravel_index(np.argmax(img, axis=None), (num_u, num_v)) 
        
        # compute ray direction of pixel corresponding to 1B 
        ray_1B = rays[:, idx[0], idx[1]]
        
        # Estimate 1B ToF (can use matched filter to improve estimate)
        hist_1b = hists[ii][idx[0], idx[1], :]
        pulses_1B[ii] = hist_1b
        t1 = np.argmax(hist_1b) * bin_width
        
        # Extract illuminated position
        laser_pos[ii] = ray_1B * (c*t1/2)
        
        # Plot image and laser location
        plt.subplot(4, 4, ii+1)
        img_plot = np.log10(img)
        plt.plot(idx[1], idx[0], 'or')
        plt.title(str(np.round(ray_1B, 1)), size=10)
        plt.imsave(os.path.join(output_dir, "laser_spots.png".format(fname)), img_plot)

    np.save("logs/bflaserpos_{}.npy".format(fname), laser_pos)
    print("laser pos is saved!")

    # === Get pulse template === #
    # Choose pulse index
    p_idx = 0
    pulse_template = pulses_1B[p_idx] / np.sum(pulses_1B[p_idx])
    # Compute pulse FFT
    pulse_fft = np.fft.fft(pulse_template)
    t0 = np.argmax(pulse_template)-1
    pulse_template = np.roll(pulse_template, -t0)

    start_time = time.time()
    # === Process two bounce returns === #
    pulse_fft = np.expand_dims(np.fft.fft(pulse_template), axis=[0, 1])
    tof_2b = np.zeros((numSpots, num_u, num_v))
    shadow_masks = np.zeros((numSpots, num_u, num_v))
    for ii in range(numSpots):
        # compute match filter similarity
        img = hists[ii]
        corr = np.abs(np.fft.ifft(pulse_fft * np.fft.fft(img, axis=-1), axis=-1))
        # compute confidence and tof
        tof = bin_width * np.argmax(corr, axis=-1)
        confidence = np.max(corr, axis=-1)
        # determine shadows pixels by thresholding confidence
        in_shadow = (np.log10(confidence) > -2.5)
        # Store values
        tof_2b[ii, :, :] = tof
        shadow_masks[ii, :, :] = in_shadow
    end_time = time.time()
    print(f'Run time: {end_time-start_time} seconds')

    print("Saving shadow masks!")
    np.save("logs/bfshadowmasks_{}.npy".format(fname), shadow_masks)

    class Ellipse:
        def __init__(self, tof, x_illum, camera_params):
            """
            INPUTS:
                tof = total 2B tof
                x_c = camera_location (focus 1)
                x_illum = laser illumination point (focus 2)
                camera_params = camera/laser location
            """
            # === Ellipse Parameters === #
            # Focus vectors and length
            self.f_1 = camera_params.x_c # focus 1
            self.f_2 = x_illum # focus 2
            self.f_vec = (self.f_2 - self.f_1) / 2
            self.f = np.linalg.norm(self.f_vec) # focus length

            # Major and Minor axis
            self.a = (c*tof - np.linalg.norm(x_illum - camera_params.x_l)) / 2 # major axis
            self.skip = True
            if self.a**2 - self.f**2 > 0:
                self.b = sqrt(self.a**2 - self.f**2) # minor axis

                # === Compute Transformations Parameters === #
                self.center = (self.f_1 + self.f_2) / 2
                self.t_xz = atan(abs(self.f_vec[2] / self.f_vec[0])) # NOTE: Assumes z > 0
                if self.f_vec[0] < 0:
                    self.t_xz = math.pi - self.t_xz
                new_x_mag = np.linalg.norm([self.f_vec[0], self.f_vec[2]])
                self.t_xy = -atan(self.f_vec[1] / new_x_mag)

                # === Compute Unit Sphere Transformation Matrix === #
                # Rotation matrices
                rot_xz = np.array([[cos(self.t_xz) , 0, sin(self.t_xz)],
                                   [      0        , 1,        0      ],
                                   [-sin(self.t_xz), 0, cos(self.t_xz)]])
                rot_xy = np.array([[cos(self.t_xy), -sin(self.t_xy), 0],
                                   [sin(self.t_xy),  cos(self.t_xy), 0],
                                   [       0      ,        0       , 1]])

                # Scaling matrix
                scale_matrix = np.array([[(1/self.a),      0    ,     0     ],
                                         [  0       , (1/self.b),     0     ],
                                         [  0       ,      0    , (1/self.b)]])
                self.T = scale_matrix @ rot_xy @ rot_xz
                self.skip = False

        def compute_ray_ellipse_intersection(self, ray):
            if abs(np.linalg.norm(ray) - 1.0) > 1E-5:
                raise AssertionError("Input rays should be normalized to unit length")
            rho = self.compute_rho(ray)
            return rho * ray

        def compute_rho(self, x_dir):
            m1 = self.T @ x_dir
            m2 = self.T @ self.center
            a = np.linalg.norm(m1)**2
            b = -2*np.dot(m1, m2)
            c = np.linalg.norm(m2)**2 - 1
            return (-b + sqrt(b**2-4*a*c)) / (2*a) 

    class Camera:
        def __init__(self, x_c, x_l):
            self.x_c = x_c
            self.x_l = x_l
    start_time = time.time()
    # === Compute point cloud === #
    camera_params = Camera(x_c, x_l)
    pt_cloud = np.zeros((numSpots, num_u, num_v, 3))
    for n in range(numSpots):
        x_illum = laser_pos[n, :]
        for iy in range(num_u):
            for ix in range(num_v):
                tof = tof_2b[n, iy, ix]
                ellipse_data = Ellipse(tof, x_illum, camera_params)
                if ellipse_data.skip or not shadow_masks[n, iy, ix]:
                    pt_cloud[n, iy, ix, :] = None
                    continue
                ray_pix = rays[:, iy, ix]
                pt_cloud[n, iy, ix, :] = ellipse_data.compute_ray_ellipse_intersection(ray_pix)
    end_time = time.time()
    print(f'Run time: {end_time-start_time} seconds')

    fin_pt_cloud_ = np.nanmedian(pt_cloud, axis=0)
    fin_pt_cloud = np.reshape(fin_pt_cloud_, (num_u*num_v, 3))

    data = {"pc": fin_pt_cloud_,
            "flattened": fin_pt_cloud}
    np.save("logs/bfpc_{}.npy".format(fname), data)
