#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import json
import scipy.io

def load_tof_data(basedir, ignore=[]):
    splits = ['train']#, 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_tof = []
    all_cam_poses = []
    all_walls_cam = []
    all_walls_light = []
    all_light_o = []
    all_light_d = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        walls = meta['walls']
        for i, frame in enumerate(meta['frames']):
            if i in ignore:
                print("Not loading ToF image {}.".format(i))
                continue
            fname = os.path.join(basedir, frame['file_path'] + '.npy')
            wall_cam = walls[frame['wall_idx_cam']]
            wall_light = walls[frame['wall_idx_light']]
            tof = np.expand_dims(np.load(fname),0)
            cam_pose = np.expand_dims(np.array(frame['transform_matrix'], dtype=np.float32), 0)
            light_o = np.expand_dims(np.array(frame['light_origin'], dtype=np.float32), 0)
            light_d = np.expand_dims(np.array(frame['light_direction'], dtype=np.float32), 0)
            light_d = light_d / np.linalg.norm(light_d) # normalize light ray direction

            all_tof.append(tof)
            all_cam_poses.append(cam_pose)
            all_walls_cam.append(np.array(wall_cam['plane'] + wall_cam['point'] + [wall_cam['x']] + [wall_cam['y_min']] + [wall_cam['y_max']] + [wall_cam['z_min']] + [wall_cam['z_max']]))
            all_walls_light.append(np.array(wall_light['plane'] + wall_light['point'] + [wall_light['x']] + [wall_light['y_min']] + [wall_light['y_max']] + [wall_light['z_min']] + [wall_light['z_max']]))
            all_light_o.append(light_o)
            all_light_d.append(light_d)

    tof = np.concatenate(all_tof, 0)
    cam_poses = np.concatenate(all_cam_poses, 0)
    light_origins = np.concatenate(all_light_o, 0)
    light_directions = np.concatenate(all_light_d, 0)
    all_walls_cam = np.stack(all_walls_cam, 0) # batch x 11, where 11 = plane, point, x, y min, y max, z min, z max
    all_walls_light = np.stack(all_walls_light, 0) # batch x 11, where 11 = plane, point, x, y min, y max, z min, z max
    
    H, W = tof.shape[1:3]
    camera_angle_x = np.radians(float(meta['camera_angle_x']))
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return tof, cam_poses, light_origins, light_directions, [H, W, focal], all_walls_cam, all_walls_light


def load_real(basedir, res=200, num_lights=16, bin_width=8e-12, light_position=np.array([0.257, 0, 0])):
    tdata = scipy.io.loadmat(os.path.join(basedir, 't2.mat'))
    ddata = scipy.io.loadmat(os.path.join(basedir, 'detector_scan_200x200.mat'))
    sdata = scipy.io.loadmat(os.path.join(basedir, 'two_bounce_detections.mat'))

    # Load relevant parameters
    t2 = tdata['t2'] # two bounce time of flight (in seconds)
    phiMap = ddata['phiMap'] # angles for rays
    thetaMap = ddata['thetaMap'] # angles for rays
    has_2B = sdata['has_2B'] # pre-processed shadow map
    illum_pts = sdata['illum_pts'][:num_lights] # world coordinates of illumination

    # Reshape and flip all data
    t2 = np.reshape(t2, [t2.shape[0], res, res])
    phiMap = np.reshape(phiMap, [res, res])
    thetaMap = np.reshape(thetaMap, [res, res])
    has_2B = np.reshape(has_2B, [has_2B.shape[0], res, res])
    t2 = np.flip(t2, 1)
    phiMap = np.flip(phiMap, 0)
    thetaMap = np.flip(thetaMap, 0)
    has_2B = np.flip(has_2B, 1)

    # Compute rays
    rays = np.zeros([3, res, res])
    point = np.array([1,0,0])
    for i in range(res):
        for j in range(res):
            theta = thetaMap[i][j]
            phi = phiMap[i][j]
            rays[:,i,j] = np.array([np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)])
    rays = np.moveaxis(rays, 0, -1)

    light_position = np.repeat(light_position[np.newaxis,:], num_lights, axis=0)
    rays = np.repeat(rays[np.newaxis, :, :, :], num_lights, axis=0)
    rays = np.stack([np.zeros(rays.shape), rays], axis=1)

    t2 = np.nan_to_num(t2)
    has_2B = np.nan_to_num(has_2B)
    illum_pts = np.nan_to_num(illum_pts)
    rays = np.nan_to_num(rays)

    return t2, has_2B, illum_pts, rays, bin_width, light_position
