#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Implementation is from the original Bounce Flash Lidar repo:
# https://github.com/co24401/BounceFlashLidar

import numpy as np
import matplotlib.pyplot as plt

class grid():
    def __init__(self, nx, ny, nz, minBound, maxBound):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.minBound = minBound
        self.maxBound = maxBound
        
def robust_carving_frame(outside_voxel, inside_voxel, illumination_point, lit_point_list, shadow_point_list):
    # INPUTS:
    #     outside_voxel: cumulative count of number of times voxel lies outside shadow
    #     inside_voxel: cumulative count of number of times voxel lies inside shadow
    #     illumination_point: 3 x 1 array position of illumination point in world coordinate
    #     lit_point_list: N x 3 array containing locations of  pixels not in shadow
    #     shadow_point_list: M x 3 array containing pixels locations of pixels in shadow 
    
    minBound = np.array([outside_voxel.x_lims[0], outside_voxel.y_lims[0], outside_voxel.z_lims[0]]) #transpose
    maxBound = np.array([outside_voxel.x_lims[1], outside_voxel.y_lims[1], outside_voxel.z_lims[1]]) #transpose
    grid3D = grid(outside_voxel.num_x, outside_voxel.num_y, outside_voxel.num_z, minBound, maxBound)
    
    lit_volume = np.zeros((outside_voxel.num_x, outside_voxel.num_y, outside_voxel.num_z))
    shadow_volume = np.zeros((outside_voxel.num_x, outside_voxel.num_y, outside_voxel.num_z))

    line_origin = np.reshape(illumination_point, (3, 1))
    
    for ii in range(lit_point_list.shape[0]): # for each line
        line_destination = np.reshape(lit_point_list[ii, :], (3, 1))
        line_direction = line_destination - line_origin
        line_direction = line_direction / np.linalg.norm(line_direction)
        lit_volume = increment_line(lit_volume, line_origin, line_direction, grid3D)

    for ii in range(shadow_point_list.shape[0]): # for each line
        line_destination = np.reshape(shadow_point_list[ii, :], (3, 1))
        line_direction = line_destination - line_origin
        line_direction = line_direction / np.linalg.norm(line_direction)
        shadow_volume = increment_line(shadow_volume, line_origin, line_direction, grid3D)
    
    outside_voxel.volume = outside_voxel.volume + ((lit_volume > 0) & (shadow_volume == 0))
    inside_voxel.volume = inside_voxel.volume + (shadow_volume > 0) 
    
    return outside_voxel, inside_voxel

def increment_line(volume, origin, direction, grid3D):
    flag, tmin = rayBoxIntersection(origin, direction, grid3D.minBound, grid3D.maxBound)
    if flag == 1:
        if tmin < 0:
            tmin = 0
        
        start = origin + tmin * direction
        boxSize = grid3D.maxBound - grid3D.minBound
        
        x = int(np.floor(((start[0]-grid3D.minBound[0]) / boxSize[0]) * grid3D.nx) + 1)
        y = int(np.floor(((start[1]-grid3D.minBound[1]) / boxSize[1]) * grid3D.ny) + 1)
        z = int(np.floor(((start[2]-grid3D.minBound[2]) / boxSize[2]) * grid3D.nz) + 1)
        
        if x == grid3D.nx+1:
            x -= 1
        if y == grid3D.ny+1:
            y -= 1
        if z == grid3D.nz+1:
            z -= 1
            
        if direction[0] >= 0:
            tVoxelX = x / grid3D.nx
            stepX = 1
        else:
            tVoxelX = (x-1) / grid3D.nx
            stepX = -1
            
        if direction[1] >= 0:
            tVoxelY = y / grid3D.ny
            stepY = 1
        else: 
            tVoxelY = (y-1) / grid3D.ny
            stepY = -1
        
        if direction[2] >= 0:
            tVoxelZ = z / grid3D.nz
            stepZ = 1
        else:
            tVoxelZ = (z-1) / grid3D.nz
            stepZ = -1
            
        voxelMaxX = grid3D.minBound[0] + tVoxelX * boxSize[0]
        voxelMaxY = grid3D.minBound[1] + tVoxelY * boxSize[1]
        voxelMaxZ = grid3D.minBound[2] + tVoxelZ * boxSize[2]
        
        tMaxX = tmin + (voxelMaxX - start[0]) / direction[0]
        tMaxY = tmin + (voxelMaxY - start[1]) / direction[1]
        tMaxZ = tmin + (voxelMaxZ - start[2]) / direction[2]
        
        voxelSizeX = boxSize[0] / grid3D.nx
        voxelSizeY = boxSize[1] / grid3D.ny
        voxelSizeZ = boxSize[2] / grid3D.nz
        
        tDeltaX = voxelSizeX / np.abs(direction[0])
        tDeltaY = voxelSizeY / np.abs(direction[1])
        tDeltaZ = voxelSizeZ / np.abs(direction[2])
        
        while ((x<=grid3D.nx) and (x>=1)) and ((y<=grid3D.ny) and (y>=1)) and ((z<=grid3D.nz) and (z>=1)):
            volume[x-1, y-1, z-1] += 1
            
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    x = x + stepX
                    tMaxX = tMaxX + tDeltaX
                else:
                    z = z + stepZ
                    tMaxZ = tMaxZ + tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    y = y + stepY
                    tMaxY = tMaxY + tDeltaY
                else:
                    z = z + stepZ
                    tMaxZ = tMaxZ + tDeltaZ
                
    return volume

def rayBoxIntersection(origin, direction, vmin, vmax):
        
    if direction[0] >= 0:
        tmin = (vmin[0] - origin[0]) / direction[0]
        tmax = (vmax[0] - origin[0]) / direction[0]
    else:
        tmin = (vmax[0] - origin[0]) / direction[0]
        tmax = (vmin[0] - origin[0]) / direction[0]
        
    if direction[1] >= 0:
        tymin = (vmin[1] - origin[1]) / direction[1]
        tymax = (vmax[1] - origin[1]) / direction[1]
    else:
        tymin = (vmax[1] - origin[1]) / direction[1]
        tymax = (vmin[1] - origin[1]) / direction[1]
    
    if ((tmin > tymax) or (tymin > tmax)):
        flag = 0
        tmin = -1
        return flag, tmin
    
    if (tymin > tmin):
        tmin = tymin
    
    if (tymax < tmax):
        tmax = tymax
    
    if direction[2] >= 0:
        tzmin = (vmin[2] - origin[2]) / direction[2]
        tzmax = (vmax[2] - origin[2]) / direction[2]
    else:
        tzmin = (vmax[2] - origin[2]) / direction[2]
        tzmax = (vmin[2] - origin[2]) / direction[2]
    
    if ((tmin > tzmax) or (tzmin > tmax)):
        flag = 0
        tmin = -1
        return flag, tzmin
        
    if (tzmin > tmin):
        tmin = tzmin
    
    if (tzmax < tmax):
        tmax = tzmax
    
    flag = 1
        
    return flag, tmin

def visualize_probablistic(inside_voxel, outside_voxel, PO, T, testLas, testDets, generate_plot):
    inside_volume = inside_voxel.volume
    outside_volume = outside_voxel.volume
    volume = np.zeros_like(inside_volume)
    probability_volume = np.zeros_like(inside_volume)
    testVar = []
    for ii in range(inside_volume.shape[0]):
        for j in range(inside_volume.shape[1]):
            for k in range(inside_volume.shape[2]):
                idx1 = int(inside_volume[ii, j, k])
                idx2 = int(outside_volume[ii, j, k])
                a = PO[idx1, idx2]
                testVar.append(a)
                probability_volume[ii, j, k] = a
                if PO[idx1, idx2] > T:
                    volume[ii,j,k] = 1
    plt.hist(testVar, 100)        
    
    if generate_plot and True:
        voxel_x_size = (inside_voxel.x_lims[1] - inside_voxel.x_lims[0]) / inside_voxel.num_x
        voxel_y_size = (inside_voxel.y_lims[1] - inside_voxel.y_lims[0]) / inside_voxel.num_y
        voxel_z_size = (inside_voxel.z_lims[1] - inside_voxel.z_lims[0]) / inside_voxel.num_z
        fill_indices = np.nonzero(volume)
        
        xx = inside_voxel.x_lims[0] + voxel_x_size * fill_indices[0]
        yy = inside_voxel.y_lims[0] + voxel_y_size * fill_indices[1]
        zz = inside_voxel.z_lims[0] + voxel_z_size * fill_indices[2]
        
        
        testShape = np.transpose(np.stack((xx, yy, zz), axis=0))    
        ptCloud = np.vstack((testShape, testLas, testDets))
        
#         plotly.offline.init_notebook_mode()
#         # Configure the trace.
#         det_plot_pts = go.Scatter3d(
#             x=np.ndarray.flatten(det_locs[:, :, 0]),  # <-- Put your data instead
#             y=np.ndarray.flatten(det_locs[:, :, 1]),  # <-- Put your data instead
#             z=np.ndarray.flatten(det_locs[:, :, 2]),  # <-- Put your data instead
#             mode='markers',
#             marker={
#                 'size': 5,
#                 'opacity': 1,
#             }
#         )

#         las_plot_pts = go.Scatter3d(
#             x=np.ndarray.flatten(las_locs[:, :, 0]),  # <-- Put your data instead
#             y=np.ndarray.flatten(las_locs[:, :, 1]),  # <-- Put your data instead
#             z=np.ndarray.flatten(las_locs[:, :, 2]),  # <-- Put your data instead
#             mode='markers',
#             marker={
#                 'size': 5,
#                 'opacity': 1,
#             }
#         )

#         testShape = go.Scatter3d(
#             x=np.ndarray.flatten(testShape[:, 0]),  # <-- Put your data instead
#             y=np.ndarray.flatten(testShape[:, 1]),  # <-- Put your data instead
#             z=np.ndarray.flatten(testShape[:, 2]),  # <-- Put your data instead
#             mode='markers',
#             marker={
#                 'size': 5,
#                 'opacity': 1,
#             }
#         )




#         # Configure the layout.
#         layout = go.Layout(
#             margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
#         )
#         data = [det_plot_pts, las_plot_pts, testShape]
#         plot_figure = go.Figure(data=data, layout=layout)

#         # Render the plot.
#         plotly.offline.iplot(plot_figure)

        # Visualize point cloud (WARNING: This will open another window and you will be forced to kill kernal)
        # if True:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(ptCloud)
        #     o3d.io.write_point_cloud("../../data/multiplexed_twobounce_021322/result.ply", pcd)
        #     cloud = o3d.io.read_point_cloud("../../data/multiplexed_twobounce_021322/result.ply") # Read the point cloud
        #     o3d.visualization.draw_geometries([cloud])

    return volume, probability_volume
