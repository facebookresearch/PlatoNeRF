#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import OpenEXR, Imath
import codecs
import re
from copy import deepcopy
import numpy as np 
import sys
import matplotlib.pyplot as plt
import json

def compute_extrinsics(origin, lookat):
    """
    Computes 4x4 extrinsic matrix (c2w)
    Based on: https://github.com/mitsuba-renderer/mitsuba2/issues/259
              https://medium.com/@carmencincotti/lets-look-at-magic-lookat-matrices-c77e53ebdf781
    """

    """ Mitsuba --> OpenCV Coordinate System """
    up = [0, 1, 0]
    coord_origin = [0, 0, 0]

    """ List --> NumPy """
    origin = np.array(origin).reshape(3,1)
    lookat = np.array(lookat).reshape(3,1)
    up = np.array(up).reshape(3,1)
    coord = np.array(coord_origin).reshape(3,1)

    """ Rotation """
    zaxis = lookat - origin
    zaxis = zaxis/np.linalg.norm(zaxis)

    xaxis = np.cross(up, zaxis, axis=0)
    xaxis = xaxis/np.linalg.norm(xaxis)

    yaxis = np.cross(zaxis, xaxis, axis=0)
    R = np.array([xaxis, yaxis, zaxis]).squeeze()

    """ Translation """
    xt = np.dot(origin.squeeze(), xaxis.squeeze())
    yt = np.dot(origin.squeeze(), yaxis.squeeze())
    zt = np.dot(origin.squeeze(), zaxis.squeeze())
    T = np.array([xt,yt,zt])

    """ Extrinsic Matrix """
    extrin = np.zeros((4,4))
    extrin[:3,:3] = R
    extrin[:3,3] = T
    extrin[3,:] = np.array([0,0,0,1])
    return extrin

def preprocess(data, name, camloc, camlookat, lightloc, lightlookat, fov):
    """
    Converts Mitsuba's exr format to npy and creates json files with relevant parameters recorded

    data: path to exr to process
    name: output name of npy file
    camloc: csv specifying cam position (e.g. "0,0,0")
    camlookat: csv specifying cam look at (e.g. "0,0,3")
    lightloc: csv specifying light position (e.g. "0,0,0")
    lightlookat: csv specifying light look at (e.g. "0,0,3")
    fov: field of view in degrees
    """
    path = "/".join(data.split("/")[:-1])
    origin = np.array([float(num) for num in camloc.split(",")])
    lookat = np.array([float(num) for num in camlookat.split(",")])
    cam_extrin = compute_extrinsics(origin, lookat)

    """
    Going from: x left, y up, z forward --> OpenGL (x right, y up, z backward)
    """
    cam_extrin[0,:] = cam_extrin[0,:] * -1
    cam_extrin[2,:] = cam_extrin[2,:] * -1
    origin = np.array([float(num) for num in lightloc.split(",")])
    lookat = np.array([float(num) for num in lightlookat.split(",")])

    camera_angle_x = float(fov)

    # Transform to correct coordinate system --> the extrin matrices are transformed in nerf code currently
    origin[0] *= -1
    origin[2] *= -1
    lookat[0] *= -1
    lookat[2] *= -1

    meta = {}
    meta["camera_angle_x"] = camera_angle_x
    meta["frames"] = [None]
    meta["frames"][0] = {
            "file_path": "./train/{}".format(name),
            "transform_matrix": cam_extrin.tolist(),
            "light_origin": origin.tolist(),
            "light_direction": lookat.tolist(),
            "wall_idx": "null"
    }

    with codecs.open(os.path.join(path, "transforms_train_{}.json".format(name)), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4)

    inputFile = OpenEXR.InputFile(data)
    pixelType = Imath.PixelType(Imath.PixelType.HALF)
    dataWin = inputFile.header()['dataWindow']
    imgSize = (dataWin.max.x - dataWin.min.x + 1, dataWin.max.y - dataWin.min.y + 1)
    tmp = list(inputFile.header()['channels'].keys())

    if(len(tmp) != 3):
        prog = re.compile(r"\d+")
        channels = np.array(np.argsort([int(re.match(prog, x).group(0)) for x in tmp], -1, 'stable'))
        channels[0::3], channels[2::3] = deepcopy(channels[2::3]),deepcopy(channels[0::3])
        tmp = np.array(tmp)
        tmp = tmp[list(channels)]
    else:
        tmp = np.array(tmp)
        tmp[0], tmp[2] = tmp[2], tmp[0]

    video = inputFile.channels(tmp, pixelType)
    video = [np.reshape(np.frombuffer(video[i], dtype=np.float16), imgSize) for i in range(len(video))]
    video = np.stack(video, axis=2)
    video = np.stack([video[...,0::3],
                      video[...,1::3],
                      video[...,2::3]], axis=-2)
    image = video.sum(-1)
    np.save(os.path.join(path,"{}.npy".format(name)), video[:,:,0,:])

def replace_line(i, num_files):
    if i < (num_files/2):
        return "            \"wall_idx_cam\": 1,\n            \"wall_idx_light\": 0\n"
    else:
        return "            \"wall_idx_cam\": 0,\n            \"wall_idx_light\": 1\n"

if __name__ == "__main__":
    """
    example: python preprocess_tof.py scenes/chair

    What does this script do?
        1. Converts each exr to npy
        2. Creates a metadata json for each npy
        3. Combines all metadata into a single json called transforms_train.json

    Currently written to assume 24 exrs (light sources); this can be changed by modifying below variable.
    """
    lights = 3 #24
    path = sys.argv[1]
    for i in range(lights):
        if os.path.exists(os.path.join(path,"{}.npy".format(str(i).zfill(3)))):
            print(str(i) + " already exists, continuing!")
            continue
        fp = open(os.path.join(path, "illumination{}.xml".format(str(i).zfill(2))),"r")
        x, y, z = None, None, None
        for line in fp:
            if "target" in line:
                line = line.split("target=\"")[1]
                line = line.split(", ")
                line[-1] = line[-1][:-4]
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
        fp.close()

        preprocess(
                data = os.path.join(path, "frame{}.exr".format(str(i).zfill(2))),
                name = str(i).zfill(3),
                camloc = "0,0,0",
                camlookat = "0,0,3",
                lightloc = "0,0,0",
                lightlookat = "{},{},{}".format(x,y,z),
                fov = 90
        )

    # Combine jsons into a single file
    lines = []
    eod = 42
    for i in range(lights):
        fp_path = os.path.join(path, "transforms_train_{}.json".format(str(i).zfill(3)))
        fp = open(fp_path, "r")
        if i == 0:
            for j, line in enumerate(fp):
                if "wall_idx" in line: 
                    line = replace_line(i, lights)
                if j < eod:
                    lines.append(line)
                if j == eod:
                    lines.append(line.replace("\n","") + ",\n")
        elif i == lights - 1:
            for j, line in enumerate(fp):
                if "wall_idx" in line: 
                    line = replace_line(i, lights)
                if j > 2:
                    lines.append(line)
        else:
            for j, line in enumerate(fp):
                if "wall_idx" in line: 
                    line = replace_line(i, lights)
                if j > 2 and j < eod:
                    lines.append(line)
                if j == eod:
                    lines.append(line.replace("\n","") + ",\n")
        fp.close()
        os.system("rm {}".format(fp_path))

    fp = open(os.path.join(path, "transforms_train.json"), "w")
    for line in lines:
        fp.write(line)
    fp.close()
