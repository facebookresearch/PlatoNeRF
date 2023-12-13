#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Several functions in this codebase were repurposed from nerf-pytorch 
# https://github.com/yenchenlin/nerf-pytorch
# nerf-pytorch is released under the MIT license.

import cv2
import imageio
import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from utils.load_tof import load_tof_data
from utils.nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, debug_title="", per_ray_far=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], debug_title=debug_title, per_ray_far=per_ray_far, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(rays,
           chunk=1024*32,
           c2w=None,
           ndc=True,
           near=0.,
           far=6.,
           use_viewdirs=False,
           c2w_staticcam=None,
           debug_title="",
           per_ray_far=None,
           **kwargs):
    """Render rays
    Args:
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      intensity_map: [batch_size, 1]. Predicted intensity values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    # use provided ray batch
    rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, debug_title=debug_title, per_ray_far=per_ray_far, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['intensity_map', 'disp_map', 'acc_map', 'depth_map', 'trans']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, intensity=True).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    intensity = torch.sigmoid(raw[...,0])  # [N_rays, N_samples, 3]
    intensity = intensity[...,None]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,1] + noise, dists)  # [N_rays, N_samples]
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    #weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    intensity_map = torch.sum(weights[...,None] * intensity, -2)  # [N_rays, 3]
    final_trans = transmittance[:,-2]

    """ compute distance along ray to surface """

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return intensity_map, disp_map, acc_map, weights, depth_map, final_trans


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                debug_title="",
                per_ray_far=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if per_ray_far is not None:
        far = per_ray_far[:,None]
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn)
    intensity_map, disp_map, acc_map, weights, depth_map, trans = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        intensity_map_0, disp_map_0, acc_map_0, depth_map_0, trans_0 = intensity_map, disp_map, acc_map, depth_map, trans

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        intensity_map, disp_map, acc_map, weights, depth_map, trans = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'intensity_map' : intensity_map, 
            'disp_map' : disp_map, 
            'acc_map' : acc_map, 
            'depth_map' : depth_map, 
            'trans' : trans, 
            'pts' : pts}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['intensity0'] = intensity_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['depth0'] = depth_map_0
        ret['trans0'] = trans_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--output_dir', type=str, default='./', help='where to save test depth results')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--use_all_rays", type=int, default=0,
                        help='whether or not to use all rays')
    parser.add_argument("--per_image_thresh", type=float, action='append', required=False)
    parser.add_argument("--debug", type=int, default=0,
                        help='whether or not to debug')
    parser.add_argument("--near", type=float, default=0.0,
                        help='near plane')
    parser.add_argument("--dist_weight", type=int, default=1000,
                        help='dist weight')
    parser.add_argument("--extract_first", type=int, default=0,
                        help='whether or not to extract 1b distance to compute projected illumination')
    parser.add_argument("--parallel", type=float, default=0.05,
                        help='parallel filter')
    parser.add_argument("--shadw", type=float, default=1.0,
                        help='shadow loss weight for shadow pixels')
    parser.add_argument("--nonshadw", type=float, default=1.0,
                        help='shadow loss weight for non shadow pixels')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--downsample", type=int, default=1,
                        help='downsample rays by factor of x')
    parser.add_argument("--downsample_temp", type=int, default=0,
                        help='downsample rays by factor of x')

    # training options
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian noise on time of arrival")
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin'),
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging'),
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving'),
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving'),
    parser.add_argument("--simple_mode", action='store_true',
                        help='whether or not to only render 1B to 2B rays'),
    parser.add_argument("--sigmoid", action='store_true',
                        help='whether or not to apply sigmoid on shadows'),
    parser.add_argument("--bce", action='store_false',
                        help='whether or not to use BCE loss'),

    return parser

def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((4, 4))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = -z

    view_mat[:3, 3] = vec_pos
    view_mat[3,:] = np.array([0.,0.,0.,1.])

    return view_mat

def compute_points_around_circle(origin, radius, num_points, start_angle):
    angles = np.linspace(start_angle, start_angle + 2*np.pi, num_points, endpoint=False)
    x_coords = origin[0] + radius * np.cos(angles)
    y_coords = origin[1] + radius * np.sin(angles)
    points = np.column_stack((x_coords, y_coords))
    return points

def train():

    parser = config_parser()
    args = parser.parse_args()

    # FOR TESTING ONLY
    args.N_samples = 1024
    batch_size = 512

    output_dir = os.path.join(args.output_dir, "depth_predictions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    K = None
    if args.dataset_type == "dtof":
        tof, poses, light_o, light_d, hwf, walls_cam, walls_light = load_tof_data(args.datadir)
        print('Loaded ToF data', tof.shape, light_o.shape, light_d.shape, hwf, args.datadir, walls_cam.shape, walls_light.shape)
        render_poses = torch.tensor([0.0]).float()
        i_train = np.arange(tof.shape[0])
        i_val = []
        i_test = []

        print("Train idxs: {}".format(i_train))

        near = args.near
        far = 6.0
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    from copy import deepcopy
    Nframes = 40
    render_poses = []
    for i in range(Nframes):
        rpose = deepcopy(poses[0])
        rpose[2,3] += 0.06282151815625667 * i
        rpose[2,:] *= -1
        rpose[0,:] *= -1

        rpose[1,3] = -1.5

        if i == Nframes-1:
            continue
        render_poses.append(torch.Tensor(rpose))

    # Compute poses of camera moving in a circle around object at (0,0,-3)
    origin = (0, 3)  # Center of the circle
    radius = 0.99    # Radius of the circle
    num_points = 100 # Number of points around the circle
    points = compute_points_around_circle(origin, radius, num_points, -np.pi/2)

    y = -1.5
    lookat = np.array([0,y,-3])
    for point in points:
        origin = np.array([-point[0], y, -point[1]])
        cam_extrin = look_at(origin, lookat)
        render_poses.append(torch.Tensor(cam_extrin))

    render_poses = torch.stack(render_poses, 0)

    rgbs = []
    for pose_i, pose in enumerate(render_poses):
        if os.path.exists(os.path.join(output_dir, "depth_map_{}.png".format(str(pose_i).zfill(3)))):
            print("{} exists, continuing".format(pose_i))
            continue
        print("{} of {}".format(pose_i, len(render_poses)))
        pose = pose.unsqueeze(0).detach().cpu().numpy()

        rays = np.stack([get_rays_np(H, W, K, p) for p in pose[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays = np.transpose(rays, [0,2,3,1,4]) # [N, H, W, ro+rd+lo+ld, 3]
        rays = rays.astype(np.float32)[0]
        rays = torch.Tensor(rays).to(device)
        rays = rays.reshape(-1, rays.shape[2], rays.shape[3])

        depth_map = torch.zeros(rays.shape[0])
        for i in range(0, rays.shape[0], batch_size):
            batch_rays = rays[i:i+batch_size]
            batch_rays = torch.transpose(batch_rays, 0, 1)
            intensity, disp, acc, depth, trans, extras = render(batch_rays, chunk=args.chunk,
                                                  verbose=i < 10, retraw=True, debug_title="0_bounce",
                                                  **render_kwargs_train)
            depth_map[i:i+batch_size] = depth.detach().cpu()

        depth_map = depth_map.reshape(512,512)
        depth_map = depth_map.detach().cpu().numpy()
        np.save(os.path.join(output_dir, "depth_map_{}.npy".format(str(pose_i).zfill(3))), depth_map)
        depth_map = depth_map / 4.65

        depth_map = np.stack([depth_map, depth_map, depth_map],axis=2)
        depth_map = np.float32(depth_map)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, "depth_map_{}.png".format(str(pose_i).zfill(3))),depth_map*255)
        rgbs.append(depth_map)

    imageio.mimwrite(os.path.join(output_dir,'video.mp4'), to8b(rgbs), fps=15, quality=8)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
