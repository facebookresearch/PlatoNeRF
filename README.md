<div align="center">
  <h1>PlatoNeRF: 3D Reconstruction in Plato's Cave via Single-View Two-Bounce Lidar</h1>

  <p style="font-size:1.2em">
    <a href="https://tzofi.github.io/"><strong>Tzofi Klinghoffer</strong></a> ·
    <a href="https://engineering.purdue.edu/people/xiaoyu.xiang.1"><strong>Xiaoyu Xiang</strong></a> ·
    <a href="https://sidsoma.github.io/"><strong>Siddharth Somasundaram</strong></a> ·
    <a href="https://ychfan.github.io/"><strong>Yuchen Fan</strong></a><br> 
    <a href="https://richardt.name/"><strong>Christian Richardt</strong></a> · 
    <a href="https://www.media.mit.edu/people/raskar/overview/"><strong>Ramesh Raskar</strong></a> ·
    <a href=""><strong>Rakesh Ranjan</strong></a>
  </p>

  <p align="center" style="margin: 2em auto;">
    <a href='https://platonerf.github.io' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/PlatoNeRF-Project_page-orange?style=flat&logo=googlechrome&logoColor=orange' alt='Project Page'></a>
    <a href=''><img src='https://img.shields.io/badge/arXiv-Paper_PDF-red?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'></a>
  </p>

  <p align="center" style="font-size:16px">Official PyTorch implementation of PlatoNeRF, a method for recovering 3D geometry from single-view two-bounce lidar measurements. PlatoNeRF learns 3D geometry by reconstructing lidar measurements. PlatoNeRF is named after the <a href="https://en.wikipedia.org/wiki/Allegory_of_the_cave">allegory of Plato's Cave</a>, in which reality is discerned from shadows cast on a cave wall.</p>
  <p align="center">
    <img src="media/teaser.gif" />
  </p>
</div>

## Table of contents
-----
  * [Installation](#Installation)
  * [Downloading Datasets and Checkpoints](#downloading-datasets-and-checkpoints)
  * [Running Pretrained Models](#running-pretrained-models)
  * [Training](#Training)
  * [Rendering Lidar Data](#rendering-lidar-data-with-mitsuba)
  * [Baselines](#Baselines)
  * [Acknowledgements](#Acknowledgements)
  * [Citation](#Citation)
  * [License](#License)
------

## Installation

To install all dependencies for training and inference, please run:

```
pip install -r requirements.txt
```

## Downloading Datasets and Checkpoints

We provide datasets for each simulated scene reported in our paper and a real world scene. Each simulated scene's dataset includes measurements from 24 individual illumination spots. In addition, we include ground truth depth and RGB images for both the train view and the 120 test views used in the paper. The RGB images are not used in our work. For each simulated scene, we provide the checkpoint from our pretrained model.

To download the datasets and checkpoints:

```
bash download.sh
```

The data will be decompressed into './data' and the checkpoints will be decompressed into './pretrained'.

## Running Pretrained Models

Geometry can be extracted from the pretrained models either as a mesh or depth. Marching cubes can be used to extract a mesh. To render depth across all test views:

```
mkdir -p logs/chair # if this directory does not exist (you can specify any output directory)
python src/render_test_depth.py --config configs/chair.txt --ft_path pretrained/chair.tar --output_dir logs/chair
```

Output depth images will be saved in 'output\_dir/depth\_predictions'. Raw depth is stored as .npy and normalized depth is stored as .png. In the above example, inference is run on the chair scene, but any scene can be used by modifying the arguments.

## Training

### Train with Simulated Data

Download data and choose a scene to train with. The corresponding config can be found in './configs'. For example, if you choose the chair scene, then you would train with the following command:

```
python src/run_platonerf.py --config configs/chair.txt
```

Checkpoints will be saved in the './logs' directory.

In addition, we provide several other optional flags for visualization, ablation, and debugging:

1. `--vis_rays 1` enables creation of a video of the rays from the first training batch (0 by default)
2. `--downsample x` specifies the magnitude of spatial resolution downsampling, i.e. 2 reduces spatial resolution by 2x 
3. `--downsample_temp` specifies the magnitude of temporal resolution downsampling, i.e. 2 reduces temporal resolution by 2x
4. `--noise x` specifies the amount of gaussian noise to add to the measurement's time of flight, i.e. 50 adds a mean of 50 ps (default is no added noise)
4. `--debug 1` enables skipping shadow extraction from tof, which is very slow, so later code can quickly be reached for debugging (0 by default)

Once trained, you can render depth from test views with the following command (latest ckpt is automatically loaded):

```
python src/render_test_depth.py --config configs/chair.txt --output_dir logs/chair
```

### Train with Real-World Data

To train with real-world data, run the following command:

```
python src/run_platonerf_real.py --config configs/real.txt
```

Once trained, you can render depth with the following command (latest ckpt is automatically loaded):

```
python src/render_depth_real.py --config configs/real.txt --output_dir logs/real
```

## Rendering Lidar Data with Mitsuba

In addition to providing the rendered datasets, we also provide the code and assets for rendering these datasets (or new datasets) yourself. We render all ToF data using [MitsubaToF](https://github.com/cmu-ci-lab/MitsubaToFRenderer) and all depth and RGB data using [Mitsuba3](https://github.com/mitsuba-renderer/mitsuba3). All rendering is done within a MitsubaToF Docker container.

1. Follow [MitsubaToF instructions](https://github.com/cmu-ci-lab/MitsubaToFRenderer) for creating Docker container. On our systems, we run the following commands to create and enter the container:
```
docker run --gpus all -dit --shm-size 50G -p $(id -u):8888 -u $(id -u):2000 -e YOUR_HOST=$(hostname) -e YOUR_USERNAME=$(whoami) -e YOUR_UID=$(id -u) --name $(whoami)-mitsubatof adithyapediredla/mitsubatofrenderer
docker exec -it --user root $(whoami)-mitsubatof bash
```
2. Install Miniconda:
```
apt-get update
apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
``` 
3. Clone this repository and navigate to the rendering directory:
```
git clone https://github.com/facebookresearch/PlatoNeRF.git
cd PlatoNeRF/rendering
```
4. Install rendering dependencies and add MitsubaToF to PATH:
```
pip install -r requirements.txt
export PATH=/root/MitsubaToFRenderer/build/release/mitsuba:$PATH
```
5. Render ToF for scene of your choice (included are chair, dragon, bunny and occlusion scenes from the paper):
```
python render_tof.py scenes/chair 
```
6. Convert exr to npy and create metadata file:
```
python preprocess_tof.py scenes/chair
```
7. Rendering ground truth depth (both raw and normalized) and RGB images from test views (uses Mitsuba3):
```
mkdir scenes/chair/ground_truth # data will be saved here
python render_ground_truth.py scenes/chair/rgb.xml scenes/chair/ground_truth
```

## Baselines

We compare PlatoNeRF with two baseline methods: S3-NeRF and Bounce Flash Lidar. To run S3-NeRF, please refer to the [S3 NeRF repository](https://github.com/ywq/s3nerf). We rewrote the original [Bounce Flash Lidar codebase](https://github.com/co24401/BounceFlashLidar) in Python and provide the implementation in the './bf\_lidar' directory of this repo. To run the BF Lidar baseline on the simulated datasets, we run the following steps:

1. Estimate depth of the visible scene:
```
python bf_lidar/bf_lidar_depth.py # generates depth predictions for each scene
```
2. Run shadow carving algorithm:
```
python bf_lidar/bf_lidar_shadows.py chair 100 # runs shadow carving on the specified scene
```
3. Tune parameters for point cloud generation by visibly inspecting the point cloud. To do this, run `bf_lidar/save_point_cloud.ipynb`. We primarily tune the T parameter (which is a probability from 0 to 1).

## Acknowledgements

Our implementation is based on [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch). We thank the authors for their work.

## Citation

```
@article{PlatoNeRF,
	author    = {Klinghoffer, Tzofi and
		     Xiang, Xiaoyu and
		     Somasundaram, Siddharth and
		     Fan, Yuchen and 
		     Richardt, Christian and
		     Raskar, Ramesh and
		     Ranjan, Rakesh},
	title     = {{PlatoNeRF}: 3D Reconstruction in Plato's Cave via Single-View Two-Bounce Lidar},
	booktitle = {arXiv preprint},
	year      = {2023},
	url       = {https://platonerf.github.io},
}
```

## License

The PlatoNeRF code is available under the [MIT license](LICENSE).
