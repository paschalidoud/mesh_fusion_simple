# Watertight and Simplified Meshes through TSDF Fusion and ManifoldPlus

This repository contains a simple Python pipeline for obtaining watertight
meshes from arbitrary triangual meshes. We provide a simple to use python
wrapper for two methods that can be used for watertight conversion: (i)
[mesh-fusion](https://github.com/davidstutz/mesh-fusion) by David Stutz and
(ii) [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) by Jingwei Huang.
Our code is adapted by
[mesh-fusion](https://github.com/davidstutz/mesh-fusion), which in turn uses on
adapted versions of Gernot Riegler's
[pyrender](https://github.com/griegler/pyrender) and
[pyfusion](https://github.com/griegler/pyfusion); it also uses
[PyMCubes](https://github.com/pmneila/PyMCubes).

In case you use any of this code, please make sure to cite David's and Jingwei's work:

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

    @article{huang2020manifoldplus,
      title={ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups},
      author={Huang, Jingwei and Zhou, Yichao and Guibas, Leonidas},
      journal={arXiv preprint arXiv:2005.11621},
      year={2020}
    }

Please also check the individual GitHub repositories within this repo for
additional citations. Also check the corresponding project
pages of [TSDF Fusion](http://davidstutz.de/projects/shape-completion/)
amd [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus).

Note that if you want to perform watertight conversion using the ManifoldPlus
algorithm, you need to first compile the original code following the guidelines from
[here](https://github.com/hjwdzh/ManifoldPlus). As soon as you have compiled their code
you can directly pass the executable to any of the scripts in this repository.


## Installation & Dependencies

This codebase has the following dependencies:

- [numpy](https://numpy.org/doc/stable/user/install.html)
- [cython](https://cython.readthedocs.io/en/latest/src/quickstart/build.html)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [pycollada](https://pycollada.readthedocs.io/en/latest/install.html)
- [scipy](https://scipy.org/install/)
- [trimesh](https://github.com/mikedh/trimesh)
- [tqdm](https://github.com/tqdm/tqdm)
- [h5py](https://www.h5py.org/)
- [pymeshlab](https://pymeshlab.readthedocs.io/en/latest/)

For the visualizations, we use [simple-3dviz](http://simple-3dviz.com).
Note that
[simple-3dviz](http://simple-3dviz.com) provides a lightweight and easy-to-use
scene viewer using [wxpython](https://www.wxpython.org/). 

The simplest way to make sure that you have all dependencies in place is to use
[conda](https://docs.conda.io/projects/conda/en/4.6.1/index.html). You can
create a conda environment called ```mesh_fusion``` using
```
conda env create -f environment.yaml
conda activate mesh_fusion
```

Next compile the extenstion modules. You can do this via
```
python setup.py build_ext --inplace
pip install -e .
```

We also provide a Dockerfile that you can use to build a Docker image that contains all
dependencies for running inside a container. You can build the Docker image using the
following command:
```
docker build -f docker/Dockerfile --build-arg UBUNTU_VERSION=18.04 --build-arg PYTHON_VERSION=3.8 --tag mesh_fusion_simple:latest .
```

You can then run the Docker container in interactive mode, and use the scripts provided in the
`mesh_fusion_simple` directory of the Docker image. Make sure to mount the path with the 
location of the original meshes:
```
docker run -it --user $(id -u):$(id -g) --mount type=bind,source=[DATA_DIRECTORY_PATH],target=/data mesh_fusion_simple:latest
```
Keep in mind that the ManifoldPlus executable is located inside the folder `/mesh_fusion_simple/scripts` of the docker image,
named `manifoldplus`.

## Convert to watertight meshes

To run our code, we provide the `convert_to_watertight.py` script. In order to
run this script you only need to provide a path to the dataset directory, as
well as the dataset type, namely ShapeNet, Dynamic FAUST, 3D-FRONT etc. For now,
our code supports the 3D-FRONT, the ShapeNet, the Dynamic FAUST and the FreiHAND
dataset. If you want to use another dataset, you simply need to implement a
Dataset class that extends the `ModelCollection` class. For more details please
refer to the `watertight_transformer/datasets/model_collections.py` file. To run the
conversion script simply run
```
python convert_to_watertight.py path_to_dataset_directory --dataset_type dataset_type
```
Currently, our code only runs on CPU. However, you can lauch this script from
multiple CPU nodes in order to speed up the computation time. This script
automatically checks whether a model has already been converted before initiating
the transformation process. In case you want to run this script at an Ubuntu
machine, with no monitor, simply run
```
for i in {1..10}; do xvfb-run -a python convert_to_watertight.py /orion/u/paschald/Datasets/ShapeNetCore.v1/ --watertight_method tsdf_fusion --category_tags 02691156 --dataset_type shapenet_v1 & done
wait
```
This script launches 10 CPU jobs. However, you can launch more or less
depending on the availability of your resources.

You can also use the `make_mesh_watertight.py` script to convert a single mesh
to a watertight mesh by specifying its path as follows
```
python make_mesh_watertight.py path_to_mesh path_to_output_directory --watertight_method tsdf_fusion
```

Note that for both scripts you can set `--simplify` in order to simplify the
final watertight mesh using
[pymeshlab](https://pymeshlab.readthedocs.io/en/latest/). You can also rescale
the watertight mesh, either using bounding box bounds (`--bbox`), or to make it
fit inside a unit cube (`--unit_cube`).
