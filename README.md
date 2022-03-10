# Watertight and Simplified Meshes through TSDF Fusion

This repository contains a simple Python pipeline for obtaining watertight
meshes from arbitrary triangual meshes. The code is adapted from
[mesh-fusion](https://github.com/davidstutz/mesh-fusion) by David Stutz. Note
that [mesh-fusion](https://github.com/davidstutz/mesh-fusion) is largely based
on adapted versions of Gernot Riegler's
[pyrender](https://github.com/griegler/pyrender) and
[pyfusion](https://github.com/griegler/pyfusion); it also uses
[PyMCubes](https://github.com/pmneila/PyMCubes).

In case you use any of this code, please make sure to cite David's work:

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

Please also check the individual GitHub repositories within this repo for
additional citations.  Also check the corresponding [project
page](http://davidstutz.de/projects/shape-completion/).


## Installation & Dependencies

This codebase has the following dependencies:

- [numpy](https://numpy.org/doc/stable/user/install.html)
- [cython](https://cython.readthedocs.io/en/latest/src/quickstart/build.html)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [trimesh](https://github.com/mikedh/trimesh)
- [tqdm](https://github.com/tqdm/tqdm)

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
 
