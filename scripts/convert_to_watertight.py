#!/usr/bin/env python
"""Script for converting non-watertight meshes to watertight meshes.
"""
import argparse
import logging
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from tqdm import tqdm
import trimesh

from mesh_fusion.datasets import ModelCollectionBuilder
from mesh_fusion.fusion import TSDFFusion


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Convert non-watertight meshes to watertight"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "--dataset_type",
        default="shapenet_v1",
        choices=[
            "shapenet_v1",
            "dynamic_faust",
            "freihand",
            "3d_future",
            "deforming_things_4d"
        ],
        help="The type of the dataset type to be used"
    )
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Tags to the models to be used"
    )
    parser.add_argument(
        "--category_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Category tags to the models to be used"
    )
    parser.add_argument(
        "--path_to_simplification_script",
        default="simplification.mlx",
        help="Script to be used for mesh simplification"
    )

    args = parser.parse_args(argv)
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    dataset = (
        ModelCollectionBuilder()
        .with_dataset(args.dataset_type)
        .filter_category_tags(args.category_tags)
        .filter_tags(args.model_tags)
        .build(args.dataset_directory)
    )

    tsdf_fuser = TSDFFusion()

    for sample in tqdm(dataset):
        # Assemble the target path and ensure the parent dir exists
        path_to_file = sample.path_to_watertight_mesh_file

        # Check optimistically if the file already exists
        if os.path.exists(path_to_file):
            continue
        ensure_parent_directory_exists(path_to_file)

        # Make sure we are the only ones creating this file
        with DirLock(path_to_file + ".lock") as lock:
            if not lock.is_acquired:
                continue
            if os.path.exists(path_to_file):
                continue
            # Scale the mesh to range [-0.5,0.5]^3
            raw_mesh = sample.groundtruth_mesh
            raw_mesh.to_unit_cube()
            # Extract the points and the faces from the raw_mesh
            points, faces = raw_mesh.to_points_and_faces()

            tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)
            # Make the mesh watertight with TSDF Fusion
            tr_mesh_watertight = tsdf_fuser.to_watertight(
                tr_mesh, path_to_file
            )
            # Call the meshlabserver to simplify the mesh
            simplification_script = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                args.path_to_simplification_script
            )
            subprocess.call([
                "meshlabserver"
                f" -i {path_to_file} -o {path_to_file} -s {simplification_script}",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
