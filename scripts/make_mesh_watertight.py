#!/usr/bin/env python
import argparse
import logging
import os
import subprocess
import sys

import numpy as np
from tqdm import tqdm
import trimesh

from simple_3dviz import Mesh

from mesh_fusion.fusion import TSDFFusion


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Convert non-watertight meshes to watertight"
    )
    parser.add_argument(
        "path_to_mesh",
        help="Path to the mesh to be converted"
    )
    parser.add_argument(
        "path_to_output_directory",
        help="Path to save the watertight mesh"
    )
    parser.add_argument(
        "--path_to_simplification_script",
        default="simplification.mlx",
        help="Script to be used for mesh simplification"
    )
    args = parser.parse_args(argv)
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    path_to_file = os.path.join(
        args.path_to_output_directory, "model_watertight.off"
    )
    # Check optimistically if the file already exists
    ensure_parent_directory_exists(path_to_file)

    # Load and scale the mesh to range [-0.5,0.5]^3
    raw_mesh = Mesh.from_file(args.path_to_mesh)
    raw_mesh.to_unit_cube()
    # Extract the points and the faces from the raw_mesh
    points, faces = raw_mesh.to_points_and_faces()

    tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    # Make the mesh watertight with TSDF Fusion
    tsdf_fuser = TSDFFusion()
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
