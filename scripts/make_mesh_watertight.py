#!/usr/bin/env python
import argparse
import logging
import os
import subprocess
import sys

import numpy as np
from tqdm import tqdm
import trimesh

import pymeshlab

from simple_3dviz import Mesh

from mesh_fusion.fusion import TSDFFusion


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Convert non-watertight meshes to watertight"
    )
    parser.add_argument(
        "path_to_meshes",
        help="Path to folder containing the mesh/meshes to be converted"
    )
    parser.add_argument(
        "path_to_output_directory",
        help="Path to save the watertight mesh"
    )
    parser.add_argument(
        "--simplify",
        action="store_true"
        help="Simplify the watertight mesh"
    )
    parser.add_argument(
        "--num_target_faces",
        type=float,
        default=7000,
        help="Max number of faces in the simplified mesh"
    )
    args = parser.parse_args(argv)
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if os.path.isdir(args.path_to_meshes):
        path_to_meshes = [
            os.path.join(args.path_to_meshes, mi)
            for mi in os.listdir(args.path_to_meshes)
            if mi.endswith(".obj") or mi.endswith(".off")
        ]
    else:
        path_to_meshes = [args.path_to_meshes]

    # Check optimistically if the file already exists
    ensure_parent_directory_exists(args.path_to_output_directory)

    for pi in tqdm(path_to_meshes):
        file_name = pi.split("/")[-1].split(".")[0]
        path_to_output_file = os.path.join(
            args.path_to_output_directory, f"{file_name}.off"
        )
        # Load and scale the mesh to range [-0.5,0.5]^3
        raw_mesh = Mesh.from_file(pi)
        raw_mesh.to_unit_cube()
        # Extract the points and the faces from the raw_mesh
        points, faces = raw_mesh.to_points_and_faces()

        tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        # Check if the mesh is indeed non-watertight before making the
        # conversion
        if tr_mesh.is_watertight:
            print(f"Mesh file: {pi} is watertight...")
            tr_mesh.export(path_to_output_file, file_type="obj")
            continue

        # Make the mesh watertight with TSDF Fusion
        tsdf_fuser = TSDFFusion()
        tr_mesh_watertight = tsdf_fuser.to_watertight(
            tr_mesh, path_to_output_file
        )
        if args.simplify:
            # Call the meshlabserver to simplify the mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path_to_output_file)
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=args.num_target_faces,
                qualitythr=0.5,
                preservenormal=True
            )
            ms.save_current_mesh(path_to_output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
