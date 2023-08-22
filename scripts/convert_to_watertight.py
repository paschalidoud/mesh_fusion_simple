#!/usr/bin/env python
"""Script for converting non-watertight meshes to watertight meshes. This
script is intended for meshes organized in datasets."""
import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm
import trimesh

import pymeshlab

from watertight_transformer.datasets import ModelCollectionBuilder
from watertight_transformer import WatertightTransformerFactory

from arguments import add_tsdf_fusion_parameters, \
    add_manifoldplus_parameters
from .utils import DirLock, ensure_parent_directory_exists


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
        "--watertight_method",
        default="tsdf_fusion",
        choices=[
            "tsdf_fusion",
            "manifoldplus"
        ]
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the watertight mesh"
    )
    parser.add_argument(
        "--num_target_faces",
        type=float,
        default=7000,
        help="Max number of faces in the simplified mesh"
    )
    parser.add_argument(
        "--unit_cube",
        action="store_true",
        help="Normalize mesh to fit a unit cube"
    )
    parser.add_argument(
        "--bbox",
        type=lambda x: list(map(float, x.split(","))),
        default=None,
        help=("Bounding box to be used for scaling. "
              "By default we use the unit cube"),
    )

    add_tsdf_fusion_parameters(parser)
    add_manifoldplus_parameters(parser)
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

    wat_transformer = WatertightTransformerFactory(
        args.watertight_method,
        image_height=args.image_size[0],
        image_width=args.image_size[1],
        focal_length_x=args.focal_point[0],
        focal_length_y=args.focal_point[1],
        principal_point_x=args.principal_point[0],
        principal_point_y=args.principal_point[1],
        resolution=args.resolution,
        truncation_factor=args.truncation_factor,
        n_views=args.n_views,
        depth_offset_factor=args.depth_offset_factor,
        manifoldplus_script=args.manifoldplus_script,
        depth=args.depth
    )

    for sample in tqdm(dataset):
        # Assemble the target path and ensure the parent dir exists
        path_to_file = sample.path_to_watertight_mesh_file

        # Check optimistically if the file already exists
        if os.path.exists(path_to_file):
            continue
        ensure_parent_directory_exists(path_to_file)
        # Extract the file type from the output file
        file_type = path_to_file.split(".")[-1]
        if file_type not in ["off", "obj"]:
            raise Exception(f"The {file_type} is not a valid mesh extension")

        # Make sure we are the only ones creating this file
        with DirLock(path_to_file + ".lock") as lock:
            if not lock.is_acquired:
                continue
            if os.path.exists(path_to_file):
                continue
            # Scale the mesh to range [-0.5,0.5]^3
            raw_mesh = sample.groundtruth_mesh

            if args.bbox is not None:
                # Scale the mesh to range specified from the input bounding box
                bbox_min = np.array(bbox[:3])
                bbox_max = np.array(bbox[3:])
                dims = bbox_max - bbox_min
                raw_mesh._vertices -= dims / 2 + bbox_min
                raw_mesh._vertices /= dims.max()
            else:
                if unit_cube:
                    # Scale the mesh to range [-0.5,0.5]^3
                    # This is needed for TSDF Fusion!
                    raw_mesh.to_unit_cube()

            # Extract the points and the faces from the raw_mesh
            points, faces = raw_mesh.to_points_and_faces()

            tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)
            # Check if the mesh is indeed non-watertight before making the
            # conversion
            if tr_mesh.is_watertight:
                print(f"Mesh file: {path_to_file} is watertight...")
                tr_mesh.export(path_to_file, file_type=file_type)
                continue

            # Make the mesh watertight with TSDF Fusion or Manifold Plus
            tr_mesh_watertight = wat_transformer.to_watertight(
                tr_mesh, path_to_file, file_type=file_type
            )

            if args.simplify:
                # Call the meshlabserver to simplify the mesh
                print("Performing mesh simplification...")
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(path_to_file)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=args.num_target_faces,
                    qualitythr=0.5,
                    preservenormal=True
                    planarquadric=True,
                    preservetopology=True,
                    autoclean=False,  # very important for the
                                      # watertightness preservation
                )
                ms.save_current_mesh(path_to_file)


if __name__ == "__main__":
    main(sys.argv[1:])
