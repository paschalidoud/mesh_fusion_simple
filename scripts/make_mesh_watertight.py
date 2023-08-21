#!/usr/bin/env python
import argparse
import logging
import os
import sys
from functools import partial

import trimesh
from simple_3dviz import Mesh
from tqdm.contrib.concurrent import process_map
from watertight_transformer import WatertightTransformerFactory

from arguments import add_manifoldplus_parameters, add_tsdf_fusion_parameters
from utils import ensure_parent_directory_exists, mesh_to_watertight


def distribute_files(
    mesh_paths: list,
    output_folder_path: str,
    wat_transformer,
    bbox: list = None,
    unit_cube: bool = False,
    simplify: bool = None,
    num_target_faces: int = None,
    ratio_target_faces: float = None,
    num_cpus: int = 1,
):
    # Assuming that dataset iterator contains only one instance of each path
    process_map(
        partial(
            mesh_path_to_watertight,
            output_folder_path=output_folder_path,
            wat_transformer=wat_transformer,
            bbox=bbox,
            unit_cube=unit_cube,
            simplify=simplify,
            num_target_faces=num_target_faces,
            ratio_target_faces=ratio_target_faces,
        ),
        mesh_paths,
        max_workers=num_cpus,
    )


def mesh_path_to_watertight(
    mesh_path: str,
    output_folder_path: str,
    wat_transformer,
    bbox: list = None,
    unit_cube: bool = False,
    simplify: bool = None,
    num_target_faces: int = None,
    ratio_target_faces: float = None,
):
    file_name = mesh_path.split("/")[-1].split(".")[0]
    # path_to_file = os.path.join(output_folder_path, f"{file_name}.obj")
    path_to_file = os.path.join(output_folder_path, "model_watertight.obj")
    raw_mesh = Mesh.from_file(mesh_path)
    mesh_to_watertight(
        mesh=raw_mesh,
        wat_transformer=wat_transformer,
        path_to_file=path_to_file,
        bbox=bbox,
        unit_cube=unit_cube,
        simplify=simplify,
        num_target_faces=num_target_faces,
        ratio_target_faces=ratio_target_faces,
    )


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
        help="Path to save the watertight mesh")
    parser.add_argument(
        "--watertight_method",
        default="tsdf_fusion",
        choices=["tsdf_fusion", "manifoldplus"]
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
              "By default we use the unit cube")
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the watertight mesh"
    )
    parser.add_argument(
        "--num_target_faces",
        type=int,
        default=None,
        help="Max number of faces in the simplified mesh",
    )
    parser.add_argument(
        "--ratio_target_faces",
        type=float,
        default=None,
        help="Ratio of target faces with regards to input mesh faces",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of processes to be used for the multiprocessing setup",
    )

    add_tsdf_fusion_parameters(parser)
    add_manifoldplus_parameters(parser)
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
        depth=args.depth,
    )

    distribute_files(
        mesh_paths=path_to_meshes,
        output_folder_path=args.path_to_output_directory,
        wat_transformer=wat_transformer,
        bbox=args.bbox,
        unit_cube=args.unit_cube,
        simplify=args.simplify,
        num_target_faces=args.num_target_faces,
        ratio_target_faces=args.ratio_target_faces,
        num_cpus=args.num_cpus,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
