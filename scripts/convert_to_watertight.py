#!/usr/bin/env python
"""Script for converting non-watertight meshes to watertight meshes.
"""
import argparse
import logging
import sys
from functools import partial

import trimesh
from tqdm.contrib.concurrent import process_map
from watertight_transformer import WatertightTransformerFactory
from watertight_transformer.datasets import ModelCollectionBuilder
from watertight_transformer.datasets.model_collections import BaseModel, ModelCollection

from arguments import add_manifoldplus_parameters, add_tsdf_fusion_parameters
from utils import mesh_to_watertight


def distribute_files(
    dataset: ModelCollection,
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
            ds_sample_to_watertight,
            wat_transformer=wat_transformer,
            bbox=bbox,
            unit_cube=unit_cube,
            simplify=simplify,
            num_target_faces=num_target_faces,
            ratio_target_faces=ratio_target_faces,
        ),
        dataset,
        max_workers=num_cpus,
    )


def ds_sample_to_watertight(
    sample: BaseModel,
    wat_transformer,
    bbox: list = None,
    unit_cube: bool = False,
    simplify: bool = None,
    num_target_faces: int = None,
    ratio_target_faces: float = None,
):
    mesh = sample.groundtruth_mesh
    path_to_file = sample.path_to_watertight_mesh_file
    mesh_to_watertight(
        mesh=mesh,
        wat_transformer=wat_transformer,
        path_to_file=path_to_file,
        bbox=bbox,
        unit_cube=unit_cube,
        simplify=simplify,
        num_target_faces=num_target_faces,
        ratio_target_faces=ratio_target_faces,
    )


def main(argv):
    parser = argparse.ArgumentParser(description="Convert non-watertight meshes to watertight")
    parser.add_argument("dataset_directory", help="Path to the directory containing the dataset")
    parser.add_argument(
        "--dataset_type",
        default="shapenet_v1",
        choices=["shapenet_v1", "dynamic_faust", "freihand", "3d_future", "deforming_things_4d"],
        help="The type of the dataset type to be used",
    )
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Tags to the models to be used",
    )
    parser.add_argument(
        "--category_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Category tags to the models to be used",
    )
    parser.add_argument(
        "--watertight_method", default="tsdf_fusion", choices=["tsdf_fusion", "manifoldplus"]
    )
    parser.add_argument(
        "--unit_cube", action="store_true", help="Normalize mesh to fit a unit cube"
    )
    parser.add_argument(
        "--bbox",
        type=lambda x: list(map(float, x.split(","))),
        default=None,
        help=("Bounding box to be used for scaling. " "By default we use the unit cube"),
    )
    parser.add_argument("--simplify", action="store_true", help="Simplify the watertight mesh")
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
        depth=args.depth,
    )

    distribute_files(
        dataset=dataset,
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
