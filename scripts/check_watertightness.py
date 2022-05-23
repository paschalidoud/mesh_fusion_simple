#!/usr/bin/env python
"""Script for checking watertightness property of meshes.
"""
import argparse
import logging
import os
import sys

import numpy as np
import trimesh
from tqdm import tqdm
from watertight_transformer.datasets import ModelCollectionBuilder


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Check if generated meshes are watertight and save non-watertight model paths to a file"
    )
    parser.add_argument("dataset_directory", help="Path to the directory containing the dataset")
    parser.add_argument(
        "text_directory",
        help="Path to the directory that will have the list of non watertight meshes file",
    )
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

    count = 0
    with open(f"{args.text_directory}/non_watertight_list.txt", "w") as f:
        for sample in tqdm(dataset):
            # Assemble the target path and ensure the parent dir exists
            path_to_file = sample.path_to_watertight_mesh_file

            # If path does not exist we need to log that
            if not os.path.exists(path_to_file):
                print(f"File does not exist in location: {path_to_file}")

            try:
                # Load mesh using Trimesh to check watertightness
                tr_mesh = trimesh.load(path_to_file, process=False, force="mesh")
            except Exception as e:
                print(f"Error raised while loading file {path_to_file}")
                raise
            if not tr_mesh.is_watertight:
                count += 1
                f.write(path_to_file + "\n")
    if not count:
        print("All meshes in the relevant directory are watertight!")


if __name__ == "__main__":
    main(sys.argv[1:])
