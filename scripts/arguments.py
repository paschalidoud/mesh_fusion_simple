def add_tsdf_fusion_parameters(parser):
    parser.add_argument(
        "--n_views",
        type=int,
        default=100,
        help="Number of views per model"
    )
    parser.add_argument(
        "--image_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="640,640",
        help="The size of the depth images"
    )
    parser.add_argument(
        "--focal_point",
        type=lambda x: tuple(map(int, x.split(","))),
        default="640,640",
        help="The focal length in x and y direction"
    )
    parser.add_argument(
        "--principal_point",
        type=lambda x: tuple(map(int, x.split(","))),
        default="320,320",
        help="The principal point location in x and y direction"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="The resolution for the fusion"
    )
    parser.add_argument(
        "--truncation_factor",
        type=int,
        default=15,
        help=("The truncation for fusion is derived as "
              "truncation_factor*voxel_size.")
    )
    parser.add_argument(
        "--depth_offset_factor",
        type=float,
        default=1.5,
        help=("The depth maps are offsetted using "
              "depth_offset_factor*voxel_size.")
    )


def add_manifoldplus_parameters(parser):
    parser.add_argument(
        "--manifoldplus_script",
        default=None,
        help="Path to the script used for implemented the Manifold algorithm"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Number of depth values used in the Manifold algorithm"
    )
