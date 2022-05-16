import os

import numpy as np
import pymeshlab
import trimesh
from watertight_transformer.base import WatertightTransformerFactory
from watertight_transformer.datasets.model_collections import Mesh


def ensure_parent_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def mesh_to_watertight(
    mesh: Mesh,
    wat_transformer: WatertightTransformerFactory,
    path_to_file: str,
    bbox: list = None,
    unit_cube: bool = False,
    simplify: bool = None,
    num_target_faces: int = None,
    ratio_target_faces: float = None,
):
    # Check optimistically if the file already exists
    if os.path.exists(path_to_file):
        return
    ensure_parent_directory_exists(path_to_file)

    if bbox is not None:
        # Scale the mesh to range specified from the input bounding box
        bbox_min = np.array(bbox[:3])
        bbox_max = np.array(bbox[3:])
        dims = bbox_max - bbox_min
        mesh._vertices -= dims / 2 + bbox_min
        mesh._vertices /= dims.max()
    else:
        if unit_cube:
            # Scale the mesh to range [-0.5,0.5]^3
            # This is needed for TSDF Fusion!
            mesh.to_unit_cube()
    # Extract the points and the faces from the mesh
    points, faces = mesh.to_points_and_faces()

    tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    # Check if the mesh is indeed non-watertight before making the
    # conversion
    if tr_mesh.is_watertight:
        print(f"Mesh file: {path_to_file} is watertight...")
        tr_mesh.export(path_to_file, file_type="obj")
        return
    # Make the mesh watertight with TSDF Fusion
    tr_mesh_watertight = wat_transformer.to_watertight(tr_mesh, path_to_file, file_type="obj")

    if simplify:
        if num_target_faces:
            num_faces = num_target_faces
        else:
            num_faces = int(ratio_target_faces * len(faces))
        # Call the meshlabserver to simplify the mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path_to_file)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=num_faces,
            qualitythr=0.5,
            preservenormal=True,
            planarquadric=True,
            preservetopology=True,
            autoclean=False,  # very important for watertightness preservation
        )
        ms.save_current_mesh(path_to_file)
