import subprocess

import trimesh

from simple_3dviz import Mesh


class ManifoldPlus:
    """Performs the watertight conversion using the Manifold algorithm from [1]

    [1] ManifoldPlus: A Robust and Scalable Watertight Manifold Surface
    Generation Method for Triangle Soups, by Huang, Jingwei and Zhou, Yichao
    and Guibas, Leonidas
    """
    def __init__(
        self,
        manifoldplus_script,
        depth=10,
    ):
        self.manifoldplus_script = manifoldplus_script
        self.depth = depth

    def to_watertight(self, path_to_mesh, path_to_watertight, file_type="off"):
        subprocess.call([
            self.manifoldplus_script,
            "--input", path_to_mesh,
            "--output", path_to_watertight,
            "--depth", str(self.depth),
        ], stdout=subprocess.DEVNULL)

        tr_mesh = Mesh.from_file(path_to_watertight)
        # Extract the points and the faces from the raw_mesh
        points, faces = tr_mesh.to_points_and_faces()
        tr_mesh = trimesh.Trimesh(vertices=points, faces=faces)

        #trimesh.repair.fix_winding(tr_mesh)
        tr_mesh.export(path_to_watertight, file_type)
        return tr_mesh

