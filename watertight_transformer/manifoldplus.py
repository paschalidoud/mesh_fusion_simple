import subprocess

import trimesh


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

    def to_watertight(self, path_to_mesh, path_to_watertight):
        subprocess.call([
            self.manifoldplus_script,
            "--input", path_to_mesh,
            "--output", path_to_watertight,
            "--depth", str(self.depth),
        ], stdout=subprocess.DEVNULL)

        return trimesh.load(path_to_watertight)
