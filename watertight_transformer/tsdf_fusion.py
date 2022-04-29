import math
import numpy as np
from scipy import ndimage

import trimesh

from .external.librender import pyrender
from .external.libmcubes import mcubes
from .external.libfusioncpu import cyfusion as libfusion
from .external.libfusioncpu.cyfusion import tsdf_cpu as compute_tsdf

from .utils import write_hdf5, read_hdf5


class TSDFFusion:
    """Perform the TSDF fusion.
    Code adapted from
    https://github.com/davidstutz/mesh-fusion/blob/master/2_fusion.py
    """
    def __init__(
        self,
        image_height=640,
        image_width=640,
        focal_length_x=640,
        focal_length_y=640,
        principal_point_x=320,
        principal_point_y=320,
        resolution=256,
        truncation_factor=15,
        n_views=100,
        depth_offset_factor=1.5
    ):
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.ppx = principal_point_x
        self.ppy = principal_point_y
        self.image_height = image_height
        self.image_width = image_width
        self.resolution = resolution
        self.truncation_factor = truncation_factor
        self.n_views = n_views
        self.depth_offset_factor = depth_offset_factor

        self.render_intrinsics = np.array([
            self.fx, self.fy, self.ppx, self.ppy
        ], dtype=float)
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrisics = np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ])
        self.image_size = np.array([
            self.image_height, self.image_width,
        ], dtype=np.int32)
        # Mesh will be centered at (0, 0, 1)!
        self.znf = np.array([
            1 - 0.75,
            1 + 0.75
        ], dtype=float)
        # Derive voxel size from resolution.
        self.voxel_size = 1.0 / self.resolution
        self.truncation = self.truncation_factor * self.voxel_size

    def get_points_on_sphere(self):
        """Code adapted from

        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.
        """
        rnd = 1.
        points = []
        offset = 2. / self.n_views
        increment = math.pi * (3. - math.sqrt(5.));

        for i in range(self.n_views):
            y = ((i * offset) - 1) + (offset / 2);
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % self.n_views) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        return np.array(points)

    def get_views(self):
        """Generate a set of views to generate depth maps from.
        """
        Rs = []
        points = self.get_points_on_sphere()

        for i in range(points.shape[0]):
            # Code adapted from
            # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
            longitude = - math.atan2(points[i, 0], points[i, 1])
            latitude = math.atan2(
                points[i, 2],
                math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2)
            )

            R_x = np.array([
                [1, 0, 0],
                [0, math.cos(latitude), -math.sin(latitude)],
                [0, math.sin(latitude), math.cos(latitude)]
            ])
            R_y = np.array([
                [math.cos(longitude), 0, math.sin(longitude)],
                [0, 1, 0],
                [-math.sin(longitude), 0, math.cos(longitude)]
            ])
            R = R_y.dot(R_x)
            Rs.append(R)

        return Rs

    def render(self, mesh, Rs, output_path=None):
        """Render the given mesh using the generated views.

        Arguments:
        -----------
            mesh: trimesh.Mesh object
            Rs: rotation matrices
            output_path: path to store the computed depth maps
        """
        depthmaps = []
        for i in range(len(Rs)):
            np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
            np_vertices[2, :] += 1

            np_faces = mesh.faces.astype(np.float64)
            np_faces += 1

            depthmap, mask, img = pyrender.render(
                np_vertices.copy(),
                np_faces.T.copy(),
                self.render_intrinsics,
                self.znf,
                self.image_size
            )

            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs).
            depthmap -= self.depth_offset_factor * self.voxel_size
            depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))
            depthmaps.append(depthmap)

        if output_path is not None:
            write_hdf5(output_path, np.array(depthmaps))
        return depthmaps

    def fusion(self, depthmaps, Rs):
        """Fuse the rendered depth maps.

        Arguments:
        -----------
            depthmaps: np.array of depth maps
            Rs: rotation matrices
        """

        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        Ts = []
        for i in range(len(Rs)):
            Rs[i] = Rs[i]
            Ts.append(np.array([0, 0, 1]))

        Ts = np.array(Ts).astype(np.float32)
        Rs = np.array(Rs).astype(np.float32)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        # Note that this is an alias defined as libfusiongpu.tsdf_gpu or
        # libfusioncpu.tsdf_cpu!
        return compute_tsdf(
            views,
            self.resolution,
            self.resolution,
            self.resolution,
            self.voxel_size,
            self.truncation,
            False
        )
    
    def to_watertight(self, mesh, output_path=None, file_type="off"):
        # Get the views that we will use for the rendering
        Rs = self.get_views()
        # Render the depth maps
        depths = self.render(mesh, Rs)
        tsdf = self.fusion(depths, Rs)[0]
        # To ensure that the final mesh is indeed watertight
        tsdf = np.pad(tsdf, 1, "constant", constant_values=1e6)
        vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
        # Remove padding offset
        vertices -= 1
        # Normalize to [-0.5, 0.5]^3 cube
        vertices /= self.resolution
        vertices -= 0.5

        tr_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        if output_path is not None:
            tr_mesh.export(output_path, file_type)
        return tr_mesh
