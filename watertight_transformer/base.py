from tempfile import NamedTemporaryFile

from .manifoldplus import ManifoldPlus
from .tsdf_fusion import TSDFFusion


class WatertightTransformerFactory:
    """
    Arguments:
    ----------
        image_height: Image height of depth map generated during TSDFFusion
        image_width: Image width of depth map generated during TSDFFusion
        focal_length_x: The focal length along the x-axis for TSDFFusion
        focal_length_y: The focal length along the y-axis for TSDFFusion
        principal_point_x: The principal point along the x-axis for TSDFFusion
        principal_point_y: The principal point along the y-axis for TSDFFusion
        resolution: The resoluion for the TSDFFusion
        truncation_factor: The truncation_factor for the TSDFFusion is derived as
                            truncation_factor*voxel_size
        depth_offset_factor: The depth maps are offsetted using 
                             depth_offset_factor*voxel_size in TSDFFusion
        n_views: The number of views used in TSDFFusion
        manifold_plus_script: Path to the binary file to be used to perform the
                              Manifold algorithm
        depth: Number of depth values used in the Manifold algorithm 
    """
    def __init__(
        self,
        name,
        image_height=640,
        image_width=640,
        focal_length_x=640,
        focal_length_y=640,
        principal_point_x=320,
        principal_point_y=320,
        resolution=256,
        truncation_factor=15,
        n_views=100,
        depth_offset_factor=1.5,
        manifoldplus_script=None,
        depth=10,
    ):
        self.name = name
        if self.name == "manifoldplus":
            # Make sure that the correct arguments are provided
            if manifoldplus_script is None:
                raise Exception(
                    "Cannot run ManifoldPlus without specifying a script"
                )
            self.wat_transformer = ManifoldPlus(
                manifoldplus_script=manifoldplus_script,
                depth=depth
            )
        elif self.name == "tsdf_fusion":
            self.wat_transformer = TSDFFusion(
                image_height=image_height,
                image_width=image_width,
                focal_length_x=focal_length_y,
                focal_length_y=focal_length_y,
                principal_point_x=principal_point_x,
                principal_point_y=principal_point_y,
                resolution=resolution,
                truncation_factor=truncation_factor,
                n_views=n_views,
                depth_offset_factor=depth_offset_factor
            )
        else:
            raise NotImplementedError()

    
    def to_watertight(self, mesh, path_to_watertight, file_type="off"):
        if self.name == "manifoldplus":
            # Create a temporary file and store the mesh
            path_to_mesh = NamedTemporaryFile().name + file_type
            mesh.export(path_to_mesh, file_type=file_type)
            self.wat_transformer.to_watertight(
                path_to_mesh, path_to_watertight, file_type
            )
        elif self.name == "tsdf_fusion":
            self.wat_transformer.to_watertight(
                mesh, path_to_watertight, file_type
            )
        else:
            raise NotImplementedError()
