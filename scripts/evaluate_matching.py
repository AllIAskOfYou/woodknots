from pathlib import Path
import cv2
import numpy as np

from mltools.matching.utils import parse_meshroom_cameras
from mltools.matching.render import MeshPositionRasterizer


if __name__ == "__main__":
    project_dir = Path("data/matching/meshroom")
    cameras_file = project_dir / "cameras.sfm"
    mesh_file = project_dir / "downsampled.obj" # mesh.obj"

    cameras = parse_meshroom_cameras(cameras_file)
    rasterizer = MeshPositionRasterizer(mesh_file, near=0.01, far=1000.0)

    # Example: load image and cast every pixel. Using the casted rays create a depth map of the mesh.
    image_path = Path("data/matching/resized_1024") / "2041.JPG"

    image = cv2.imread(str(image_path))
    image_name = image_path.name
    height, width, _ = image.shape

    camera = cameras[image_name]
    points, mask = rasterizer.render(camera, width=width, height=height)

    # generate depth map from points and mask
    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_map[mask] = points[mask][:, 2]

    # store raw depth map as a .npy file
    depth_map_npy_path = Path("data/matching/depth_maps") / f"{image_name}.npy"
    depth_map_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_map_npy_path, depth_map)

    # save depth map as 16-bit PNG
    depth_map_path = Path("data/matching/depth_maps") / f"{image_name}_vis.png"
    depth_map_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(depth_map_path), (depth_map * 1000).astype(np.uint16))  # scale to millimeters and save as uint16
