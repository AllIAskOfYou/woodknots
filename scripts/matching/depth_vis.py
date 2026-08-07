from mltools.matching.render import MeshRenderer
from mltools.matching.utils import MeshroomCamerasParser
import numpy as np


if __name__ == "__main__":
    cameras_path = "data/matching/meshroom/cameras.sfm"
    obj_path = "data/matching/meshroom/mesh.obj"

    cameras = MeshroomCamerasParser(cameras_path).parse()
    print(list(cameras.values())[0])
    
    # Initialize the renderer once
    renderer = MeshRenderer(obj_path, default_width=1024, default_height=682)
    output_dir = "data/matching/vis"
    
    for image_id, camera in list(cameras.items())[:5]:
        output_filename = f"{output_dir}/depth_{image_id}.png"
        renderer.visualize_depth(output_filename, camera)
    
    rotation_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ]

    # Define multiple camera perspectives
    camera_parameters = {
        'position': [0.0, 1.5, 0.0],
        'rotation_matrix': rotation_matrix,
        'focal_length': 35.0,
        'sensor_width': 36.0,
        'sensor_height': 24.0,
        'width': 300,
        'height': 300,
        'cx': 0.0,
        'cy': 0.0,
        'distortion_params': np.array([0.0, 0.0, 0.0], dtype=np.float32)  # No distortion
    }

    try:
        output_filename = f"data/matching/depth_visualization.png"
        renderer.render(output_filename, camera_parameters)
    finally:
        # Always clean up resources safely when done
        renderer.close()