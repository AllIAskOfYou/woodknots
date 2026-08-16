import numpy as np
import os
import tqdm

from mltools.matching.render import MeshRenderer
from mltools.matching.utils import MeshroomCamerasParser

if __name__ == "__main__":
    cameras_path = "data/matching/cameras/cameras_v2.sfm"
    obj_path = "data/matching/textured_mesh_v2/texturedMesh.obj"
    output_dir = "data/matching/resized_1024_3d_v3_distortion"
    os.makedirs(output_dir, exist_ok=True)

    cameras = MeshroomCamerasParser(cameras_path).parse()
    tmp_camera = list(cameras.values())[0]
    w, h = tmp_camera["width"], tmp_camera["height"]
    
    # Initialize the renderer once
    renderer = MeshRenderer(obj_path, default_width=w, default_height=h)
    
    for image_id, camera in tqdm.tqdm(list(cameras.items())):
        pos = renderer.render(camera)
        pixel_data_file = f"{output_dir}/{image_id}.npy"
        np.save(pixel_data_file, pos)