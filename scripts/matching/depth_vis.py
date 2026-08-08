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

    print(f"Rendering depth maps for {len(cameras)} cameras...")
    for image_id, camera in list(cameras.items()):
        # if image_id != "2037":
        #     continue
        render = renderer.render(camera)
        valid_pixels = render[render[..., 3] > 0.5]
        # count number of axis with at least one non-zero value
        is_zero = np.all(valid_pixels[:, :3] == 0, axis=0)
        print(f"{image_id}: 0 axis: {is_zero}")

        # output_filename = f"{output_dir}/depth_{image_id}.png"
        # renderer.visualize_depth(camera, output_filename)
    
    renderer.close()