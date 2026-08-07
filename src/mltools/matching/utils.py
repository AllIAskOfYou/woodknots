import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


def load_sfm_data(sfm_path: str | Path) -> dict:
    sfm_path = Path(sfm_path)
    if not sfm_path.exists():
        raise FileNotFoundError(f"SfM file not found at: {sfm_path}")

    with open(sfm_path, "r") as f:
        return json.load(f)

class MeshroomCamerasParser:
    def __init__(self, cameras_path: str):
        self.cameras_path = cameras_path
        self.data = load_sfm_data(cameras_path)

    def _parse_intrinsics(self):
        intrinsics = {}
        for item in self.data["intrinsics"]:
            id = item["intrinsicId"]
            pp = item.get("principalPoint", [0.0, 0.0])
            intrinsics[id] = {
                "width": int(item["width"]),
                "height": int(item["height"]),
                "sensor_width": float(item["sensorWidth"]),
                "sensor_height": float(item["sensorHeight"]),
                "focal_length": float(item["focalLength"]),
                "cx": float(pp[0]),
                "cy": float(pp[1]),
                "distortion_params": np.array(item["distortionParams"], dtype=np.float32)
            }
        return intrinsics
    
    def _parse_poses(self):
        poses = {}
        for item in self.data["poses"]:
            id = item["poseId"]
            transform = item["pose"]["transform"]
            R = np.array(transform["rotation"], dtype=np.float32)
            C = np.array(transform["center"], dtype=np.float32)
            rotation_matrix = np.array([
                [R[0], -R[1], -R[2]],
                [-R[3], R[4], R[5]],
                [-R[6], R[7], R[8]]
            ], dtype=np.float32)
            position = np.array([C[0], -C[1], -C[2]], dtype=np.float32)
            poses[id] = {
                "rotation_matrix": rotation_matrix,
                "position": position
            }
        return poses

    def parse(self):
        intrinsics = self._parse_intrinsics()
        poses = self._parse_poses()
        
        cameras = {}
        for view in self.data["views"]:
            intrinsic = intrinsics.get(view.get("intrinsicId"))
            pose = poses.get(view.get("poseId"))
            if intrinsic is None or pose is None:
                continue
            cameras[view["frameId"]] = {
                "position": pose["position"],
                "rotation_matrix": pose["rotation_matrix"],
                "focal_length": intrinsic["focal_length"],
                "sensor_width": intrinsic["sensor_width"],
                "sensor_height": intrinsic["sensor_height"],
                "width": intrinsic["width"],
                "height": intrinsic["height"],
                "cx": 2*intrinsic["cy"],
                "cy": -2*intrinsic["cx"],
                "distortion_params": intrinsic["distortion_params"]
            }
        return cameras