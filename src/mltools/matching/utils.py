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


def parse_intrinsics(sfm_data: dict) -> Dict[str, np.ndarray]:
    intrinsics = {}
    for item in sfm_data.get("intrinsics", []):
        intrinsic_id = item["intrinsicId"]

        focal = float(item.get("pxFocalLength", item.get("focalLength", 0.0)))
        pp = item.get("principalPoint", [0.0, 0.0])
        px, py = float(pp[0]), float(pp[1])

        K = np.array([[focal, 0.0, px], [0.0, focal, py], [0.0, 0.0, 1.0]], dtype=np.float64)

        intrinsics[intrinsic_id] = K

    return intrinsics


def parse_meshroom_cameras(sfm_path: str | Path) -> Dict[str, Dict[str, Any]]:
    sfm_data = load_sfm_data(sfm_path)
    intrinsics = parse_intrinsics(sfm_data)

    views_by_pose = {}
    for view in sfm_data.get("views", []):
        pose_id = view.get("poseId")
        if pose_id:
            views_by_pose[pose_id] = view

    cameras = {}
    for pose_item in sfm_data.get("poses", []):
        pose_id = pose_item["poseId"]
        view = views_by_pose.get(pose_id)
        if not view:
            continue

        img_path = view.get("path")
        intrinsic_id = view.get("intrinsicId")
        K = intrinsics.get(intrinsic_id)

        if K is None or not img_path:
            continue

        # width/height are stored as strings in the .sfm (e.g. "1024")
        width = view.get("width")
        height = view.get("height")
        width = int(width) if width is not None else None
        height = int(height) if height is not None else None

        transform = pose_item["pose"]["transform"]

        # ✅ FIXED: Meshroom stores rotation in column-major order (order="F")
        R = np.array(
            [float(x) for x in transform["rotation"]], dtype=np.float64
        ).reshape(3, 3, order="F")
        C = np.array([float(x) for x in transform["center"]], dtype=np.float64)

        # Translation vector t = -R * C
        t = -R @ C

        # 3x4 Projection Matrix P = K @ [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt

        img_name = Path(img_path).name
        cameras[img_name] = {
            "K": K,
            "R": R,
            "C": C,
            "t": t,
            "P": P,
            "width": width,
            "height": height,
            "view_id": view.get("viewId"),
            "pose_id": pose_id,
        }

    return cameras