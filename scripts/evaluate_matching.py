import json
from pathlib import Path
from typing import Any, Dict
import cv2
import numpy as np
import trimesh

from mltools.matching.render import MeshPositionRasterizer


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
    
    # Define coordinate correction transform (matching your working PyVista setup: flip Y and Z)
    T_corr = np.eye(3, dtype=np.float64)
    T_corr[1, 1] = -1.0
    T_corr[2, 2] = -1.0

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

        transform = pose_item["pose"]["transform"]

        # Meshroom stores rotation in column-major order (order="F")
        R_raw = np.array(
            [float(x) for x in transform["rotation"]], dtype=np.float64
        ).reshape(3, 3, order="F")
        C_raw = np.array([float(x) for x in transform["center"]], dtype=np.float64)

        # ✅ Apply coordinate system correction to match mesh alignment
        C = T_corr @ C_raw
        R = T_corr @ (-R_raw)

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
            "view_id": view.get("viewId"),
            "pose_id": pose_id,
        }

    return cameras


def render_view_from_sfm(mesh_path: str | Path, sfm_path: str | Path, target_image_name: str, output_path: str = None):
    # 1. Parse all cameras using the updated parser
    cameras = parse_meshroom_cameras(sfm_path)
    if target_image_name not in cameras:
        raise ValueError(f"Image '{target_image_name}' not found in parsed SfM cameras.")

    cam = cameras[target_image_name]
    
    # Retrieve image dimensions from intrinsics or fallback to standard dimensions
    K = cam["K"]
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    print(f"Loaded camera '{target_image_name}' successfully.")

    # 2. Initialize the ModernGL rasterizer
    rasterizer = MeshPositionRasterizer(mesh_path, sfm_path=sfm_path, near=0.01, far=1000.0)

    # 3. Render points and mask from this camera perspective
    points, mask = rasterizer.render(cam, width=width, height=height)

    # 4. Post-process shading
    canvas = np.full((height, width, 3), 35, dtype=np.uint8)
    if mask.any():
        dp_du = cv2.Sobel(points, cv2.CV_32F, 1, 0, ksize=3)
        dp_dv = cv2.Sobel(points, cv2.CV_32F, 0, 1, ksize=3)
        normals = np.cross(dp_du, dp_dv)
        norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6
        normals /= norm

        light_dir = np.array([0.3, -0.3, 0.9], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        shading = np.clip(np.sum(normals * light_dir, axis=2), 0.2, 1.0)
        shaded_val = (shading * 220).astype(np.uint8)

        mesh_bgr = np.dstack([
            (shaded_val * 0.75).astype(np.uint8),
            (shaded_val * 0.85).astype(np.uint8),
            (shaded_val * 0.95).astype(np.uint8),
        ])
        canvas[mask] = mesh_bgr[mask]

    rasterizer.release()

    # 5. Save or display result
    if output_path:
        cv2.imwrite(output_path, canvas)
        print(f"Render saved to {output_path}")

    return canvas


if __name__ == "__main__":
    mesh_file = Path("data/matching/meshroom/downsampled.obj")
    sfm_file = Path("data/matching/meshroom/cameras.sfm")
    target_img = "2041.JPG"

    rendered_image = render_view_from_sfm(mesh_file, sfm_file, target_img, output_path="data/matching/rendered_2041.png")