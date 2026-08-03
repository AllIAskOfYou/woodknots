"""
vis.py

Reads Meshroom camera parameters (via utils.parse_meshroom_cameras),
renders the reconstructed mesh from every camera's viewpoint using
render.MeshRenderer, and stores a normalized depth map (Euclidean
distance from the camera center to the visible surface) as a PNG for
each image in the reconstruction.

Usage:
    python vis.py path/to/cameras.sfm path/to/mesh.obj path/to/output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mltools.matching.render import MeshRenderer
from mltools.matching.utils import parse_meshroom_cameras


def compute_depth_map(position: np.ndarray, mask: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Per-pixel Euclidean distance from the camera center C to the 3D
    surface point visible at that pixel. Pixels where the mesh was not
    hit are set to 0.
    """
    depth = np.linalg.norm(position - C.reshape(1, 1, 3), axis=-1)
    depth = depth.astype(np.float32)
    depth[~mask] = 0.0
    return depth


def normalize_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Min-max normalize the *valid* depth values to the [0, 255] uint8
    range (closer surfaces map to lower values). Background pixels are 0.
    """
    out = np.zeros_like(depth, dtype=np.uint8)
    if not np.any(mask):
        return out

    valid = depth[mask]
    d_min, d_max = float(valid.min()), float(valid.max())

    if d_max - d_min < 1e-8:
        out[mask] = 255
        return out

    normalized = (depth - d_min) / (d_max - d_min)
    out[mask] = np.clip(normalized[mask] * 255.0, 0, 255).astype(np.uint8)
    return out


def save_depth_png(depth_u8: np.ndarray, out_path: Path) -> None:
    Image.fromarray(depth_u8, mode="L").save(out_path)


def visualize_depths(
    sfm_path: str | Path,
    obj_path: str | Path,
    output_dir: str | Path,
    near: float = 0.01,
    far: float = 1000.0,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = parse_meshroom_cameras(sfm_path)
    if not cameras:
        raise RuntimeError(f"No cameras found in {sfm_path}")

    with MeshRenderer(obj_path) as renderer:
        for img_name, cam in cameras.items():
            if img_name != "2041.JPG":
                continue
            K, R, C = cam["K"], cam["R"], cam["C"]
            width, height = cam["width"], cam["height"]
            result = renderer.render(K, R, C, width, height, near=near, far=far)
            depth = compute_depth_map(result["position"], result["mask"], C)
            depth_u8 = normalize_depth(depth, result["mask"])

            out_path = output_dir / f"{Path(img_name).stem}_depth.png"
            save_depth_png(depth_u8, out_path)

            hit_ratio = result["mask"].mean()
            print(f"[ok] {img_name} -> {out_path.name}  ({width}x{height}, {hit_ratio:.1%} hit)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sfm_path", help="Path to Meshroom cameras.sfm")
    p.add_argument("obj_path", help="Path to the reconstructed .obj mesh")
    p.add_argument("output_dir", help="Directory to write depth PNGs to")
    p.add_argument("--near", type=float, default=0.01, help="Near clip plane")
    p.add_argument("--far", type=float, default=1000.0, help="Far clip plane")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_depths(
        args.sfm_path,
        args.obj_path,
        args.output_dir,
        near=args.near,
        far=args.far,
    )