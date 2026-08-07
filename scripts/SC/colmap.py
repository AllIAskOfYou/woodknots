#!/usr/bin/env python3
"""
log_reconstruction.py
======================

End-to-end pycolmap pipeline for reconstructing a handheld-captured cylindrical
wood log: sparse SfM -> camera poses -> dense MVS -> mesh, plus a reusable
`LogModel` class that lets you map an (image, pixel) pair to a 3D world
location for annotation transfer and ray casting against the mesh.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import pycolmap
except ImportError as e:
    raise SystemExit(
        "pycolmap is required. Install with `pip install pycolmap` "
        "(or a CUDA-enabled build, e.g., `pip install pycolmap-cuda12`, "
        "to get GPU-accelerated dense stereo)."
    ) from e

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("log_reconstruction")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    image_dir: Path
    workspace: Path

    # Feature extraction
    use_affine_features: bool = True   # affine-covariant SIFT
    max_num_features: int = 16000
    peak_threshold: float = 0.004
    domain_size_pooling: bool = True   # extra scale robustness
    single_camera: bool = True         # True if one physical camera/lens for all shots
    mask_path: Optional[Path] = None   # optional per-image background masks

    # Matching
    guided_matching: bool = True
    max_num_matches: int = 65536

    # Mapper / bundle adjustment
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True

    # Dense MVS
    dense_max_image_size: int = 3200
    patch_match_window_radius: int = 5
    patch_match_num_iterations: int = 5

    # Meshing
    poisson_depth: int = 11
    poisson_trim: float = 7.0

    @property
    def database_path(self) -> Path:
        return self.workspace / "database.db"

    @property
    def sparse_path(self) -> Path:
        return self.workspace / "sparse"

    @property
    def mvs_path(self) -> Path:
        return self.workspace / "mvs"

    @property
    def export_path(self) -> Path:
        return self.workspace / "export"


def get_attr_any(module, *names):
    """Return the first attribute found on `module` matching `names`."""
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"None of {names} found on {module}")


# --------------------------------------------------------------------------
# Stage 1: Feature Extraction
# --------------------------------------------------------------------------

def extract_features(cfg: Config) -> None:
    log.info("Extracting features (SIFT, GPU=%s, affine=%s)", not cfg.use_affine_features, cfg.use_affine_features)

    feat_opts = pycolmap.FeatureExtractionOptions()
    feat_opts.sift.max_num_features = cfg.max_num_features
    feat_opts.sift.peak_threshold = cfg.peak_threshold

    if hasattr(feat_opts.sift, "estimate_affine_shape"):
        feat_opts.sift.estimate_affine_shape = cfg.use_affine_features
    if hasattr(feat_opts.sift, "domain_size_pooling"):
        feat_opts.sift.domain_size_pooling = cfg.domain_size_pooling

    reader_opts = pycolmap.ImageReaderOptions()
    if cfg.mask_path is not None and hasattr(reader_opts, "mask_path"):
        reader_opts.mask_path = str(cfg.mask_path)

    camera_mode = (
        pycolmap.CameraMode.SINGLE if cfg.single_camera else pycolmap.CameraMode.AUTO
    )

    device = pycolmap.Device.auto if cfg.use_affine_features else pycolmap.Device.cuda

    pycolmap.extract_features(
        database_path=str(cfg.database_path),
        image_path=str(cfg.image_dir),
        camera_mode=camera_mode,
        reader_options=reader_opts,
        extraction_options=feat_opts,
        device=device,
    )


# --------------------------------------------------------------------------
# Stage 2: Matching
# --------------------------------------------------------------------------

def match_features(cfg: Config) -> None:
    log.info("Matching features exhaustively on GPU (guided_matching=%s)", cfg.guided_matching)

    # 1. Matching options (guided matching, max matches)
    matching_opts = pycolmap.FeatureMatchingOptions()
    if hasattr(matching_opts, "guided_matching"):
        matching_opts.guided_matching = cfg.guided_matching
    if hasattr(matching_opts, "max_num_matches"):
        matching_opts.max_num_matches = cfg.max_num_matches

    # 2. Pairing options (exhaustive pairing options)
    pairing_opts = pycolmap.ExhaustivePairingOptions()

    pycolmap.match_exhaustive(
        database_path=str(cfg.database_path),
        matching_options=matching_opts,
        pairing_options=pairing_opts,
        device=pycolmap.Device.cuda,
    )


# --------------------------------------------------------------------------
# Stage 3: Sparse Incremental Reconstruction
# --------------------------------------------------------------------------

def run_incremental_mapping(cfg: Config) -> pycolmap.Reconstruction:
    log.info("Running incremental sparse reconstruction (bundle adjustment)")
    cfg.sparse_path.mkdir(parents=True, exist_ok=True)

    OptionsCls = get_attr_any(
        pycolmap, "IncrementalPipelineOptions", "IncrementalMapperOptions"
    )
    options = OptionsCls()

    ba_target = getattr(options, "ba_refine", getattr(options, "mapper", options))
    if hasattr(ba_target, "ba_refine_focal_length"):
        ba_target.ba_refine_focal_length = cfg.ba_refine_focal_length
    if hasattr(ba_target, "ba_refine_principal_point"):
        ba_target.ba_refine_principal_point = cfg.ba_refine_principal_point
    if hasattr(ba_target, "ba_refine_extra_params"):
        ba_target.ba_refine_extra_params = cfg.ba_refine_extra_params

    maps = pycolmap.incremental_mapping(
        database_path=str(cfg.database_path),
        image_path=str(cfg.image_dir),
        output_path=str(cfg.sparse_path),
        options=options,
    )

    if not maps:
        raise RuntimeError(
            "Incremental mapping produced no reconstruction. Check that "
            "images actually overlap and that feature extraction succeeded."
        )

    items = list(maps.items()) if hasattr(maps, "items") else list(enumerate(maps))
    items.sort(key=lambda kv: kv[1].num_reg_images(), reverse=True)
    best_id, best_rec = items[0]

    log.info(
        "Found %d sub-model(s). Using model %s with %d images registered.",
        len(items), best_id, best_rec.num_reg_images(),
    )

    model_dir = cfg.sparse_path / str(best_id)
    if not model_dir.exists():
        model_dir = cfg.sparse_path / "0"
        model_dir.mkdir(parents=True, exist_ok=True)

    best_rec.write(str(model_dir))
    log.info("Reconstruction summary:\n%s", best_rec.summary())
    return best_rec


# --------------------------------------------------------------------------
# Stage 4: Dense MVS
# --------------------------------------------------------------------------

def run_dense_reconstruction(cfg: Config, sparse_model_dir: Path) -> Path:
    log.info("Undistorting images for dense stereo")
    cfg.mvs_path.mkdir(parents=True, exist_ok=True)

    pycolmap.undistort_images(
        output_path=str(cfg.mvs_path),
        input_path=str(sparse_model_dir),
        image_path=str(cfg.image_dir),
        max_image_size=cfg.dense_max_image_size,
    )

    log.info("Running patch-match stereo on GPU")
    PMOptions = get_attr_any(pycolmap, "PatchMatchOptions")
    pm_options = PMOptions()
    if hasattr(pm_options, "window_radius"):
        pm_options.window_radius = cfg.patch_match_window_radius
    if hasattr(pm_options, "num_iterations"):
        pm_options.num_iterations = cfg.patch_match_num_iterations
    if hasattr(pm_options, "gpu_index"):
        pm_options.gpu_index = "-1"

    try:
        pycolmap.patch_match_stereo(
            workspace_path=str(cfg.mvs_path),
            options=pm_options,
        )
    except Exception as e:
        raise RuntimeError(
            "patch_match_stereo failed. This step requires a CUDA-enabled "
            "COLMAP build. If you installed plain PyPI `pycolmap`, install "
            "`pycolmap-cuda12` or build COLMAP with CUDA_ENABLED=ON."
        ) from e

    log.info("Fusing depth maps into a dense point cloud")
    fused_path = cfg.mvs_path / "fused.ply"

    # Fix: output_type="ply" prevents COLMAP from expecting a directory
    pycolmap.stereo_fusion(
        output_path=str(fused_path),
        workspace_path=str(cfg.mvs_path),
        output_type="ply",
    )

    log.info("Dense point cloud written to %s", fused_path)
    return fused_path


# --------------------------------------------------------------------------
# Stage 5: Meshing
# --------------------------------------------------------------------------

def run_meshing(cfg: Config, fused_ply: Path) -> Path:
    log.info("Running Poisson surface reconstruction")
    mesh_path = cfg.export_path / "mesh.ply"
    cfg.export_path.mkdir(parents=True, exist_ok=True)

    PoissonOptions = get_attr_any(pycolmap, "PoissonMeshingOptions")
    options = PoissonOptions()
    if hasattr(options, "depth"):
        options.depth = cfg.poisson_depth
    if hasattr(options, "trim"):
        options.trim = cfg.poisson_trim

    pycolmap.poisson_meshing(
        input_path=str(fused_ply),
        output_path=str(mesh_path),
        options=options,
    )
    log.info("Mesh written to %s", mesh_path)
    return mesh_path


# --------------------------------------------------------------------------
# Stage 6: Export Camera Poses & Intrinsics
# --------------------------------------------------------------------------

def export_cameras_and_poses(cfg: Config, rec: pycolmap.Reconstruction) -> Path:
    cfg.export_path.mkdir(parents=True, exist_ok=True)

    cameras_out = {}
    for cam_id, cam in rec.cameras.items():
        cameras_out[str(cam_id)] = {
            "model": cam.model_name,
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
            "params_info": cam.params_info,
            "calibration_matrix": cam.calibration_matrix().tolist(),
        }

    images_out = {}
    for img_id, img in rec.images.items():
        if not img.has_pose:
            continue
        cam_from_world = img.cam_from_world()
        images_out[str(img_id)] = {
            "name": img.name,
            "camera_id": img.camera_id,
            "quat_xyzw": cam_from_world.rotation.quat.tolist(),
            "rotation_matrix": cam_from_world.rotation.matrix().tolist(),
            "translation": cam_from_world.translation.tolist(),
            "cam_from_world_3x4": cam_from_world.matrix().tolist(),
            "projection_center_world": img.projection_center().tolist(),
            "viewing_direction_world": img.viewing_direction().tolist(),
            "num_points3D": img.num_points3D,
        }

    payload = {
        "cameras": cameras_out,
        "images": images_out,
        "num_points3D": rec.num_points3D(),
    }

    out_path = cfg.export_path / "cameras_and_poses.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote camera intrinsics + poses to %s", out_path)

    text_dir = cfg.export_path / "sparse_text"
    text_dir.mkdir(exist_ok=True)
    rec.write_text(str(text_dir))

    ply_path = cfg.export_path / "sparse_points.ply"
    rec.export_PLY(str(ply_path))
    log.info("Wrote sparse point cloud to %s", ply_path)

    return out_path


# --------------------------------------------------------------------------
# LogModel: Reload Reconstruction & Answer 2D <-> 3D Queries
# --------------------------------------------------------------------------

class LogModel:
    """
    Loads exported reconstruction + mesh and provides:
      - pixel_to_3d(image_name, u, v): unproject pixel into world space and
        intersect ray with mesh -> exact 3D surface point.
      - project_3d_to_pixel(image_name, xyz): reproject a 3D point into image coordinates.
      - cast_ray(origin, direction): generic ray/mesh intersection.
    """

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        export_dir = self.workspace / "export"

        payload = json.loads((export_dir / "cameras_and_poses.json").read_text())
        self.cameras = payload["cameras"]
        self.images = payload["images"]
        self._name_to_image_id = {v["name"]: k for k, v in self.images.items()}

        rec = pycolmap.Reconstruction()
        sparse_dirs = sorted((self.workspace / "sparse").glob("*"))
        if sparse_dirs:
            rec.read(str(sparse_dirs[0]))
        else:
            rec.read(str(export_dir / "sparse_text"))
        self.reconstruction = rec

        self._mesh = None
        self._raycasting_scene = None
        mesh_path = export_dir / "mesh.ply"
        if mesh_path.exists():
            self._load_mesh(mesh_path)
        else:
            log.warning(
                "No mesh.ply found in %s -- pixel_to_3d via raycasting will "
                "be unavailable until dense/meshing stages complete.",
                export_dir,
            )

    def _load_mesh(self, mesh_path: Path) -> None:
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError("open3d is required for raycasting: `pip install open3d`") from e

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        self._mesh = mesh

        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
        self._raycasting_scene = scene

    def _image_by_name(self, image_name: str) -> pycolmap.Image:
        img = self.reconstruction.find_image_with_name(image_name)
        if img is None:
            raise KeyError(f"Image '{image_name}' not found in reconstruction.")
        return img

    def pixel_to_3d(self, image_name: str, u: float, v: float) -> Optional[np.ndarray]:
        if self._raycasting_scene is None:
            raise RuntimeError("No mesh loaded -- run dense/meshing stages first.")

        img = self._image_by_name(image_name)
        cam = self.reconstruction.camera(img.camera_id)

        pt2d = np.array([u, v], dtype=np.float64)
        if hasattr(cam, "cam_ray_from_img"):
            ray_cam = cam.cam_ray_from_img(pt2d)
        elif hasattr(cam, "cam_from_img"):
            ray_cam = cam.cam_from_img(pt2d)
            if len(ray_cam) == 2:
                ray_cam = np.array([ray_cam[0], ray_cam[1], 1.0], dtype=np.float64)
        else:
            raise AttributeError("Camera object lacks unprojection method.")

        cam_from_world = img.cam_from_world()
        world_from_cam = cam_from_world.inverse()
        origin_world = world_from_cam.translation
        direction_world = world_from_cam.rotation.matrix() @ ray_cam
        direction_world = direction_world / np.linalg.norm(direction_world)

        return self._cast_ray_raw(origin_world, direction_world)

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        if self._raycasting_scene is None:
            raise RuntimeError("No mesh loaded -- run dense/meshing stages first.")
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        return self._cast_ray_raw(np.asarray(origin, dtype=np.float64), direction)

    def _cast_ray_raw(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        import open3d as o3d

        rays = np.concatenate([origin, direction]).astype(np.float32).reshape(1, 6)
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = self._raycasting_scene.cast_rays(rays_t)
        t_hit = result["t_hit"].numpy()[0]
        if not np.isfinite(t_hit):
            return None
        return origin + t_hit * direction

    def project_3d_to_pixel(self, image_name: str, xyz: np.ndarray) -> Optional[np.ndarray]:
        img = self._image_by_name(image_name)
        return img.project_point(np.asarray(xyz, dtype=np.float64))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(cfg: Config) -> None:
    cfg.workspace.mkdir(parents=True, exist_ok=True)

    if not cfg.database_path.exists():
        extract_features(cfg)
        match_features(cfg)
    else:
        log.info("Database already exists at %s, skipping extraction/matching.", cfg.database_path)

    existing_models = sorted(cfg.sparse_path.glob("*")) if cfg.sparse_path.exists() else []
    if existing_models:
        log.info("Loading existing sparse model from %s", existing_models[0])
        rec = pycolmap.Reconstruction()
        rec.read(str(existing_models[0]))
        best_model_dir = existing_models[0]
    else:
        rec = run_incremental_mapping(cfg)
        best_model_dir = sorted(cfg.sparse_path.glob("*"))[0]

    export_cameras_and_poses(cfg, rec)

    fused_ply = cfg.mvs_path / "fused.ply"
    if not fused_ply.exists():
        fused_ply = run_dense_reconstruction(cfg, best_model_dir)
    else:
        log.info("Dense point cloud already exists at %s, skipping MVS.", fused_ply)

    mesh_path = cfg.export_path / "mesh.ply"
    if not mesh_path.exists():
        run_meshing(cfg, fused_ply)
    else:
        log.info("Mesh already exists at %s, skipping meshing.", mesh_path)

    log.info("Pipeline complete. Output files written to %s:", cfg.export_path)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path, help="Directory of input images")
    p.add_argument("--workspace", required=True, type=Path, help="Output workspace directory")
    p.add_argument("--masks", type=Path, default=None, help="Optional per-image mask directory")
    p.add_argument(
        "--no-affine-features", action="store_true",
        help="Disable affine SIFT (uses GPU SIFT instead of CPU variant)",
    )
    p.add_argument(
        "--multi-camera", action="store_true",
        help="Set if different cameras/lenses were used across capture",
    )
    args = p.parse_args()
    return Config(
        image_dir=args.images,
        workspace=args.workspace,
        mask_path=args.masks,
        use_affine_features=not args.no_affine_features,
        single_camera=not args.multi_camera,
    )


def main() -> None:
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()