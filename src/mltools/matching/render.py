"""
render.py

GPU-based mesh renderer (ModernGL) for mapping image pixels from a
Meshroom camera view into 3D world space.

The renderer takes a camera's intrinsic matrix K, rotation R, and
camera center C (Meshroom / OpenCV convention: X_cam = R @ (X_world - C),
with X right, Y down, Z forward) and rasterizes the mesh so that every
pixel of the rendered output stores the *world-space 3D coordinate* of
the surface point visible at that pixel from that camera.

This gives you, for free:
  - a per-pixel image -> 3D mapping (just index into "position")
  - per-pixel depth (Euclidean distance from the camera center)
  - correct occlusion handling (via the GPU depth test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import moderngl
import numpy as np
import trimesh


_VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;

out vec3 v_world_pos;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330

in vec3 v_world_pos;
out vec4 out_position;

void main() {
    // rgb = world-space position of the visible surface point
    // a   = 1.0 marks the pixel as "hit" (used downstream as a validity mask)
    out_position = vec4(v_world_pos, 1.0);
}
"""


def _to_gl(matrix: np.ndarray) -> bytes:
    """
    Pack a numpy 4x4 matrix (used as M @ v, row-major) into the
    column-major float32 byte layout GLSL's mat4 expects, while
    preserving M @ v semantics inside the shader.
    """
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()


def intrinsics_to_gl_projection(
    K: np.ndarray, width: int, height: int, near: float = 0.01, far: float = 1000.0
) -> np.ndarray:
    """
    Build an OpenGL clip-space projection matrix directly from a pinhole
    intrinsic matrix K (as produced by utils.parse_intrinsics), so it can
    be applied to points already expressed in the OpenCV/Meshroom camera
    convention (X right, Y down, Z forward). This avoids having to flip
    the view/rotation matrix -- the Y-flip is baked into the projection.

    Derived so that clip_w = +Z (positive for points in front of the
    camera, as they must be for OpenGL's homogeneous clipping -- a
    negative clip_w silently mangles/clips valid geometry).

    Assumes K is not skewed and uses the standard pinhole form:
        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 2.0 * fx / width
    proj[0, 2] = 2.0 * cx / width - 1.0
    proj[1, 1] = -2.0 * fy / height          # flip Y: image-down -> NDC-up
    proj[1, 2] = 1.0 - 2.0 * cy / height
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = 1.0
    return proj


def pose_to_gl_view(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Build the 4x4 world -> camera matrix from a Meshroom pose:

        X_cam = R @ (X_world - C) = R @ X_world + (-R @ C)
    """
    t = -R @ C
    V = np.eye(4, dtype=np.float64)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


class MeshRenderer:
    """
    Loads an .obj mesh onto the GPU with ModernGL and renders it from
    arbitrary Meshroom camera poses, producing the world-space 3D
    position visible at every pixel.
    """

    def __init__(self, obj_path: str | Path, ctx: Optional["moderngl.Context"] = None):
        self.obj_path = Path(obj_path)
        if not self.obj_path.exists():
            raise FileNotFoundError(f"Mesh not found at: {self.obj_path}")

        # A standalone (headless/offscreen) context. If this fails on your
        # machine, you likely need an EGL-capable GPU driver, or to run
        # under `xvfb-run` / with an OSMesa-enabled moderngl build.
        self.ctx = ctx or moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.program = self.ctx.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )

        self._load_mesh()

        # framebuffer is (re)created lazily to match the requested output size
        self._fbo_size = None
        self._fbo = None
        self._color_tex = None
        self._depth_rbo = None

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _load_mesh(self) -> None:
        mesh = trimesh.load(self.obj_path, process=False, force="mesh")

        vertices = np.asarray(mesh.vertices, dtype="f4")
        faces = np.asarray(mesh.faces, dtype="i4")

        self.bounds = mesh.bounds  # (2, 3) min/max -- handy for picking near/far
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)

        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(faces.tobytes())

        self.vao = self.ctx.vertex_array(
            self.program, [(vbo, "3f", "in_position")], ibo
        )

    def _ensure_framebuffer(self, width: int, height: int) -> None:
        if self._fbo_size == (width, height):
            return

        if self._fbo is not None:
            self._fbo.release()
            self._color_tex.release()
            self._depth_rbo.release()

        # RGBA32F: rgb = world position, a = hit mask (1.0 where mesh drawn)
        self._color_tex = self.ctx.texture((width, height), 4, dtype="f4")
        self._depth_rbo = self.ctx.depth_renderbuffer((width, height))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex], depth_attachment=self._depth_rbo
        )
        self._fbo_size = (width, height)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(
        self,
        K: np.ndarray,
        R: np.ndarray,
        C: np.ndarray,
        width: int,
        height: int,
        near: float = 0.01,
        far: float = 1000.0,
        model: Optional[np.ndarray] = None,
        cull_back_faces: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Render the mesh from the given Meshroom camera (K, R, C) at the
        given output resolution (should generally match the source
        image's width/height so pixel (u, v) in the photo lines up with
        pixel (u, v) of the result).

        Returns:
            {
              "position": (H, W, 3) float32 world-space XYZ per pixel,
              "mask":     (H, W) bool, True where the mesh was hit,
            }
        """
        self._ensure_framebuffer(width, height)
        model = np.eye(4, dtype=np.float64) if model is None else model

        if cull_back_faces:
            self.ctx.enable(moderngl.CULL_FACE)
        else:
            self.ctx.disable(moderngl.CULL_FACE)

        view = pose_to_gl_view(R, C)
        proj = intrinsics_to_gl_projection(K, width, height, near, far)
        mvp = proj @ view @ model

        self.program["u_mvp"].write(_to_gl(mvp))
        self.program["u_model"].write(_to_gl(model))

        self._fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.vao.render(moderngl.TRIANGLES)

        raw = self._color_tex.read(alignment=1)
        buf = np.frombuffer(raw, dtype="f4").reshape(height, width, 4)
        # OpenGL reads back with origin at bottom-left; flip to the usual
        # top-left image convention used by the source photos.
        buf = np.flipud(buf).copy()

        position = buf[..., :3]
        mask = buf[..., 3] > 0.5

        return {"position": position, "mask": mask}

    def unproject_pixel(
        self, u: int, v: int, render_result: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Convenience accessor: world-space 3D point for pixel (u, v)
        in a previous render() result, or None if the mesh wasn't hit."""
        if not render_result["mask"][v, u]:
            return None
        return render_result["position"][v, u]

    def release(self) -> None:
        self.vao.release()
        if self._fbo is not None:
            self._fbo.release()
            self._color_tex.release()
            self._depth_rbo.release()
        self.program.release()

    # context manager convenience
    def __enter__(self) -> "MeshRenderer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()