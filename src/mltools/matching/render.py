"""
Rasterize a mesh from SfM camera views and recover per-pixel 3D world positions,
using ModernGL. Designed for cases where you have 100+ views and want to reuse
one GL context + one uploaded mesh across all of them.

Camera dict format expected (e.g. AliceVision/Meshroom-style):
    {
        "K": (3,3) intrinsics matrix,
        "R": (3,3) world-to-camera rotation,
        "C": (3,)  camera center in world space,
        "t": (3,)  world-to-camera translation, such that Xc = R @ Xw + t
                   (equivalently t = -R @ C),
        "P": (3,4) full projection matrix (not used here -- see note below),
        "view_id": ...,
        "pose_id": ...,
    }

Convention assumed: standard computer-vision camera space
(+X right, +Y down, +Z forward, looking down +Z). This matches AliceVision,
OpenMVG, COLMAP, OpenCV, etc. If your R/t use a different convention, adjust
the CV_TO_GL matrix below.

Note on P: we deliberately do NOT reuse the supplied P matrix. P projects
world points to pixel coordinates (a 3x4 matrix), which is the right thing
for CPU-side projection, but OpenGL rasterization needs a proper 4x4
clip-space matrix with a valid depth mapping for its z-buffer. We rebuild
that from K, R, and t instead. The math is standard and should agree with
P for the x/y pixel location -- if you want to double check, project a
known 3D point with both and compare pixel coordinates.
"""

import numpy as np
import moderngl
import trimesh


# CV camera space (+X right, +Y down, +Z forward) -> GL camera space
# (+X right, +Y up, -Z forward)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
out vec3 world_pos;
void main() {
    world_pos = in_position;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 world_pos;
out vec4 frag_color;
void main() {
    // alpha channel doubles as a "hit mask": 1.0 wherever the mesh
    // actually covers this pixel, 0.0 in cleared/background pixels.
    frag_color = vec4(world_pos, 1.0);
}
"""


class MeshPositionRasterizer:
    """
    Loads a mesh once and rasterizes it from arbitrary camera poses,
    returning a per-pixel world-space 3D position map + hit mask.

    Usage:
        rasterizer = MeshPositionRasterizer("scan.obj")
        for camera in cameras:
            points, mask = rasterizer.render(camera, width=1920, height=1080)
        rasterizer.release()
    """

    def __init__(self, mesh_path, near=0.01, far=1000.0, gl_backend="egl"):
        self.near = near
        self.far = far

        # --- load mesh ---
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            # flatten multi-geometry OBJs into a single mesh
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]
            )
        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.faces = np.asarray(mesh.faces, dtype=np.uint32)

        # --- create headless GL context ---
        self.ctx = self._create_context(gl_backend)
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )

        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        self.ibo = self.ctx.buffer(self.faces.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_position")], self.ibo
        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)  # photogrammetry meshes often
                                               # have inconsistent winding

        self._fbo = None
        self._fbo_size = None

    @staticmethod
    def _create_context(gl_backend):
        try:
            return moderngl.create_context(standalone=True, backend=gl_backend)
        except Exception:
            # fall back to whatever default backend moderngl picks
            # (e.g. CGL on macOS, or a non-EGL GLX context on Linux with a display)
            return moderngl.create_context(standalone=True)

    def _get_fbo(self, width, height):
        if self._fbo is None or self._fbo_size != (width, height):
            if self._fbo is not None:
                self._fbo.release()
            color_tex = self.ctx.texture((width, height), 4, dtype="f4")
            depth_rb = self.ctx.depth_renderbuffer((width, height))
            self._fbo = self.ctx.framebuffer(
                color_attachments=[color_tex], depth_attachment=depth_rb
            )
            self._fbo_size = (width, height)
        return self._fbo

    def _build_mvp(self, camera, width, height):
        K = np.asarray(camera["K"], dtype=np.float64)
        R = np.asarray(camera["R"], dtype=np.float64)
        t = np.asarray(camera["t"], dtype=np.float64).reshape(3)

        # world -> camera (CV convention), as a 4x4
        view_cv = np.eye(4, dtype=np.float64)
        view_cv[:3, :3] = R
        view_cv[:3, 3] = t

        view_gl = _CV_TO_GL.astype(np.float64) @ view_cv

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        near, far = self.near, self.far

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 2.0 * fx / width
        proj[0, 2] = 2.0 * cx / width - 1.0
        proj[1, 1] = 2.0 * fy / height
        proj[1, 2] = 2.0 * cy / height - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        mvp = proj @ view_gl
        return mvp.astype(np.float32)

    def render(self, camera, width, height):
        """
        Returns:
            points: (H, W, 3) float32 array of world-space 3D positions
            mask:   (H, W) bool array, True where the mesh was hit
        """
        mvp = self._build_mvp(camera, width, height)

        fbo = self._get_fbo(width, height)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.TRIANGLES)

        raw = fbo.color_attachments[0].read(dtype="f4")
        data = np.frombuffer(raw, dtype=np.float32).reshape(height, width, 4)

        # OpenGL's row 0 is the bottom of the image; flip to standard
        # top-down image row order.
        data = np.flipud(data)

        points = data[..., :3].copy()
        mask = data[..., 3] > 0.5
        return points, mask

    def release(self):
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
        if self._fbo is not None:
            self._fbo.release()
        self.ctx.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def compute_3d_positions(camera, mesh_path, width, height, near=0.01, far=1000.0):
    """
    Convenience one-off wrapper (creates and tears down its own context).
    For 100+ views, prefer MeshPositionRasterizer directly so the context
    and uploaded mesh are reused across calls.
    """
    with MeshPositionRasterizer(mesh_path, near=near, far=far) as rasterizer:
        return rasterizer.render(camera, width, height)


if __name__ == "__main__":
    # Example usage across many views, reusing the context/mesh:
    #
    # rasterizer = MeshPositionRasterizer("reconstruction.obj")
    # for camera in cameras:  # your list of camera dicts
    #     points, mask = rasterizer.render(camera, width=1920, height=1080)
    #     valid_points = points[mask]  # (N, 3) array of hit 3D positions
    # rasterizer.release()
    pass