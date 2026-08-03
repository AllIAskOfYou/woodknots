from pathlib import Path
import cv2
import numpy as np

from mltools.matching.render import MeshPositionRasterizer


class TrackballCameraApp:

    def __init__(
        self, mesh_path: str | Path, sfm_path: str | Path, width: int = 960, height: int = 720
    ):
        self.width = width
        self.height = height

        # Load rasterizer with mesh and sfm data
        near = 0.01
        far = 1000.0
        self.rasterizer = MeshPositionRasterizer(
            mesh_path=mesh_path, sfm_path=sfm_path, near=near, far=far
        )

        # Estimate scene bounding/median for trackball navigation
        sample_verts = self.rasterizer.vertices
        if len(sample_verts) > 15000:
            indices = np.random.choice(len(sample_verts), 15000, replace=False)
            sample_verts = sample_verts[indices]

        self.target = np.median(sample_verts, axis=0).astype(np.float64)
        distances = np.linalg.norm(sample_verts - self.target, axis=1)
        self.max_extent = float(np.percentile(distances, 95) * 2.0)

        # Trackball state
        self.C_offset = np.array(
            [0.0, 0.0, self.max_extent * 1.5], dtype=np.float64
        )
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Mouse tracking flags
        self.is_dragging = False
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.dirty = True

        self.last_points = None
        self.last_mask = None

        # Render Modes: "shaded", "depth_gray", "positions"
        self.render_modes = ["shaded", "depth_gray", "positions"]
        self.mode_index = 0

        # Intrinsics
        fov_deg = 60.0
        focal = (0.5 * height) / np.tan(np.radians(fov_deg / 2.0))
        self.K = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_camera_dict(self) -> dict:
        C = self.target + self.C_offset
        forward = self.target - C
        dist = np.linalg.norm(forward)
        if dist < 1e-6:
            r3 = np.array([0.0, 0.0, 1.0])
        else:
            r3 = forward / dist

        r1 = np.cross(r3, self.up_vector)
        norm_r1 = np.linalg.norm(r1)
        if norm_r1 < 1e-6:
            r1 = np.array([1.0, 0.0, 0.0])
        else:
            r1 /= norm_r1

        r2 = np.cross(r3, r1)
        r2 /= np.linalg.norm(r2)

        R = np.vstack([r1, r2, r3])
        t = -R @ C
        return {"K": self.K, "R": R, "C": C, "t": t}

    def zoom(self, factor: float, x: int = None, y: int = None):
        cam = self.get_camera_dict()
        C = cam["C"]

        if (
            x is not None
            and y is not None
            and self.last_mask is not None
            and 0 <= y < self.height
            and 0 <= x < self.width
            and self.last_mask[y, x]
        ):
            hit_3d = self.last_points[y, x].astype(np.float64)
            delta = (hit_3d - C) * (1.0 - factor)
        else:
            delta = (self.target - C) * (1.0 - factor)

        C_new = C + delta
        self.target = self.target + delta
        self.C_offset = C_new - self.target
        self.dirty = True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_x, self.last_mouse_y = x, y

        elif event == cv2.EVENT_MBUTTONDOWN or (
            event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY)
        ):
            self.is_panning = True
            self.last_mouse_x, self.last_mouse_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if (
                self.last_mask is not None
                and 0 <= y < self.height
                and 0 <= x < self.width
                and self.last_mask[y, x]
            ):
                hit_3d = self.last_points[y, x].astype(np.float64)
                cam = self.get_camera_dict()
                C = cam["C"]
                self.target = hit_3d
                self.C_offset = C - self.target
                self.dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y

            if self.is_dragging:
                cam = self.get_camera_dict()
                R_cam = cam["R"]
                right_axis = R_cam[0]
                up_axis = -R_cam[1]

                R_x, _ = cv2.Rodrigues(up_axis * (-dx * 0.005))
                R_y, _ = cv2.Rodrigues(right_axis * (-dy * 0.005))
                R_delta = R_y @ R_x

                self.C_offset = R_delta @ self.C_offset
                self.up_vector = R_delta @ self.up_vector
                self.dirty = True

            elif self.is_panning:
                cam = self.get_camera_dict()
                R_cam = cam["R"]
                right_axis = R_cam[0]
                up_axis = -R_cam[1]

                dist = np.linalg.norm(self.C_offset)
                pan_speed = dist * 0.0015
                self.target += (-dx * right_axis + dy * up_axis) * pan_speed
                self.dirty = True

            self.last_mouse_x, self.last_mouse_y = x, y

        elif event == cv2.EVENT_MOUSEWHEEL:
            scale = 0.85 if flags > 0 else 1.15
            self.zoom(scale, x, y)

    def render_frame(self) -> np.ndarray:
        cam = self.get_camera_dict()

        dist = np.linalg.norm(self.C_offset)
        self.rasterizer.near = max(0.001, dist * 0.005)
        self.rasterizer.far = max(100.0, dist * 20.0)

        points, mask = self.rasterizer.render(
            cam, width=self.width, height=self.height
        )

        self.last_points = points
        self.last_mask = mask

        canvas = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        current_mode = self.render_modes[self.mode_index]

        if mask.any():
            if current_mode == "shaded":
                dp_du = cv2.Sobel(points, cv2.CV_32F, 1, 0, ksize=3)
                dp_dv = cv2.Sobel(points, cv2.CV_32F, 0, 1, ksize=3)
                normals = np.cross(dp_du, dp_dv)
                norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6
                normals /= norm

                light_dir = np.array([0.3, -0.3, 0.9], dtype=np.float32)
                light_dir /= np.linalg.norm(light_dir)

                shading = np.clip(np.sum(normals * light_dir, axis=2), 0.2, 1.0)
                shaded_val = (shading * 220).astype(np.uint8)

                mesh_bgr = np.dstack(
                    [
                        (shaded_val * 0.75).astype(np.uint8),
                        (shaded_val * 0.85).astype(np.uint8),
                        (shaded_val * 0.95).astype(np.uint8),
                    ]
                )
                canvas[mask] = mesh_bgr[mask]

            elif current_mode == "depth_gray":
                valid_pts = points[mask]
                depth_vals = np.linalg.norm(valid_pts - cam["C"], axis=1)

                d_min, d_max = depth_vals.min(), depth_vals.max()
                if d_max > d_min:
                    norm_depth = 1.0 - (depth_vals - d_min) / (d_max - d_min)
                else:
                    norm_depth = np.ones_like(depth_vals)

                depth_uint8 = (norm_depth * 255).astype(np.uint8)
                canvas[mask] = np.stack([depth_uint8] * 3, axis=-1)

            elif current_mode == "positions":
                p_min, p_max = points[mask].min(), points[mask].max()
                if p_max > p_min:
                    norm_pts = np.clip((points - p_min) / (p_max - p_min), 0, 1)
                else:
                    norm_pts = np.zeros_like(points)
                canvas[mask] = (norm_pts[mask] * 255).astype(np.uint8)

        mode_label = f"Mode: {current_mode.upper()}"
        cv2.putText(
            canvas,
            mode_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def run(self):
        window_name = "Trackball Mesh & Cameras Viewer"

        cv2.namedWindow(
            window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
        )
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("\n=== Controls ===")
        print("  - Left Mouse Drag         : Rotate Camera")
        print("  - Mouse Scroll Wheel      : Zoom Towards Cursor")
        print("  - '+' / '-' Keys          : Key Zoom Fallback")
        print("  - Double Left Click       : Re-center Pivot to Clicked Point")
        print("  - Middle Drag / Shift+Drag : Pan View")
        print("  - 'd'                     : Toggle Render Mode")
        print("  - 'q' or ESC              : Exit\n")

        try:
            while True:
                if self.dirty:
                    frame = self.render_frame()
                    cv2.imshow(window_name, frame)
                    self.dirty = False

                key = cv2.waitKey(16) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("d"):
                    self.mode_index = (self.mode_index + 1) % len(
                        self.render_modes
                    )
                    self.dirty = True
                elif key in (ord("+"), ord("=")):
                    self.zoom(0.85)
                elif key in (ord("-"), ord("_")):
                    self.zoom(1.15)

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.rasterizer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mesh_file = Path("data/matching/meshroom/downsampled.obj")
    sfm_file = Path("data/matching/meshroom/cameras.sfm")

    app = TrackballCameraApp(mesh_file, sfm_file, width=960, height=720)
    app.run()