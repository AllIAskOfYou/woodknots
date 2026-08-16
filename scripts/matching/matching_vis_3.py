import sys
import os
import ctypes
import json
import numpy as np
from pathlib import Path

# Set platform before importing OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'glfw'

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from mltools.matching.volume import VoxelGrid
from mltools.matching.utils import confusion_matrix, correct_annotations

# ==========================================
# SHADERS
# ==========================================
MESH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texcoord;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
out vec2 v_texcoord;
void main() {
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
}
"""

MESH_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform int u_has_texture;
void main() {
    if (u_has_texture == 1) {
        fragColor = texture(u_texture, v_texcoord);
    } else {
        fragColor = vec4(0.8, 0.8, 0.8, 1.0);
    }
}
"""

SPHERE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_center;
layout(location = 1) in vec2 a_quad_offset;
layout(location = 2) in vec3 a_color;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float u_sphere_radius;
out vec2 v_uv;
out vec3 v_color;
void main() {
    v_uv = a_quad_offset + 0.5;
    v_color = a_color;
    vec3 camera_right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 camera_up    = vec3(view[0][1], view[1][1], view[2][1]);
    vec3 world_position = a_center 
                        + (camera_right * a_quad_offset.x * u_sphere_radius) 
                        + (camera_up * a_quad_offset.y * u_sphere_radius);
    gl_Position = projection * view * model * vec4(world_position, 1.0);
}
"""

SPHERE_FRAGMENT_SHADER = """
#version 330 core
uniform float u_alpha;
in vec2 v_uv;
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec2 coord = v_uv * 2.0 - 1.0;
    float dist = dot(coord, coord);
    if (dist > 1.0) { discard; }
    fragColor = vec4(v_color, u_alpha);
}
"""

# ==========================================
# OBJ/MTL LOADER
# ==========================================
def load_obj_mtl(filepath):
    dir_name = os.path.dirname(filepath)
    vertices, texcoords, faces_by_material = [], [], {}
    material_texture_map = {}
    current_material = "default"
    mtl_path = None

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'mtllib':
                mtl_path = os.path.join(dir_name, values[1])
            elif values[0] == 'usemtl':
                current_material = values[1]
                if current_material not in faces_by_material:
                    faces_by_material[current_material] = []
            elif values[0] == 'v':
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif values[0] == 'vt':
                texcoords.append([float(values[1]), float(values[2])])
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    vi = int(w[0]) - 1
                    ti = int(w[1]) - 1 if len(w) > 1 and w[1] != '' else 0
                    face.append((vi, ti))

                if current_material not in faces_by_material:
                    faces_by_material[current_material] = []
                faces_by_material[current_material].append(face)

    # Parse MTL file
    if mtl_path and os.path.exists(mtl_path):
        with open(mtl_path, 'r') as mf:
            mname = None
            for line in mf:
                parts = line.split()
                if not parts: continue
                if parts[0] == 'newmtl':
                    mname = parts[1]
                elif parts[0] == 'map_Kd' and mname:
                    material_texture_map[mname] = os.path.join(dir_name, parts[1])

    # Build sub-mesh definitions
    sub_meshes = []
    for mat_name, faces in faces_by_material.items():
        data = []
        for face in faces:
            v0, t0 = face[0]
            for i in range(1, len(face) - 1):
                v1, t1 = face[i]
                v2, t2 = face[i+1]
                for vi, ti in [(v0, t0), (v1, t1), (v2, t2)]:
                    pos = vertices[vi]
                    uv = texcoords[ti] if 0 <= ti < len(texcoords) else [0.0, 0.0]
                    data.extend([pos[0], pos[1], pos[2], uv[0], uv[1]])

        if not data:
            continue

        data_np = np.array(data, dtype=np.float32)

        # Load texture for this material group
        tex_file = material_texture_map.get(mat_name)
        tex_id = 0
        has_texture = 0

        if tex_file and os.path.exists(tex_file):
            img = Image.open(tex_file).transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.convert("RGBA").tobytes()
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            has_texture = 1

        sub_meshes.append({
            'data': data_np,
            'vertex_count': len(data_np) // 5,
            'tex_id': tex_id,
            'has_texture': has_texture
        })

    return sub_meshes

# ==========================================
# ANNOTATION OVERLAY HELPERS
# ==========================================
OVERLAY_MAX_SIZE = 220        # px, longest side of the overlay thumbnail
OVERLAY_MASK_COLOR = (255, 60, 60)  # RGB tint applied where the mask is set
OVERLAY_MASK_ALPHA = 0.45     # blend strength of the mask tint
OVERLAY_PADDING = 10          # px, gap from the window edges


def _compose_overlay_rgba(image: np.ndarray, mask) -> np.ndarray:
    """image: (H, W, 3) uint8 array. mask: (H, W) array-like (bool / 0-1 / 0-255),
    or None. Returns (H, W, 4) uint8 RGBA with the mask alpha-blended on top
    in OVERLAY_MASK_COLOR."""
    image = np.asarray(image)
    if image.ndim == 2:  # grayscale safety net
        image = np.stack([image] * 3, axis=-1)
    rgb = image[..., :3].astype(np.float32)

    if mask is not None:
        mask_bool = np.asarray(mask) > 0
        tint = np.array(OVERLAY_MASK_COLOR, dtype=np.float32)
        rgb[mask_bool] = (1.0 - OVERLAY_MASK_ALPHA) * rgb[mask_bool] + OVERLAY_MASK_ALPHA * tint

    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _make_black_rgba(size: int = OVERLAY_MAX_SIZE) -> np.ndarray:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def _resize_keep_aspect(rgba: np.ndarray, max_size: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return rgba
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = Image.fromarray(rgba, mode="RGBA").resize((new_w, new_h), Image.BILINEAR)
    return np.array(img)

# ==========================================
# APPLICATION
# ==========================================
class App:
    MODES = ["Volumes", "Structures", "Pairs"]

    def __init__(self, obj_path, annotations: dict, projections_dir: str, voxel_size: float, images_dir: str, color_palette, corrections_file: str):
        """
        obj_path:      path to the .obj mesh to display.
        voxel_grid:    a VoxelGrid holding one or more volumes.
        color_palette: (K, 3) array of RGB colors in [0, 1] (or [0, 255],
                        auto-detected and normalized below). A point's color
                        is looked up by `id % len(color_palette)`.
        """
        self.obj_path = obj_path
        self.annotations = annotations
        self.voxel_grid = construct_voxel_grid(annotations, projections_dir, voxel_size, save_masks=True)
        self.images_dir = images_dir
        self.images = self._load_images()
        self.num_volumes = len(self.voxel_grid.volumes)

        palette = np.array(color_palette, dtype=np.float32)
        if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
            raise ValueError("color_palette must be a non-empty (K, 3) array of RGB colors")
        if palette.max() > 1.0:
            palette = palette / 255.0
        self.palette = palette

        self.corrections_file = corrections_file
        if not os.path.exists(corrections_file):
            with open(corrections_file, 'w') as f:
                json.dump({}, f)
        with open(corrections_file, 'r') as f:
            self.corrections: list[tuple[int, int]] = json.load(f)
        correct_annotations(self.annotations, self.corrections)


        # Populated by _refresh_selection() once the GL context exists.
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.sphere_alpha = 1.0

        # ------------------------------------------------------------
        # Mode: "Volumes" | "Structures" | "Pairs"
        # ------------------------------------------------------------
        self.mode = "Volumes"

        # -- Volumes mode state --
        # -1 == show every volume in the voxel grid; otherwise show just one.
        self.volume_index = -1

        # -- Structures mode state --
        self.merge_threshold = 0.5
        self.structures = None  # list[list[volume_id]] | None (lazily computed / cached)

        # -- Pairs mode state --
        self.pairs = None  # list[(label, ann_id_i, ann_id_j)] | None (lazily computed / cached)
        self.pair_index = 0
        self.pair_mode = -1

        # Overlay thumbnails shown top-right. A list so Pairs mode can show
        # two side by side (one per annotation in the selected pair).
        self.overlays = [{'label': "", 'rgba': _make_black_rgba()}]
        self.overlay_tex_ids = []  # persistent GL texture ids, reused/grown across refreshes

        self.cam_dist = 5.0
        self.cam_yaw = 45.0
        self.cam_pitch = -30.0
        self.cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.sphere_radius = 0.01
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Edge-detection state for arrow-key navigation (fire once per press).
        self._left_key_prev = False
        self._right_key_prev = False
        self._up_key_prev = False
        self._down_key_prev = False

    def _load_images(self) -> dict[str, np.ndarray]:
        """Load all images referenced in the annotations into a dict of
        {image_name: (H, W, 3) uint8 arrays}."""
        images = {}
        for ann in self.annotations.values():
            image_name = ann['image_name']
            image_id = Path(image_name).stem
            if image_name not in images:
                image_path = os.path.join(self.images_dir, image_id) + ".JPG"
                if os.path.exists(image_path):
                    images[image_name] = np.array(Image.open(image_path))
                else:
                    print(f"Warning: Image file {image_path} does not exist.")
        return images

    def get_annotation_data(self, index: int) -> dict:
        """Returns per-volume annotation data for `index`:
        {"vis": (points_array, ids_array), "image": [name, image, mask]}."""
        vis = self.voxel_grid.visualize_volume(index)
        volume_id = self.voxel_grid.volume_ids[index]
        ann = self.annotations.get(volume_id)
        image_name = ann['image_name']
        image = self.images.get(image_name)
        mask = ann.get('mask', None)
        return {"id": volume_id, "vis": vis, "image": [image_name, image, mask]}

    def _overlay_for_ann_id(self, ann_id, label_prefix: str = "") -> dict:
        """Builds an overlay thumbnail dict for a given annotation id, driven
        purely off self.annotations/self.images (not the voxel grid), so it
        still works for annotations that produced no valid 3D points."""
        ann = self.annotations.get(ann_id, {})
        image_name = ann.get('image_name', '?')
        image = self.images.get(image_name)
        mask = ann.get('mask', None)
        if image is not None:
            rgba = _resize_keep_aspect(_compose_overlay_rgba(image, mask), OVERLAY_MAX_SIZE)
        else:
            rgba = _make_black_rgba()
        return {'label': f"{label_prefix}{image_name}:{ann_id}", 'rgba': rgba}

    def _points_for_ann_id(self, ann_id) -> np.ndarray:
        """Snapped voxel-center points for a given annotation id, or an empty
        array if that annotation never made it into the voxel grid (e.g. it
        had no valid projected points)."""
        if ann_id not in self.voxel_grid.volume_ids:
            return np.zeros((0, 3), dtype=np.float32)
        idx = self.voxel_grid.volume_ids.index(ann_id)
        points, _ = self.voxel_grid.visualize_volume(idx)
        return points

    # ------------------------------------------------------------
    # Mode / selection
    # ------------------------------------------------------------
    def set_mode(self, new_mode: str):
        if new_mode == self.mode:
            return
        assert new_mode in self.MODES, f"Unknown mode: {new_mode}"
        self.mode = new_mode
        self._refresh_selection()

    def set_volume_index(self, new_index: int):
        new_index = max(-1, min(self.num_volumes - 1, new_index))
        if new_index == self.volume_index:
            return
        self.volume_index = new_index
        self._refresh_selection()

    def set_merge_threshold(self, new_threshold: float):
        if new_threshold == self.merge_threshold:
            return
        self.merge_threshold = new_threshold
        # Both structures and the pair classification (tp/fp/fn) depend on
        # the threshold, so invalidate both and recompute lazily.
        self.structures = None
        self.pairs = None
        self._refresh_selection()

    def set_pair_index(self, new_index: int):
        num_pairs = len(self.pairs) if self.pairs else 0
        new_index = max(0, min(num_pairs - 1, new_index)) if num_pairs > 0 else 0
        if new_index == self.pair_index:
            return
        self.pair_index = new_index
        self._refresh_selection()

    def _ensure_structures_computed(self):
        if self.structures is None:
            self.structures = self.voxel_grid.find_structures(
                merge_threshold=self.merge_threshold, map_indexes=True
            )

    def _ensure_pairs_computed(self):
        if self.pairs is None:
            self._ensure_structures_computed()
            tps, fps, fns, tns = confusion_matrix(self.annotations, self.structures, return_pairs=True)
            self.pairs = (
                [("FP", i, j) for i, j in fps]
                # + [("TP", i, j) for i, j in tps]
                + [("FN", i, j) for i, j in fns]
            )
            self.pair_index = max(0, min(self.pair_index, len(self.pairs) - 1)) if self.pairs else 0

    def _colors_for_ids(self, ids_array: np.ndarray) -> np.ndarray:
        ids_array = np.asarray(ids_array).astype(np.int64)
        palette_indices = ids_array % len(self.palette)
        return self.palette[palette_indices].astype(np.float32)

    def _refresh_selection(self):
        if self.mode == "Volumes":
            self._refresh_volumes_selection()
        elif self.mode == "Structures":
            self._refresh_structures_selection()
        elif self.mode == "Pairs":
            self._refresh_pairs_selection()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._upload_sphere_instance_data()
        self._upload_overlay_textures()

    def _refresh_volumes_selection(self):
        """Original behaviour: show every volume at once (index == -1), or
        step through them one at a time with the image + mask overlay."""
        if self.volume_index == -1:
            points, ids = self.voxel_grid.visualize_volumes()
            self.overlays = [{'label': "(all volumes)", 'rgba': _make_black_rgba()}]
        else:
            data = self.get_annotation_data(self.volume_index)
            points, ids = data["vis"]
            points = np.asarray(points)
            ids = np.asarray(ids)
            name, image, mask = data["image"]
            rgba = _resize_keep_aspect(_compose_overlay_rgba(image, mask), OVERLAY_MAX_SIZE)
            self.overlays = [{'label': f"{name}:{data['id']}", 'rgba': rgba}]

        self.points = points.astype(np.float32)
        self.colors = self._colors_for_ids(ids)

    def _refresh_structures_selection(self):
        """Groups volumes into structures via VoxelGrid.find_structures() and
        visualizes them all at once, colored by structure id."""
        self._ensure_structures_computed()

        if not self.structures:
            self.points = np.zeros((0, 3), dtype=np.float32)
            self.colors = np.zeros((0, 3), dtype=np.float32)
            self.overlays = [{'label': "(no structures)", 'rgba': _make_black_rgba()}]
            return

        points, ids = self.voxel_grid.visualize_structures(self.structures)
        self.points = points.astype(np.float32)
        self.colors = self._colors_for_ids(ids)
        self.overlays = [{
            'label': f"{len(self.structures)} structures (threshold={self.merge_threshold:.2f})",
            'rgba': _make_black_rgba(),
        }]

    def _refresh_pairs_selection(self):
        """Classifies pairs of overlapping annotations as TP/FP/FN (via
        utils.confusion_matrix) and visualizes only the currently selected
        pair, with one overlay thumbnail per annotation in the pair."""
        self._ensure_pairs_computed()

        if not self.pairs:
            self.points = np.zeros((0, 3), dtype=np.float32)
            self.colors = np.zeros((0, 3), dtype=np.float32)
            self.overlays = [{'label': "(no pairs)", 'rgba': _make_black_rgba()}]
            return

        self.pair_index = max(0, min(len(self.pairs) - 1, self.pair_index))
        label, ann_i, ann_j = self.pairs[self.pair_index]

        pts_i = self._points_for_ann_id(ann_i)
        pts_j = self._points_for_ann_id(ann_j)
        points = []
        ids = []
        if self.pair_mode < 1 and len(pts_i) > 0:
            points.append(pts_i)
            ids.append(np.full(len(pts_i), ann_i, dtype=np.int64))
        if self.pair_mode != 0 and len(pts_j) > 0:
            points.append(pts_j)
            ids.append(np.full(len(pts_j), ann_j, dtype=np.int64))
        points = np.vstack(points) if points else np.zeros((0, 3), dtype=np.float32)
        ids = np.concatenate(ids) if ids else np.zeros((0,), dtype=np.int64)

        self.points = points.astype(np.float32)
        self.colors = self._colors_for_ids(ids)

        self.overlays = [
            self._overlay_for_ann_id(ann_i, label_prefix=f"[{label}] A: "),
            self._overlay_for_ann_id(ann_j, label_prefix=f"[{label}] B: "),
        ]

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def run(self):
        if not glfw.init():
            sys.exit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(1200, 800, "Textured Mesh & Spheres Viewer", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit(1)

        glfw.make_context_current(self.window)

        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.15, 0.15, 0.18, 1.0)

        self.mesh_shader = compileProgram(
            compileShader(MESH_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(MESH_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        self.sphere_shader = compileProgram(
            compileShader(SPHERE_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(SPHERE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        self.mesh_submeshes = load_obj_mtl(self.obj_path)
        self.setup_mesh_buffers()
        self.setup_sphere_buffers()

        # Now that GL buffers/textures exist, populate the initial selection
        # (defaults to mode == "Volumes", volume_index == -1, i.e. every volume).
        self._refresh_selection()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            imgui.new_frame()

            # Process mouse rotation directly from ImGui IO state each frame
            self.handle_mouse_input()
            self.handle_keyboard_input()

            self.build_ui()
            self.build_overlay_panel()

            self.render()

            imgui.render()
            self.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()

    def handle_mouse_input(self):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        dx = io.mouse_delta.x
        dy = io.mouse_delta.y

        # Check if Shift key is pressed via ImGui IO or GLFW directly
        shift_pressed = io.key_shift or (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or (glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        if io.mouse_down[0]:
            if shift_pressed:
                # Pan / Drag target (Shift + Left Click)
                rad_pitch = np.radians(self.cam_pitch)
                rad_yaw = np.radians(self.cam_yaw)

                forward = -np.array([
                    np.cos(rad_pitch) * np.sin(rad_yaw),
                    np.sin(rad_pitch),
                    np.cos(rad_pitch) * np.cos(rad_yaw)
                ])
                forward /= np.linalg.norm(forward)
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, world_up)
                right_norm = np.linalg.norm(right)
                if right_norm > 1e-5:
                    right /= right_norm
                up = np.cross(right, forward)

                pan_speed = self.cam_dist * 0.002
                self.cam_target -= right * dx * pan_speed
                self.cam_target += up * dy * pan_speed
            else:
                # Rotate (Left Click alone)
                if dx != 0.0 or dy != 0.0:
                    self.cam_yaw -= dx * 0.3
                    self.cam_pitch += dy * 0.3
                    self.cam_pitch = max(-89.0, min(89.0, self.cam_pitch))

        # Check mouse wheel for zooming directly from ImGui IO
        if io.mouse_wheel != 0.0:
            self.cam_dist -= min(10, self.cam_dist ** 2) * io.mouse_wheel * 0.1
            self.cam_dist = max(0.01, min(100.0, self.cam_dist))

    def handle_keyboard_input(self):
        """Left/Right arrow keys step volume_index down/up by one, wrapping
        through -1 (all volumes) at the low end. Fires once per key press,
        not continuously while held. Only active in "Volumes" mode."""
        io = imgui.get_io()
        if io.want_capture_keyboard:
            # Don't let a stale "pressed" edge fire once focus/mode returns.
            self._left_key_prev = False
            self._right_key_prev = False
            self._up_key_prev = False
            self._down_key_prev = False
            return

        if self.mode == "Volumes":
            left = glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS
            right = glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS

            if left and not self._left_key_prev:
                self.set_volume_index(self.volume_index - 1)
            if right and not self._right_key_prev:
                self.set_volume_index(self.volume_index + 1)

            self._left_key_prev = left
            self._right_key_prev = right
        elif self.mode == "Pairs":
            left = glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS
            right = glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS
            up = glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS
            down = glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS

            if left and not self._left_key_prev:
                self.set_pair_index(self.pair_index - 1)
            if right and not self._right_key_prev:
                self.set_pair_index(self.pair_index + 1)

            if up and not self._up_key_prev:
                self.pair_mode = (self.pair_mode+ + 2) % 3 - 1
                self._refresh_selection()
            if down and not self._down_key_prev:
                self.pair_mode = (self.pair_mode + 3) % 3 - 1
                self._refresh_selection()

            self._left_key_prev = left
            self._right_key_prev = right
            self._up_key_prev = up
            self._down_key_prev = down

    # (Keep mouse_button_callback and mouse_callback methods or remove them since they are replaced by handle_mouse_input)

    def correct_annotations(self, ann_i, ann_j):
        self.corrections.append((ann_i, ann_j))
        with open(self.corrections_file, 'w') as f:
            json.dump(self.corrections, f)
        correct_annotations(self.annotations, self.corrections)

    def build_ui(self):
        pos_x = OVERLAY_PADDING
        pos_y = OVERLAY_PADDING

        imgui.set_next_window_pos(imgui.ImVec2(pos_x, pos_y), imgui.Cond_.always)
        imgui.begin("Control Panel", True, imgui.WindowFlags_.always_auto_resize)
        imgui.text(f"Loaded Points: {len(self.points)}")
        imgui.text(f"Volumes: {self.num_volumes}")
        imgui.separator()

        # Mode selector: click to choose between Volumes / Structures / Pairs.
        current_mode_idx = self.MODES.index(self.mode)
        changed_mode, new_mode_idx = imgui.combo("Mode", current_mode_idx, self.MODES)
        if changed_mode:
            self.set_mode(self.MODES[new_mode_idx])

        imgui.separator()

        if self.mode == "Volumes":
            # Volume Index: -1 shows all volumes, otherwise just the one selected.
            # Also steppable with the Left/Right arrow keys (see handle_keyboard_input).
            changed_vol, new_vol = imgui.slider_int(
                "Index", self.volume_index, -1, max(self.num_volumes - 1, -1)
            )
            if changed_vol:
                self.set_volume_index(new_vol)

        elif self.mode == "Structures":
            changed_thresh, new_thresh = imgui.slider_float(
                "Merge Threshold", self.merge_threshold, 0.0, 1.0
            )
            if changed_thresh:
                self.set_merge_threshold(new_thresh)
            self._ensure_structures_computed()
            imgui.text(f"Structures found: {len(self.structures)}")

        elif self.mode == "Pairs":
            changed_thresh, new_thresh = imgui.slider_float(
                "Merge Threshold", self.merge_threshold, 0.0, 1.0
            )
            if changed_thresh:
                self.set_merge_threshold(new_thresh)

            self._ensure_pairs_computed()
            num_pairs = len(self.pairs)
            imgui.text(f"Pairs (FP/FN): {num_pairs}")

            changed_pair, new_pair = imgui.slider_int(
                "Pair Index", self.pair_index, 0, max(num_pairs - 1, 0)
            )
            if changed_pair:
                self.set_pair_index(new_pair)

            changed_pair_mode, new_pair_mode = imgui.slider_int(
                "Pair Mode", self.pair_mode, -1, 1
            )
            if changed_pair_mode:
                self.pair_mode = new_pair_mode
                self._refresh_selection()

            # add a button to correct the annotation of current pair on button release
            if num_pairs > 0:
                label, ann_i, ann_j = self.pairs[self.pair_index]
                if imgui.button("Correct Label"):
                    self.correct_annotations(ann_i, ann_j)
                    # Refresh pairs after correction
                    self.pairs = None
                    self._refresh_selection()
                    

        imgui.separator()
        imgui.text(f"Spheres")
        imgui.separator()

        # Sphere Radius Slider
        changed_rad, new_rad = imgui.slider_float("Radius", self.sphere_radius, 0.001, 0.02)
        if changed_rad:
            self.sphere_radius = new_rad
        

        # Sphere Alpha Slider
        changed_alpha, new_alpha = imgui.slider_float("Alpha", self.sphere_alpha, 0.0, 1.0)
        if changed_alpha:
            self.sphere_alpha = new_alpha

        imgui.end()

    def build_overlay_panel(self):
        """Small always-on-top panel in the top-right corner. Shows one
        thumbnail (image name + image/mask composite) in Volumes/Structures
        mode, and two side-by-side thumbnails (one per annotation) in Pairs
        mode."""
        io = imgui.get_io()
        display_w, display_h = io.display_size.x, io.display_size.y
        max_tex_w = max((item.get('tex_size', (0, 0))[0] for item in self.overlays), default=0)
        window_w = max(max_tex_w, 140) + OVERLAY_PADDING * 2
        pos_x = display_w - window_w - OVERLAY_PADDING
        pos_y = OVERLAY_PADDING

        imgui.set_next_window_pos(imgui.ImVec2(pos_x, pos_y), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.65)
        flags = (
            imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.always_auto_resize
        )
        imgui.begin("##volume_overlay", False, flags)
        for i, item in enumerate(self.overlays):
            if i > 0:
                imgui.separator()
            imgui.text(item['label'])
            tex_w, tex_h = item.get('tex_size', (0, 0))
            tex_id = item.get('tex_id')
            if tex_id is not None and tex_w > 0 and tex_h > 0:
                imgui.image(imgui.ImTextureRef(tex_id), imgui.ImVec2(float(tex_w), float(tex_h)))
        imgui.end()

    def setup_mesh_buffers(self):
        self.sub_mesh_renders = []
        for sm in self.mesh_submeshes:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, sm['data'].nbytes, sm['data'], GL_STATIC_DRAW)

            # Position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            # UV attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)

            self.sub_mesh_renders.append({
                'vao': vao,
                'vertex_count': sm['vertex_count'],
                'tex_id': sm['tex_id'],
                'has_texture': sm['has_texture']
            })

    def setup_sphere_buffers(self):
        quad_vertices = np.array([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5], dtype=np.float32)
        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        # Shared per-vertex quad offset (location 1), not instanced.
        quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        # Per-instance center (location 0). Data is filled in later by
        # _upload_sphere_instance_data() once a selection is made, so this
        # is allocated empty (and as GL_DYNAMIC_DRAW, since it's re-uploaded
        # on every volume_index change).
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribDivisor(0, 1)

        # Per-instance color (location 2), looked up from ids % palette.
        self.color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)

    def _upload_sphere_instance_data(self):
        """Re-uploads instance centers/colors after a selection change.
        Reallocates via glBufferData (sizes can differ between selections) —
        no need to delete/recreate the VBOs themselves."""
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _upload_overlay_textures(self):
        """Uploads self.overlays' RGBA buffers into a small pool of reused GL
        textures (grown as needed, e.g. to 2 for Pairs mode), and stamps the
        resulting tex_id/tex_size back onto each overlay dict for rendering."""
        while len(self.overlay_tex_ids) < len(self.overlays):
            self.overlay_tex_ids.append(glGenTextures(1))

        for i, item in enumerate(self.overlays):
            tex_id = self.overlay_tex_ids[i]
            rgba = np.ascontiguousarray(item['rgba'])
            h, w = rgba.shape[:2]
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            glBindTexture(GL_TEXTURE_2D, 0)
            item['tex_id'] = tex_id
            item['tex_size'] = (w, h)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = glfw.get_window_size(self.window)
        aspect = width / height if height > 0 else 1.0

        projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, aspect, 0.1, 100.0)
        rad_pitch = np.radians(self.cam_pitch)
        rad_yaw = np.radians(self.cam_yaw)
        cam_x = self.cam_target[0] + self.cam_dist * np.cos(rad_pitch) * np.sin(rad_yaw)
        cam_y = self.cam_target[1] + self.cam_dist * np.sin(rad_pitch)
        cam_z = self.cam_target[2] + self.cam_dist * np.cos(rad_pitch) * np.cos(rad_yaw)

        view = pyrr.matrix44.create_look_at(eye=[cam_x, cam_y, cam_z], target=self.cam_target, up=[0.0, 1.0, 0.0])
        model = pyrr.matrix44.create_identity(dtype=np.float32)

        # 1. Multi-Texture Mesh Pass
        glUseProgram(self.mesh_shader)
        glUniformMatrix4fv(glGetUniformLocation(self.mesh_shader, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(self.mesh_shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.mesh_shader, "model"), 1, GL_FALSE, model)
        for sm in self.sub_mesh_renders:
            glUniform1i(glGetUniformLocation(self.mesh_shader, "u_has_texture"), sm['has_texture'])
            if sm['has_texture']:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, sm['tex_id'])
                glUniform1i(glGetUniformLocation(self.mesh_shader, "u_texture"), 0)

            glBindVertexArray(sm['vao'])
            glDrawArrays(GL_TRIANGLES, 0, sm['vertex_count'])

        # 2. Spheres Pass
        glDepthMask(GL_FALSE)
        if len(self.points) > 0:
            glUseProgram(self.sphere_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "projection"), 1, GL_FALSE, projection)
            glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "view"), 1, GL_FALSE, view)
            glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "model"), 1, GL_FALSE, model)
            glUniform1f(glGetUniformLocation(self.sphere_shader, "u_alpha"), self.sphere_alpha)
            glUniform1f(glGetUniformLocation(self.sphere_shader, "u_sphere_radius"), self.sphere_radius)
            glBindVertexArray(self.sphere_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, len(self.points))
        glDepthMask(GL_TRUE)


if __name__ == "__main__":
    from mltools.matching.utils import parse_coco_annotations, construct_voxel_grid

    # obj_filename = "data/matching/textured_mesh/texturedMesh.obj"
    # projections_dir = "data/matching/resized_1024_3d"
    obj_filename = "data/matching/textured_mesh_v2/texturedMesh.obj"
    projections_dir = "data/matching/resized_1024_3d_v3_distortion"
    ann_filename = "data/matching/annotations/instances_default.json"
    images_dir = "data/matching/resized_1024"
    corrections_file = "data/matching/annotations/corrections.json"
    voxel_size = 0.005
    annotations = parse_coco_annotations(ann_filename)

    color_palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 255, 255], [0, 0, 0], [128, 128, 128],
        [255, 105, 180], [75, 0, 130], [255, 140, 0]
    ]

    App(obj_filename, annotations, projections_dir, voxel_size, images_dir, color_palette, corrections_file).run()