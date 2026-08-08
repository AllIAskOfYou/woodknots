import os
os.environ['PYOPENGL_PLATFORM'] = 'glfw'  # Ensure GLFW is used for OpenGL context creation

import ctypes
import numpy as np
import glfw
from OpenGL.GL import *
import trimesh
from PIL import Image
import matplotlib as mpl

# ==========================================
# 1. CUSTOM SHADER SOURCE CODE
# ==========================================
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;

uniform mat4 u_projection;
uniform mat4 u_view;

out vec3 frag_world_pos;

void main() {
    frag_world_pos = in_position;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 frag_world_pos;
out vec4 out_color;

void main() {
    out_color = vec4(frag_world_pos, 1.0);
}
"""

# ==========================================
# 2. MESH RENDERER CLASS
# ==========================================
class MeshRenderer:
    def __init__(self, obj_path, default_width=800, default_height=600):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.width = default_width
        self.height = default_height

        self.window = glfw.create_window(self.width, self.height, "Offscreen Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # Load mesh once
        mesh = trimesh.load(obj_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        vertices = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)
        self.vertex_count = len(vertices)

        # OpenGL Context Setup
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Compile Shaders and Setup Buffers once
        self.shader_program = self._create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
        glUseProgram(self.shader_program)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Matplotlib colormap setup
        self.colormap = mpl.colormaps['viridis']

        # ------------------------------------------------------------
        # Offscreen FBO with a floating-point color attachment.
        #
        # The default window framebuffer is a fixed-point RGBA8 target,
        # so any color values you write (including "fake colors" that
        # are actually world-space positions) get clamped to [0, 1]
        # before they're stored — regardless of what type you later
        # pass to glReadPixels. To get unclamped float data out, you
        # must render into a texture with a floating-point internal
        # format (GL_RGBA32F) attached to an FBO.
        # ------------------------------------------------------------
        self.fbo = glGenFramebuffers(1)
        self.color_tex = glGenTextures(1)
        self.depth_rbo = glGenRenderbuffers(1)
        self._fbo_size = (0, 0)  # force allocation on first render()
        self._allocate_fbo_storage(self.width, self.height)

    def _allocate_fbo_storage(self, w, h):
        """(Re)allocate the FBO's color/depth attachments for size (w, h)."""
        if self._fbo_size == (w, h):
            return

        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        # GL_RGBA32F is the key change: a non-normalized float format,
        # so values outside [0, 1] (including negative world coords)
        # are stored exactly instead of being clamped.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise RuntimeError(f"Framebuffer incomplete, status: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        self._fbo_size = (w, h)

    def _compile_shader(self, shader_type, source):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(f"Shader compilation error: {glGetShaderInfoLog(shader).decode()}")
        return shader

    def _create_shader_program(self, vs_source, fs_source):
        vs = self._compile_shader(GL_VERTEX_SHADER, vs_source)
        fs = self._compile_shader(GL_FRAGMENT_SHADER, fs_source)
        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(f"Program linking error: {glGetProgramInfoLog(program).decode()}")
        glDeleteShader(vs)
        glDeleteShader(fs)
        return program

    def _get_projection_matrix(self, focal_length, width, height, sensor_width, sensor_height, cx=0.0, cy=0.0, near=0.1, far=100.0):
        """
        Computes an asymmetric OpenGL perspective projection matrix incorporating
        principal point offsets (cx, cy) measured from the image center in pixels.
        """
        # Physical size scaling factor for pixels to physical units on the sensor
        dx = sensor_width / width
        dy = sensor_height / height

        # Standard symmetric frustum bounds at the near plane
        xmin = -0.5 * sensor_width * (near / focal_length)
        xmax =  0.5 * sensor_width * (near / focal_length)
        ymin = -0.5 * sensor_height * (near / focal_length)
        ymax =  0.5 * sensor_height * (near / focal_length)

        # Convert pixel offsets from center into camera/sensor space units
        # Note: If your cx/cy positive Y points down in your dataset, change to -cy * dy * ...
        x_offset = cx * dx * (near / focal_length)
        y_offset = cy * dy * (near / focal_length)

        # Shift frustum boundaries accordingly (asymmetric frustum)
        left   = xmin + x_offset
        right  = xmax + x_offset
        bottom = ymin + y_offset
        top    = ymax + y_offset

        # Construct standard asymmetric OpenGL frustum matrix
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = (2.0 * near) / (right - left)
        proj[1, 1] = (2.0 * near) / (top - bottom)

        proj[0, 2] = (right + left) / (right - left)
        proj[1, 2] = (top + bottom) / (top - bottom)

        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2.0 * far * near) / (far - near)
        proj[3, 2] = -1.0

        return proj.T

    def _get_view_matrix(self, position, rotation_matrix):
        R_cw = np.array(rotation_matrix, dtype=np.float32)[:3, :3]
        t = np.array(position, dtype=np.float32)

        R_wc = R_cw.T
        t_cam = -np.dot(R_wc, t)

        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[:3, :3] = R_wc
        view_matrix[:3, 3] = t_cam
        return view_matrix.T

    def render(self, cam_params):
        """Renders the loaded mesh from a specific camera configuration.

        Returns an (h, w, 4) float32 array of unclamped world-space
        positions (RGB channels) read back from a floating-point FBO.
        """
        glfw.make_context_current(self.window)

        w = cam_params.get('width', self.width)
        h = cam_params.get('height', self.height)

        # Resize FBO attachments if this call uses a different resolution.
        self._allocate_fbo_storage(w, h)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, w, h)
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)

        # Compute matrices for this specific camera
        proj_matrix = self._get_projection_matrix(
            cam_params['focal_length'],
            cam_params['width'],
            cam_params['height'],
            cam_params['sensor_width'],
            cam_params['sensor_height'],
            cam_params['cx'],
            cam_params['cy']
        )
        view_matrix = self._get_view_matrix(cam_params['position'], cam_params['rotation_matrix'])

        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "u_projection"), 1, GL_FALSE, proj_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "u_view"), 1, GL_FALSE, view_matrix)

        # Draw frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        # Read back pixels from the FBO's float texture (unclamped).
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixel_data = glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        pixel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((h, w, 4))
        pixel_array = np.flip(pixel_array, axis=0)

        return pixel_array

    def __del__(self):
        # Best-effort cleanup of GL resources.
        try:
            if getattr(self, 'fbo', None):
                glDeleteFramebuffers(1, [self.fbo])
            if getattr(self, 'color_tex', None):
                glDeleteTextures(1, [self.color_tex])
            if getattr(self, 'depth_rbo', None):
                glDeleteRenderbuffers(1, [self.depth_rbo])
        except Exception:
            pass

    def visualize_depth(self, cam_params, pos=None, output_png_path=None):
        pixel_array = self.render(cam_params)
        world_pos_map = pixel_array[..., :3]
        valid_mask = pixel_array[..., 3] > 0.5
        print(f"Unique positions for visualization: {np.unique(world_pos_map[valid_mask], axis=0).shape[0]}")

        # Compute depth relative to camera center
        if pos is not None:
            cam_pos = np.array(pos, dtype=np.float32)
        else:
            cam_pos = np.array(cam_params['position'], dtype=np.float32)
        distance_map = np.linalg.norm(world_pos_map - cam_pos, axis=-1)

        min_dist = distance_map[valid_mask].min() if np.any(valid_mask) else 0.0
        max_dist = distance_map[valid_mask].max() if np.any(valid_mask) else 1.0
        print(f"Depth range for visualization: min={min_dist}, max={max_dist}")
        # min_dist = 0.3
        # max_dist = 0.5
        normalized_depth = np.clip((distance_map - min_dist) / (max_dist - min_dist), 0.0, 1.0)

        # colored_depth = self.colormap(1 - normalized_depth)
        colored_depth = (1-normalized_depth).reshape(*normalized_depth.shape, 1) * np.array([1.0, 1.0, 1.0, 1.0])  # Grayscale
        colored_depth[..., 3][valid_mask] = 1.0
        colored_depth[~valid_mask] = [0.0, 0.0, 0.0, 1.0]

        final_img_uint8 = (colored_depth * 255).astype(np.uint8)
        image = Image.fromarray(final_img_uint8, mode="RGBA")
        if output_png_path is not None:
            os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
            image.save(output_png_path)
            print(f"Render successfully saved to {output_png_path}")
        return image

    def visualize_3d_position(self, cam_params, output_png_path=None):
        """Renders the mesh and saves the 3D world positions as a color-coded image."""
        pixel_array = self.render(cam_params)
        world_pos_map = pixel_array[..., :3]
        valid_mask = pixel_array[..., 3] > 0.5
        print(f"[Info] Samples of world positions (valid): {world_pos_map[valid_mask][:5]}")

        # Extract valid 3D coordinates
        valid_coords = world_pos_map[valid_mask]
        
        if len(valid_coords) == 0:
            print("[Warning] No valid geometry found to visualize.")
            return

        # Map 3D coordinates into RGB by normalizing each axis (X, Y, Z) independently to [0, 1]
        min_val = np.min(valid_coords, axis=0)
        max_val = np.max(valid_coords, axis=0)
        
        # Avoid division by zero if a dimension is completely flat
        range_val = max_val - min_val
        range_val[range_val == 0.0] = 1.0

        # Create an empty RGB image buffer
        rgb_image = np.zeros((*world_pos_map.shape[:2], 3), dtype=np.float32)
        
        # Normalize valid coordinates to [0, 1] which directly map to R, G, B channels
        normalized_coords = (world_pos_map[valid_mask] - min_val) / range_val
        rgb_image[valid_mask] = normalized_coords

        # Convert to 8-bit image and save via PIL
        img_uint8 = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8)
        img.save(output_png_path)
        print(f"[Info] Saved 3D position visualization to {output_png_path}")

    def close(self):
        """Cleans up GPU buffers, program, and window."""
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.shader_program)
        glfw.destroy_window(self.window)
        glfw.terminate()