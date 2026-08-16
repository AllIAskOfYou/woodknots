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
# 1. SCENE SHADERS (pass 1: standard linear MVP render)
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
# 2. DISTORTION SHADERS (pass 2: fullscreen post-process)
#
# Lens distortion is a non-linear polynomial operation and can't be folded
# into the linear 4x4 MVP multiply that produces gl_Position. So instead of
# touching pass 1 at all, we render the normal pinhole (undistorted) image
# to a texture, then run a second fullscreen pass that looks up each
# *distorted* output pixel from the *undistorted* source texture.
#
# radial3k (Brown-Conrady style) coefficients (k1, k2, k3) are the FORWARD
# ("addDistortion") model, defined on normalized camera-space coordinates
# (x, y) of the *ideal* pinhole projection:
#
#     r2 = x^2 + y^2
#     factor = 1 + k1*r2 + k2*r2^2 + k3*r2^3
#     x_d = x * factor   (addDistortion: ideal -> distorted)
#     y_d = y * factor
#
# There is no separately-stored inverse parameter set. For each destination
# (observed/distorted) pixel we know (x_d, y_d) and need the ideal (x, y)
# to sample the pass-1 source texture, so we compute "removeDistortion"
# numerically on the fly from the same forward k1/k2/k3, via a few steps
# of fixed-point iteration (converges quickly for typical camera distortion).
# ==========================================
FULLSCREEN_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 in_pos;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

DISTORTION_FRAGMENT_SHADER = """
#version 330 core
out vec4 out_color;

uniform sampler2D u_scene_tex;
uniform vec2 u_resolution;   // (width, height) in pixels
uniform vec2 u_focal;        // (fx, fy) in pixels
uniform vec2 u_principal;    // (cx_px, cy_px) in pixels, OpenGL y-up convention
uniform vec3 u_k;            // radial3k coefficients (k1, k2, k3)

void main() {
    // gl_FragCoord.xy is the destination (distorted) pixel center, in the
    // same OpenGL y-up pixel convention used for u_principal.
    vec2 px = gl_FragCoord.xy;

    // Normalized distorted/observed coordinates for this destination pixel.
    vec2 xy_d = (px - u_principal) / u_focal;

    // removeDistortion: invert the forward radial3k model via fixed-point
    // iteration to recover the ideal (undistorted) normalized coordinates
    // that addDistortion would have mapped to xy_d.
    vec2 xy = xy_d;
    for (int i = 0; i < 10; i++) {
        float r2 = dot(xy, xy);
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float factor = 1.0 + u_k.x * r2 + u_k.y * r4 + u_k.z * r6;
        // Guard against degenerate/negative factors from extreme coefficients.
        factor = max(factor, 1e-4);
        xy = xy_d / factor;
    }

    // Map the solved ideal coordinates back into pixel space of the pass-1
    // (undistorted) source texture and sample it there.
    vec2 src_px = xy * u_focal + u_principal;
    vec2 uv = src_px / u_resolution;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        out_color = vec4(0.0);
    } else {
        out_color = texture(u_scene_tex, uv);
    }
}
"""

# ==========================================
# 3. MESH RENDERER CLASS
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

        # --- Pass 1: scene shader + mesh buffers ---
        self.scene_program = self._create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
        glUseProgram(self.scene_program)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # --- Pass 2: distortion shader + fullscreen quad buffers ---
        self.distort_program = self._create_shader_program(FULLSCREEN_VERTEX_SHADER, DISTORTION_FRAGMENT_SHADER)

        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)

        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * quad_vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Matplotlib colormap setup
        self.colormap = mpl.colormaps['viridis']

        # ------------------------------------------------------------
        # Two offscreen FBOs, both with floating-point color attachments
        # (GL_RGBA32F) so unclamped world-space "fake color" values survive
        # both the scene pass and the distortion resample pass.
        #
        #   fbo_scene: pass 1 target - standard linear MVP render (undistorted).
        #   fbo_post:  pass 2 target - fullscreen distortion resample of fbo_scene.
        # ------------------------------------------------------------
        self.fbo_scene = glGenFramebuffers(1)
        self.scene_tex = glGenTextures(1)
        self.depth_rbo = glGenRenderbuffers(1)

        self.fbo_post = glGenFramebuffers(1)
        self.post_tex = glGenTextures(1)

        self._fbo_size = (0, 0)
        self._fbo_new_size = (0, 0)
        # self._allocate_fbo_storage({}, self.width, self.height)

    def _find_undistorted_res(self, cam_params, w, h):
        def convert_params_to_pixels(cam_params, w, h):
            dx = cam_params['sensor_width'] / w
            dy = cam_params['sensor_height'] / h
            fx = cam_params['focal_length'] / dx
            fy = cam_params['focal_length'] / dy
            cx_px = w / 2.0 + cam_params['cx']
            cy_px = h / 2.0 + cam_params['cy']
            return np.array([fx, fy], dtype=np.float32), np.array([cx_px, cy_px], dtype=np.float32)
        
        def img2cam(px, pp_px, f_px):
            return (px - pp_px) / f_px

        def cam2img(cam, pp_px, f_px):
            return cam * f_px + pp_px

        def undistort_cam_coords(cam_coords, distortion_params):
            k1, k2, k3 = distortion_params
            new_coords = cam_coords.copy()
            for i in range(10):
                r2 = np.sum(cam_coords**2, axis=-1, keepdims=True)
                r4 = r2**2
                r6 = r4 * r2
                factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
                factor = np.maximum(factor, 1e-4)  # Avoid division by zero or negative factors
                new_coords = cam_coords / factor

            return new_coords

        f_px, pp_px = convert_params_to_pixels(cam_params, w, h)
        w_indices = np.arange(w)
        h_indices = np.arange(h)
        top_border = np.array([w_indices, np.zeros_like(w_indices)], dtype=np.float32).T
        bottom_border = np.array([w_indices, np.full_like(w_indices, h - 1)], dtype=np.float32).T
        left_border = np.array([np.zeros_like(h_indices), h_indices], dtype=np.float32).T
        right_border = np.array([np.full_like(h_indices, w - 1), h_indices], dtype=np.float32).T
        border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border], axis=0)

        cam_coords = img2cam(border_pixels, pp_px, f_px)
        undistorted_cam_coords = undistort_cam_coords(cam_coords, cam_params['distortion_params'])
        undistorted_pixels = cam2img(undistorted_cam_coords, pp_px, f_px)
        bb = np.array([undistorted_pixels.min(axis=0), undistorted_pixels.max(axis=0)], dtype=np.float32)
        w_pre = int(np.ceil(bb[1, 0] - bb[0, 0]))
        h_pre = int(np.ceil(bb[1, 1] - bb[0, 1]))
        w_pre = w + int((w_pre - w) * 1.2)
        h_pre = h + int((h_pre - h) * 1.2)
        return w_pre, h_pre

    def _update_cam_params(self, cam_params, w_pre, h_pre):
        cam_params['sensor_width'] *= (w_pre / cam_params['width'])
        cam_params['sensor_height'] *= (h_pre / cam_params['height'])
        cam_params['width'] = w_pre
        cam_params['height'] = h_pre

    def _allocate_fbo_storage(self, cam_params, w, h):
        """(Re)allocate both FBOs' attachments for size (w, h)."""
        if self._fbo_size == (w, h):
            w_pre, h_pre = self._fbo_new_size
            self._update_cam_params(cam_params, w_pre, h_pre)
            return

        if cam_params.get('distortion_params') is None:
            w_pre, h_pre = w, h
        else:
            w_pre, h_pre = self._find_undistorted_res(cam_params, w, h)
            self._fbo_new_size = (w_pre, h_pre)
            print(f"Undistorted FBO size: {w_pre}x{h_pre} for requested {w}x{h}")
            self._update_cam_params(cam_params, w_pre, h_pre)


        # --- fbo_scene: color (float) + depth ---
        glBindTexture(GL_TEXTURE_2D, self.scene_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w_pre, h_pre, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w_pre, h_pre)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_scene)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.scene_tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise RuntimeError(f"fbo_scene incomplete, status: {status}")

        # --- fbo_post: color (float) only, no depth needed for a 2D resample ---
        glBindTexture(GL_TEXTURE_2D, self.post_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w_pre, h_pre, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_post)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.post_tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise RuntimeError(f"fbo_post incomplete, status: {status}")

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

        This is the standard *linear* MVP projection only - it produces the
        undistorted, pinhole gl_Position used for pass 1. Lens distortion is
        applied afterwards, as a separate non-linear fragment-shader pass.
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

    def _render_scene_pass(self, cam_params, w, h):
        """Pass 1: standard linear MVP render (no distortion) into fbo_scene."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_scene)
        glViewport(0, 0, w, h)
        glEnable(GL_DEPTH_TEST)
        glUseProgram(self.scene_program)
        glBindVertexArray(self.vao)

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

        glUniformMatrix4fv(glGetUniformLocation(self.scene_program, "u_projection"), 1, GL_FALSE, proj_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.scene_program, "u_view"), 1, GL_FALSE, view_matrix)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def _render_distortion_pass(self, cam_params, w, h):
        """Pass 2: fullscreen resample of fbo_scene into fbo_post using radial3k distortion."""
        focal_length = cam_params['focal_length']
        sensor_width = cam_params['sensor_width']
        sensor_height = cam_params['sensor_height']
        cx = cam_params['cx']
        cy = cam_params['cy']

        # Pixel-space focal lengths (allows non-square pixels if sensor
        # aspect ratio differs from the pixel aspect ratio).
        fx = w * focal_length / sensor_width
        fy = h * focal_length / sensor_height

        # Principal point in pixels, same offset-from-center convention as
        # the projection matrix above.
        cx_px = w / 2.0 + cx
        cy_px = h / 2.0 + cy

        k = np.asarray(cam_params.get('distortion_params', np.zeros(3)), dtype=np.float32).reshape(-1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_post)
        glViewport(0, 0, w, h)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.distort_program)
        glBindVertexArray(self.quad_vao)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_tex)
        glUniform1i(glGetUniformLocation(self.distort_program, "u_scene_tex"), 0)

        glUniform2f(glGetUniformLocation(self.distort_program, "u_resolution"), float(w), float(h))
        glUniform2f(glGetUniformLocation(self.distort_program, "u_focal"), fx, fy)
        glUniform2f(glGetUniformLocation(self.distort_program, "u_principal"), cx_px, cy_px)
        glUniform3f(glGetUniformLocation(self.distort_program, "u_k"), float(k[0]), float(k[1]), float(k[2]))

        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def render(self, cam_params):
        """Renders the loaded mesh from a specific camera configuration, with
        radial3k lens distortion applied as a post-process pass.

        Returns an (h, w, 4) float32 array of unclamped world-space
        positions (RGB channels) read back from a floating-point FBO.
        """
        glfw.make_context_current(self.window)

        # Resize both FBOs' attachments if this call uses a different resolution.
        w_final = cam_params.get('width', self.width)
        h_final = cam_params.get('height', self.height)
        self._allocate_fbo_storage(cam_params, w_final, h_final)

        w, h = self._fbo_new_size

        # Pass 1: standard linear MVP render (undistorted) -> fbo_scene.
        self._render_scene_pass(cam_params, w, h)

        # Pass 2: fullscreen radial3k distortion resample -> fbo_post.
        self._render_distortion_pass(cam_params, w, h)

        # Read back pixels from the final (distorted) FBO.
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_post)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixel_data = glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        pixel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((h, w, 4))
        pixel_array = np.flip(pixel_array, axis=0)

        # Trim to the original resolution
        pad_x, pad_y = (w - w_final) // 2, (h - h_final) // 2
        print(f"Trimming pixel array from {w}x{h} to {w_final}x{h_final} with padding ({pad_x}, {pad_y})")
        pixel_array = pixel_array[pad_y:pad_y+h_final, pad_x:pad_x+w_final]

        return pixel_array

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

    def __del__(self):
        # Best-effort cleanup of GL resources.
        try:
            if getattr(self, 'fbo_scene', None):
                glDeleteFramebuffers(1, [self.fbo_scene])
            if getattr(self, 'fbo_post', None):
                glDeleteFramebuffers(1, [self.fbo_post])
            if getattr(self, 'scene_tex', None):
                glDeleteTextures(1, [self.scene_tex])
            if getattr(self, 'post_tex', None):
                glDeleteTextures(1, [self.post_tex])
            if getattr(self, 'depth_rbo', None):
                glDeleteRenderbuffers(1, [self.depth_rbo])
        except Exception:
            pass