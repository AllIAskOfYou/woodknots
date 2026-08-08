import sys
import os
import numpy as np

# Set platform before importing OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'glfw'

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

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
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float u_sphere_radius;
out vec2 v_uv;
void main() {
    v_uv = a_quad_offset + 0.5;
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
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec2 coord = v_uv * 2.0 - 1.0;
    float dist = dot(coord, coord);
    if (dist > 1.0) { discard; }
    fragColor = vec4(1.0, 0.3, 0.3, 1.0);
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
# APPLICATION
# ==========================================
class App:
    def __init__(self, obj_path, points_array):
        self.obj_path = obj_path
        self.points = np.array(points_array, dtype=np.float32)
        self.cam_dist = 5.0
        self.cam_yaw = 45.0
        self.cam_pitch = -30.0
        self.cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.sphere_radius = 0.05
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

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

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            imgui.new_frame()
            
            # Process mouse rotation directly from ImGui IO state each frame
            self.handle_mouse_input()
            
            self.build_ui()

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

    # (Keep mouse_button_callback and mouse_callback methods or remove them since they are replaced by handle_mouse_input)

    def build_ui(self):
        imgui.begin("Control Panel", True)
        imgui.text(f"Loaded Points: {len(self.points)}")
        imgui.separator()
        
        # Sphere Radius Slider
        changed_rad, new_rad = imgui.slider_float("Sphere Radius", self.sphere_radius, 0.001, 0.05)
        if changed_rad:
            self.sphere_radius = new_rad

        imgui.separator()
        imgui.text("Camera Controls")
        
        # Camera Yaw Slider (-180 to 180 degrees)
        changed_yaw, new_yaw = imgui.slider_float("Yaw", self.cam_yaw, -180.0, 180.0)
        if changed_yaw:
            self.cam_yaw = new_yaw

        # Camera Pitch Slider (Clamped to avoid gimbal lock)
        changed_pitch, new_pitch = imgui.slider_float("Pitch", self.cam_pitch, -89.0, 89.0)
        if changed_pitch:
            self.cam_pitch = new_pitch

        # Camera Distance Slider
        changed_dist, new_dist = imgui.slider_float("Distance", self.cam_dist, 0.5, 50.0)
        if changed_dist:
            self.cam_dist = new_dist
            
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
        quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribDivisor(0, 1)

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
        glUseProgram(self.sphere_shader)
        glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.sphere_shader, "model"), 1, GL_FALSE, model)
        glUniform1f(glGetUniformLocation(self.sphere_shader, "u_sphere_radius"), self.sphere_radius)
        glBindVertexArray(self.sphere_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, len(self.points))

if __name__ == "__main__":
    obj_filename = sys.argv[1] if len(sys.argv) > 1 else "data/matching/textured_mesh/texturedMesh.obj"
    mask_filename = "data/matching/resized_1024_mask/2037.png"
    pixel_data_filename = "data/matching/resized_1024_3d/2037.npy"
    if not os.path.exists(mask_filename) or not os.path.exists(pixel_data_filename):
        print(f"Required files not found: {mask_filename} or {pixel_data_filename}")
        sys.exit(1)
    image_mask = np.array(Image.open(mask_filename).convert("L")) > 0
    pixel_data = np.load(pixel_data_filename)
    # subsample
    image_mask = image_mask[::4, ::4]
    pixel_data = pixel_data[::4, ::4, :]
    positions = pixel_data[..., :3]
    valid_mask = pixel_data[..., 3] > 0.5
    combined_mask = np.logical_and(image_mask, valid_mask)
    valid_positions = positions[combined_mask]
    print(f"Loaded with shape {valid_positions.shape}")
    unique_positions = np.unique(valid_positions, axis=0)
    
    print(f"Unique valid positions: {unique_positions.shape[0]}")

    # dummy_points = np.random.uniform(low=-1.5, high=1.5, size=(100, 3))
    if not os.path.exists(obj_filename):
        print(f"Usage: python app.py model.obj")
        sys.exit(1)
    App(obj_filename, unique_positions).run()