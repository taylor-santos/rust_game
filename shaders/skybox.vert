#version 450

layout(location = 0) in vec3 a_position;
layout(location = 0) out vec3 v_TexCoords;

layout (push_constant) uniform Camera {
    mat4 u_ViewProjectionMatrix;
} camera;


void main() {
    v_TexCoords = a_position;
    mat4 mat = camera.u_ViewProjectionMatrix;
    mat[3] = vec4(0, 0, 0, 0.1);
    vec4 pos = mat * vec4(a_position, 1.0);
    gl_Position = pos.xyww;
}