#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texcoord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_tangent;
layout(location = 2) out vec3 v_bitangent;
layout(location = 3) out vec2 v_texcoord;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 viewproj;
} mvp;

void main() {
    vec3 bitangent = cross(normal, tangent);

    v_normal = mat3(mvp.model) * normal;
    v_tangent = mat3(mvp.model) * tangent;
    v_bitangent = mat3(mvp.model) * bitangent;
    v_texcoord = texcoord;

    gl_Position = mvp.viewproj * mvp.model * vec4(position, 1.0);
}