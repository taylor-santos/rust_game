#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texcoord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_tangent;
layout(location = 2) out vec3 v_bitangent;
layout(location = 3) out vec2 v_texcoord;
layout(location = 4) out vec3 v_worldPos;

layout (push_constant) uniform PushConstant {
    vec3 light_direction;
    vec3 light_color;
    vec3 light_ambient;
    mat4 cam_viewproj;
    vec3 cam_position;
} push_constants;

layout (set = 1, binding = 0) uniform Object {
    mat4 model;
} object;

void main() {
    vec4 worldPos = object.model * vec4(position, 1.0);
    v_worldPos = worldPos.xyz;

    v_normal = mat3(object.model) * normal;
    v_tangent = mat3(object.model) * tangent;
    v_bitangent = mat3(object.model) * cross(normal, tangent);
    v_texcoord = texcoord;

    gl_Position = push_constants.cam_viewproj * worldPos;
}