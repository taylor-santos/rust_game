#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    vec2 uv = vec2(v_texcoord.x, v_texcoord.y);
    vec4 albedo = texture(tex, uv);
    if (albedo.a < 0.01) discard;

    float brightness = dot(normalize(v_normal), normalize(LIGHT));

    f_color = vec4(albedo.rgb * brightness, 1.0);
}