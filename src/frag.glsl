#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D u_BaseColorTex;
layout(set = 1, binding = 1) uniform sampler2D u_NormalTex;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    vec2 uv = v_texcoord.xy;
    vec4 albedo = texture(u_BaseColorTex, uv);
    if (albedo.a < 0.01) discard;

    vec3 nTex = texture(u_NormalTex, v_texcoord).rgb;

    float brightness = dot(normalize(v_normal), normalize(LIGHT));

    f_color = vec4(albedo.rgb * brightness, 1.0);
}