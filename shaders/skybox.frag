#version 450

layout (location = 0) in vec3 v_TexCoords;
layout(location = 0) out vec4 f_color;

layout (set = 0, binding = 0) uniform samplerCube u_GGXEnvSampler;

void main() {
    vec4 tex = texture(u_GGXEnvSampler, v_TexCoords);
    f_color = vec4(tex.rgb, 1.0);
}