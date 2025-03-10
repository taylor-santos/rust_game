#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 g_finalColor;

layout(set = 0, binding = 0) uniform sampler2D u_framebuffer;

#include <tonemapping.glsl>

void main() {
    vec3 color = texture(u_framebuffer, v_uv).rgb;
    const float exposure = 1.0;
    g_finalColor = vec4(toneMap(color, exposure), 1.0);
}