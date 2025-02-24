#version 450


layout (location = 0) in vec3 v_TexCoords;
layout (location = 0) out vec4 FragColor;

layout (set = 0, binding = 0) uniform Constants {
    int     u_MipCount;
    int     u_FramebufferMipCount;
    mat3    u_EnvRotation;
    float   u_EnvIntensity;
    float   u_Exposure;
    float   u_Time;
} c;

layout (set = 2, binding = 1) uniform samplerCube   u_GGXEnvSampler;

#include <tonemapping.glsl>

void main()
{
    vec4 color = texture(u_GGXEnvSampler, v_TexCoords);
    color.rgb *= c.u_EnvIntensity;
    color.a = 1.0;

#ifdef LINEAR_OUTPUT
    FragColor = color.rgba;
#else
    FragColor = vec4(toneMap(color.rgb), color.a);
#endif
}