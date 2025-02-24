#version 450

layout (set = 0, binding = 0) uniform Constants {
    int     u_MipCount;
    int     u_FramebufferMipCount;
    mat3    u_EnvRotation;
    float   u_EnvIntensity;
    float   u_Exposure;
    float   u_Time;
} c;

layout (set = 1, binding = 0) uniform Camera {
    mat4 u_ViewMatrix;
    mat4 u_ProjectionMatrix;
    mat4 u_ViewProjectionMatrix;
    vec3 u_Camera;
} camera;

layout (location = 0) in vec3 a_position;
layout (location = 0) out vec3 v_TexCoords;

void main()
{
    v_TexCoords = c.u_EnvRotation * a_position;
    mat4 mat = camera.u_ViewProjectionMatrix;
    mat[3] = vec4(0, 0, 0, 0.1);
    vec4 pos = mat * vec4(a_position, 1.0);
    gl_Position = pos.xyww;
}