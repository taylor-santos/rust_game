#version 450

#include "shared.glsl"

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec4 a_tangent;
layout (location = 3) in vec2 a_texcoord_0;
layout (location = 4) in vec2 a_texcoord_1;
layout (location = 5) in vec4 a_color_0;

layout (location = 0) out vec3 v_Position;
layout (location = 1) out vec2 v_texcoord_0;
layout (location = 2) out vec2 v_texcoord_1;
layout (location = 3) out vec4 v_Color;
layout (location = 4) out vec3 v_Normal;
layout (location = 5) out mat3 v_TBN;

#ifdef USE_INSTANCING
in mat4 a_instance_model_matrix;
#endif

vec4 getPosition()
{
    vec4 pos = vec4(a_position, 1.0);

#ifdef USE_MORPHING
    pos += getTargetPosition(gl_VertexID);
#endif

#ifdef USE_SKINNING
    pos = getSkinningMatrix() * pos;
#endif

    return pos;
}


vec3 getNormal()
{
    vec3 normal = a_normal;

#ifdef USE_MORPHING
    normal += getTargetNormal(gl_VertexID);
#endif

#ifdef USE_SKINNING
    normal = mat3(getSkinningNormalMatrix()) * normal;
#endif

    return normalize(normal);
}

vec3 getTangent()
{
    vec3 tangent = a_tangent.xyz;

#ifdef USE_MORPHING
    tangent += getTargetTangent(gl_VertexID);
#endif

#ifdef USE_SKINNING
    tangent = mat3(getSkinningMatrix()) * tangent;
#endif

    return normalize(tangent);
}


void main()
{
    gl_PointSize = 1.0f;
#ifdef USE_INSTANCING
    mat4 modelMatrix = a_instance_model_matrix;
    mat4 normalMatrix = transpose(inverse(modelMatrix));
#else
    mat4 modelMatrix = object.u_ModelMatrix;
    mat4 normalMatrix = object.u_NormalMatrix;
#endif
    vec4 pos = modelMatrix * getPosition();
    v_Position = vec3(pos.xyz) / pos.w;

    if (HAS_NORMAL_VEC3) {
        if (HAS_TANGENT_VEC4) {
            vec3 tangent = getTangent();
            vec3 normalW = normalize(vec3(normalMatrix * vec4(getNormal(), 0.0)));
            vec3 tangentW = vec3(modelMatrix * vec4(tangent, 0.0));
            vec3 bitangentW = cross(normalW, tangentW) * a_tangent.w;

            if (HAS_VERT_NORMAL_UV_TRANSFORM) {
                tangentW = material.u_vertNormalUVTransform * tangentW;
                bitangentW = material.u_vertNormalUVTransform * bitangentW;
            }

            bitangentW = normalize(bitangentW);
            tangentW = normalize(tangentW);

            v_TBN = mat3(tangentW, bitangentW, normalW);
        } else {
            v_Normal = normalize(vec3(normalMatrix * vec4(getNormal(), 0.0)));
        }
    }

    if (HAS_TEXCOORD_0_VEC2) {
        v_texcoord_0 = a_texcoord_0;
    } else {
        v_texcoord_0 = vec2(0.0, 0.0);
    }

    if (HAS_TEXCOORD_1_VEC2) {
        v_texcoord_1 = a_texcoord_1;
    } else {
        v_texcoord_1 = vec2(0.0, 0.0);
    }

#ifdef USE_MORPHING
    v_texcoord_0 += getTargetTexCoord0(gl_VertexID);
    v_texcoord_1 += getTargetTexCoord1(gl_VertexID);
#endif


    if (HAS_COLOR_0_VEC3) {
        v_Color = a_color_0;
#if defined(USE_MORPHING)
        v_Color = clamp(v_Color + getTargetColor0(gl_VertexID).xyz, 0.0f, 1.0f);
#endif
    }

    if (HAS_COLOR_0_VEC4) {
        v_Color = a_color_0;
#if defined(USE_MORPHING)
        v_Color = clamp(v_Color + getTargetColor0(gl_VertexID), 0.0f, 1.0f);
#endif
    }

    gl_Position = camera.u_ViewProjectionMatrix * pos;
}