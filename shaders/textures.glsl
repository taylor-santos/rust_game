// IBL

layout (set = 0, binding = 1) uniform samplerCube   u_LambertianEnvSampler;
layout (set = 0, binding = 2) uniform samplerCube   u_GGXEnvSampler;
layout (set = 0, binding = 3) uniform sampler2D     u_GGXLUT;
layout (set = 0, binding = 4) uniform samplerCube   u_CharlieEnvSampler;
layout (set = 0, binding = 5) uniform sampler2D     u_CharlieLUT;
layout (set = 0, binding = 6) uniform sampler2D     u_SheenELUT;


// General Material

layout (set = 1, binding = 1) uniform MatSamplers {
    float       u_NormalScale;
    int         u_NormalUVSet;
    mat3        u_NormalUVTransform;
    vec3        u_EmissiveFactor;
    int         u_EmissiveUVSet;
    mat3        u_EmissiveUVTransform;
    int         u_OcclusionUVSet;
    float       u_OcclusionStrength;
    mat3        u_OcclusionUVTransform;
#ifdef MATERIAL_METALLICROUGHNESS
    int         u_BaseColorUVSet;
    mat3        u_BaseColorUVTransform;
    int         u_MetallicRoughnessUVSet;
    mat3        u_MetallicRoughnessUVTransform;
    int         u_SheenColorUVSet;
    mat3        u_SheenColorUVTransform;
    int         u_SheenRoughnessUVSet;
    mat3        u_SheenRoughnessUVTransform;
#endif // MATERIAL_METALLICROUGHNESS
} mat_samplers;

layout (set = 1, binding = 2) uniform sampler2D u_NormalSampler;
layout (set = 1, binding = 3) uniform sampler2D u_EmissiveSampler;
layout (set = 1, binding = 4) uniform sampler2D u_OcclusionSampler;


layout (location = 1) in vec2 v_texcoord_0;
layout (location = 2) in vec2 v_texcoord_1;


vec2 getNormalUV()
{
    vec3 uv = vec3(mat_samplers.u_NormalUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_NORMAL_UV_TRANSFORM
    uv = mat_samplers.u_NormalUVTransform * uv;
#endif

    return uv.xy;
}


vec2 getEmissiveUV()
{
    vec3 uv = vec3(mat_samplers.u_EmissiveUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv = mat_samplers.u_EmissiveUVTransform * uv;
#endif

    return uv.xy;
}


vec2 getOcclusionUV()
{
    vec3 uv = vec3(mat_samplers.u_OcclusionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_OCCLUSION_UV_TRANSFORM
    uv = mat_samplers.u_OcclusionUVTransform * uv;
#endif

    return uv.xy;
}


// Metallic Roughness Material


#ifdef MATERIAL_METALLICROUGHNESS

layout (set = 1, binding = 5) uniform sampler2D u_BaseColorSampler;
layout (set = 1, binding = 6) uniform sampler2D u_MetallicRoughnessSampler;

vec2 getBaseColorUV()
{
    vec3 uv = vec3(mat_samplers.u_BaseColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv = mat_samplers.u_BaseColorUVTransform * uv;
#endif

    return uv.xy;
}

vec2 getMetallicRoughnessUV()
{
    vec3 uv = vec3(mat_samplers.u_MetallicRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv = mat_samplers.u_MetallicRoughnessUVTransform * uv;
#endif

    return uv.xy;
}

#endif


// Specular Glossiness Material


#ifdef MATERIAL_SPECULARGLOSSINESS

uniform sampler2D u_DiffuseSampler;
uniform int u_DiffuseUVSet;
uniform mat3 u_DiffuseUVTransform;

uniform sampler2D u_SpecularGlossinessSampler;
uniform int u_SpecularGlossinessUVSet;
uniform mat3 u_SpecularGlossinessUVTransform;


vec2 getSpecularGlossinessUV()
{
    vec3 uv = vec3(u_SpecularGlossinessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_SPECULARGLOSSINESS_UV_TRANSFORM
    uv = u_SpecularGlossinessUVTransform * uv;
#endif

    return uv.xy;
}

vec2 getDiffuseUV()
{
    vec3 uv = vec3(u_DiffuseUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_DIFFUSE_UV_TRANSFORM
    uv = u_DiffuseUVTransform * uv;
#endif

    return uv.xy;
}

#endif


// Clearcoat Material


#ifdef MATERIAL_CLEARCOAT

uniform sampler2D u_ClearcoatSampler;
uniform int u_ClearcoatUVSet;
uniform mat3 u_ClearcoatUVTransform;

uniform sampler2D u_ClearcoatRoughnessSampler;
uniform int u_ClearcoatRoughnessUVSet;
uniform mat3 u_ClearcoatRoughnessUVTransform;

uniform sampler2D u_ClearcoatNormalSampler;
uniform int u_ClearcoatNormalUVSet;
uniform mat3 u_ClearcoatNormalUVTransform;
uniform float u_ClearcoatNormalScale;


vec2 getClearcoatUV()
{
    vec3 uv = vec3(u_ClearcoatUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOAT_UV_TRANSFORM
    uv = u_ClearcoatUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getClearcoatRoughnessUV()
{
    vec3 uv = vec3(u_ClearcoatRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOATROUGHNESS_UV_TRANSFORM
    uv = u_ClearcoatRoughnessUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getClearcoatNormalUV()
{
    vec3 uv = vec3(u_ClearcoatNormalUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOATNORMAL_UV_TRANSFORM
    uv = u_ClearcoatNormalUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Sheen Material


#ifdef MATERIAL_SHEEN

layout (set = 1, binding = 7) uniform sampler2D u_SheenColorSampler;
layout (set = 1, binding = 8) uniform sampler2D u_SheenRoughnessSampler;


vec2 getSheenColorUV()
{
    vec3 uv = vec3(mat_samplers.u_SheenColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SHEENCOLOR_UV_TRANSFORM
    uv = u_SheenColorUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getSheenRoughnessUV()
{
    vec3 uv = vec3(mat_samplers.u_SheenRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SHEENROUGHNESS_UV_TRANSFORM
    uv = u_SheenRoughnessUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Specular Material


#ifdef MATERIAL_SPECULAR

uniform sampler2D u_SpecularSampler;
uniform int u_SpecularUVSet;
uniform mat3 u_SpecularUVTransform;
uniform sampler2D u_SpecularColorSampler;
uniform int u_SpecularColorUVSet;
uniform mat3 u_SpecularColorUVTransform;


vec2 getSpecularUV()
{
    vec3 uv = vec3(u_SpecularUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SPECULAR_UV_TRANSFORM
    uv = u_SpecularUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getSpecularColorUV()
{
    vec3 uv = vec3(u_SpecularColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SPECULARCOLOR_UV_TRANSFORM
    uv = u_SpecularColorUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Transmission Material


#ifdef MATERIAL_TRANSMISSION

uniform sampler2D u_TransmissionSampler;
uniform int u_TransmissionUVSet;
uniform mat3 u_TransmissionUVTransform;
uniform sampler2D u_TransmissionFramebufferSampler;
uniform ivec2 u_TransmissionFramebufferSize;


vec2 getTransmissionUV()
{
    vec3 uv = vec3(u_TransmissionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_TRANSMISSION_UV_TRANSFORM
    uv = u_TransmissionUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Volume Material


#ifdef MATERIAL_VOLUME

uniform sampler2D u_ThicknessSampler;
uniform int u_ThicknessUVSet;
uniform mat3 u_ThicknessUVTransform;


vec2 getThicknessUV()
{
    vec3 uv = vec3(u_ThicknessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_THICKNESS_UV_TRANSFORM
    uv = u_ThicknessUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Iridescence


#ifdef MATERIAL_IRIDESCENCE

uniform sampler2D u_IridescenceSampler;
uniform int u_IridescenceUVSet;
uniform mat3 u_IridescenceUVTransform;

uniform sampler2D u_IridescenceThicknessSampler;
uniform int u_IridescenceThicknessUVSet;
uniform mat3 u_IridescenceThicknessUVTransform;


vec2 getIridescenceUV()
{
    vec3 uv = vec3(u_IridescenceUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_IRIDESCENCE_UV_TRANSFORM
    uv = u_IridescenceUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getIridescenceThicknessUV()
{
    vec3 uv = vec3(u_IridescenceThicknessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM
    uv = u_IridescenceThicknessUVTransform * uv;
#endif
    return uv.xy;
}

#endif


// Diffuse Transmission

#ifdef MATERIAL_DIFFUSE_TRANSMISSION

uniform sampler2D u_DiffuseTransmissionSampler;
uniform int u_DiffuseTransmissionUVSet;
uniform mat3 u_DiffuseTransmissionUVTransform;

uniform sampler2D u_DiffuseTransmissionColorSampler;
uniform int u_DiffuseTransmissionColorUVSet;
uniform mat3 u_DiffuseTransmissionColorUVTransform;


vec2 getDiffuseTransmissionUV()
{
    vec3 uv = vec3(u_DiffuseTransmissionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_DIFFUSETRANSMISSION_UV_TRANSFORM
    uv = u_DiffuseTransmissionUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getDiffuseTransmissionColorUV()
{
    vec3 uv = vec3(u_DiffuseTransmissionColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM
    uv = u_DiffuseTransmissionColorUVTransform * uv;
#endif
    return uv.xy;
}

#endif

// Anisotropy

#ifdef MATERIAL_ANISOTROPY

uniform sampler2D u_AnisotropySampler;
uniform int u_AnisotropyUVSet;
uniform mat3 u_AnisotropyUVTransform;

vec2 getAnisotropyUV()
{
    vec3 uv = vec3(u_AnisotropyUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_ANISOTROPY_UV_TRANSFORM
    uv = u_AnisotropyUVTransform * uv;
#endif
    return uv.xy;
}

#endif
