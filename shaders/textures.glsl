// IBL

layout (set = 2, binding = 0) uniform samplerCube   u_LambertianEnvSampler;
layout (set = 2, binding = 1) uniform samplerCube   u_GGXEnvSampler;
layout (set = 2, binding = 2) uniform sampler2D     u_GGXLUT;
layout (set = 2, binding = 3) uniform samplerCube   u_CharlieEnvSampler;
layout (set = 2, binding = 4) uniform sampler2D     u_CharlieLUT;
layout (set = 2, binding = 5) uniform sampler2D     u_SheenELUT;


// General Material

layout (set = 3, binding = 1) uniform MatSamplers {
    float       u_NormalScale;
    int         u_NormalUVSet;
    mat3        u_NormalUVTransform;

    vec3        u_EmissiveFactor;
    int         u_EmissiveUVSet;
    mat3        u_EmissiveUVTransform;

    int         u_OcclusionUVSet;
    float       u_OcclusionStrength;
    mat3        u_OcclusionUVTransform;

    int         u_BaseColorUVSet;
    mat3        u_BaseColorUVTransform;

    int         u_MetallicRoughnessUVSet;
    mat3        u_MetallicRoughnessUVTransform;

    int         u_SheenColorUVSet;
    mat3        u_SheenColorUVTransform;
    int         u_SheenRoughnessUVSet;
    mat3        u_SheenRoughnessUVTransform;

    int         u_DiffuseUVSet;
    mat3        u_DiffuseUVTransform;

    int         u_SpecularGlossinessUVSet;
    mat3        u_SpecularGlossinessUVTransform;

    int         u_ClearcoatUVSet;
    mat3        u_ClearcoatUVTransform;
    int         u_ClearcoatRoughnessUVSet;
    mat3        u_ClearcoatRoughnessUVTransform;
    int         u_ClearcoatNormalUVSet;
    mat3        u_ClearcoatNormalUVTransform;
    float       u_ClearcoatNormalScale;

    int         u_SpecularUVSet;
    mat3        u_SpecularUVTransform;
    int         u_SpecularColorUVSet;
    mat3        u_SpecularColorUVTransform;

    int         u_TransmissionUVSet;
    mat3        u_TransmissionUVTransform;
    ivec2       u_TransmissionFramebufferSize;

    int         u_ThicknessUVSet;
    mat3        u_ThicknessUVTransform;

    int          u_IridescenceUVSet;
    mat3         u_IridescenceUVTransform;
    int          u_IridescenceThicknessUVSet;
    mat3         u_IridescenceThicknessUVTransform;

    int         u_DiffuseTransmissionUVSet;
    mat3        u_DiffuseTransmissionUVTransform;
    int         u_DiffuseTransmissionColorUVSet;
    mat3        u_DiffuseTransmissionColorUVTransform;

    int         u_AnisotropyUVSet;
    mat3        u_AnisotropyUVTransform;
} s;

layout (set = 4, binding = 0) uniform sampler2D u_NormalSampler;
layout (set = 4, binding = 1) uniform sampler2D u_EmissiveSampler;
layout (set = 4, binding = 2) uniform sampler2D u_OcclusionSampler;


layout (location = 1) in vec2 v_texcoord_0;
layout (location = 2) in vec2 v_texcoord_1;


vec2 getNormalUV()
{
    vec3 uv = vec3(s.u_NormalUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_NORMAL_UV_TRANSFORM
    uv = s.u_NormalUVTransform * uv;
#endif

    return uv.xy;
}


vec2 getEmissiveUV()
{
    vec3 uv = vec3(s.u_EmissiveUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv = s.u_EmissiveUVTransform * uv;
#endif

    return uv.xy;
}


vec2 getOcclusionUV()
{
    vec3 uv = vec3(s.u_OcclusionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_OCCLUSION_UV_TRANSFORM
    uv = s.u_OcclusionUVTransform * uv;
#endif

    return uv.xy;
}


// Metallic Roughness Material

layout (set = 4, binding = 3) uniform sampler2D u_BaseColorSampler;
layout (set = 4, binding = 4) uniform sampler2D u_MetallicRoughnessSampler;

vec2 getBaseColorUV()
{
    vec3 uv = vec3(s.u_BaseColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv = s.u_BaseColorUVTransform * uv;
#endif

    return uv.xy;
}

vec2 getMetallicRoughnessUV()
{
    vec3 uv = vec3(s.u_MetallicRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv = s.u_MetallicRoughnessUVTransform * uv;
#endif

    return uv.xy;
}

// Specular Glossiness Material

layout (set = 4, binding = 5) uniform sampler2D u_DiffuseSampler;
layout (set = 4, binding = 6) uniform sampler2D u_SpecularGlossinessSampler;

vec2 getSpecularGlossinessUV()
{
    vec3 uv = vec3(s.u_SpecularGlossinessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_SPECULARGLOSSINESS_UV_TRANSFORM
    uv = s.u_SpecularGlossinessUVTransform * uv;
#endif

    return uv.xy;
}

vec2 getDiffuseUV()
{
    vec3 uv = vec3(s.u_DiffuseUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);

#ifdef HAS_DIFFUSE_UV_TRANSFORM
    uv = s.u_DiffuseUVTransform * uv;
#endif

    return uv.xy;
}


// Clearcoat Material

layout (set = 4, binding = 7) uniform sampler2D u_ClearcoatSampler;
layout (set = 4, binding = 8) uniform sampler2D u_ClearcoatRoughnessSampler;
layout (set = 4, binding = 9) uniform sampler2D u_ClearcoatNormalSampler;

vec2 getClearcoatUV()
{
    vec3 uv = vec3(s.u_ClearcoatUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOAT_UV_TRANSFORM
    uv = s.u_ClearcoatUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getClearcoatRoughnessUV()
{
    vec3 uv = vec3(s.u_ClearcoatRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOATROUGHNESS_UV_TRANSFORM
    uv = s.u_ClearcoatRoughnessUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getClearcoatNormalUV()
{
    vec3 uv = vec3(s.u_ClearcoatNormalUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_CLEARCOATNORMAL_UV_TRANSFORM
    uv = s.u_ClearcoatNormalUVTransform * uv;
#endif
    return uv.xy;
}

// Sheen Material

layout (set = 4, binding = 10) uniform sampler2D u_SheenColorSampler;
layout (set = 4, binding = 11) uniform sampler2D u_SheenRoughnessSampler;


vec2 getSheenColorUV()
{
    vec3 uv = vec3(s.u_SheenColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SHEENCOLOR_UV_TRANSFORM
    uv = s.u_SheenColorUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getSheenRoughnessUV()
{
    vec3 uv = vec3(s.u_SheenRoughnessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SHEENROUGHNESS_UV_TRANSFORM
    uv = s.u_SheenRoughnessUVTransform * uv;
#endif
    return uv.xy;
}


// Specular Material

layout (set = 4, binding = 12) uniform sampler2D u_SpecularSampler;
layout (set = 4, binding = 13) uniform sampler2D u_SpecularColorSampler;

vec2 getSpecularUV()
{
    vec3 uv = vec3(s.u_SpecularUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SPECULAR_UV_TRANSFORM
    uv = s.u_SpecularUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getSpecularColorUV()
{
    vec3 uv = vec3(s.u_SpecularColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_SPECULARCOLOR_UV_TRANSFORM
    uv = s.u_SpecularColorUVTransform * uv;
#endif
    return uv.xy;
}


// Transmission Material

layout (set = 4, binding = 14) uniform sampler2D u_TransmissionSampler;
layout (set = 4, binding = 15) uniform sampler2D u_TransmissionFramebufferSampler;

vec2 getTransmissionUV()
{
    vec3 uv = vec3(s.u_TransmissionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_TRANSMISSION_UV_TRANSFORM
    uv = s.u_TransmissionUVTransform * uv;
#endif
    return uv.xy;
}


// Volume Material

layout (set = 4, binding = 16) uniform sampler2D u_ThicknessSampler;

vec2 getThicknessUV()
{
    vec3 uv = vec3(s.u_ThicknessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_THICKNESS_UV_TRANSFORM
    uv = s.u_ThicknessUVTransform * uv;
#endif
    return uv.xy;
}


// Iridescence

layout (set = 4, binding = 17) uniform sampler2D u_IridescenceSampler;
layout (set = 4, binding = 18) uniform sampler2D u_IridescenceThicknessSampler;

vec2 getIridescenceUV()
{
    vec3 uv = vec3(s.u_IridescenceUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_IRIDESCENCE_UV_TRANSFORM
    uv = s.u_IridescenceUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getIridescenceThicknessUV()
{
    vec3 uv = vec3(s.u_IridescenceThicknessUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM
    uv = s.u_IridescenceThicknessUVTransform * uv;
#endif
    return uv.xy;
}


// Diffuse Transmission

layout (set = 4, binding = 19) uniform sampler2D u_DiffuseTransmissionSampler;
layout (set = 4, binding = 20) uniform sampler2D u_DiffuseTransmissionColorSampler;

vec2 getDiffuseTransmissionUV()
{
    vec3 uv = vec3(s.u_DiffuseTransmissionUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_DIFFUSETRANSMISSION_UV_TRANSFORM
    uv = s.u_DiffuseTransmissionUVTransform * uv;
#endif
    return uv.xy;
}

vec2 getDiffuseTransmissionColorUV()
{
    vec3 uv = vec3(s.u_DiffuseTransmissionColorUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM
    uv = s.u_DiffuseTransmissionColorUVTransform * uv;
#endif
    return uv.xy;
}

// Anisotropy

layout (set = 4, binding = 21) uniform sampler2D u_AnisotropySampler;

vec2 getAnisotropyUV()
{
    vec3 uv = vec3(s.u_AnisotropyUVSet < 1 ? v_texcoord_0 : v_texcoord_1, 1.0);
#ifdef HAS_ANISOTROPY_UV_TRANSFORM
    uv = s.u_AnisotropyUVTransform * uv;
#endif
    return uv.xy;
}

