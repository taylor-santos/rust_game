layout (constant_id =  0) const bool HAS_NORMAL_VEC3                           = false;
layout (constant_id =  1) const bool HAS_TANGENT_VEC4                          = false;
layout (constant_id =  2) const bool HAS_TEXCOORD_0_VEC2                       = false;
layout (constant_id =  3) const bool HAS_TEXCOORD_1_VEC2                       = false;
layout (constant_id =  4) const bool HAS_COLOR_0_VEC3                          = false;
layout (constant_id =  5) const bool HAS_COLOR_0_VEC4                          = false;
layout (constant_id =  6) const bool NOT_TRIANGLE                              = false;
layout (constant_id =  7) const bool ALPHAMODE_OPAQUE                          = true;
layout (constant_id =  8) const bool ALPHAMODE_MASK                            = false;
layout (constant_id =  9) const bool ALPHAMODE_BLEND                           = false;
layout (constant_id = 10) const bool MATERIAL_UNLIT                            = false;
layout (constant_id = 11) const bool HAS_NORMAL_MAP                            = false;
layout (constant_id = 12) const bool HAS_NORMAL_UV_TRANSFORM                   = false;
layout (constant_id = 13) const bool HAS_VERT_NORMAL_UV_TRANSFORM              = false;
layout (constant_id = 14) const bool HAS_EMISSIVE_MAP                          = false;
layout (constant_id = 15) const bool HAS_EMISSIVE_UV_TRANSFORM                 = false;
layout (constant_id = 16) const bool HAS_OCCLUSION_MAP                         = false;
layout (constant_id = 17) const bool HAS_OCCLUSION_UV_TRANSFORM                = false;
layout (constant_id = 18) const bool HAS_BASE_COLOR_MAP                        = false;
layout (constant_id = 19) const bool HAS_BASECOLOR_UV_TRANSFORM                = false;
layout (constant_id = 20) const bool MATERIAL_METALLICROUGHNESS                = false;
layout (constant_id = 21) const bool HAS_METALLIC_ROUGHNESS_MAP                = false;
layout (constant_id = 22) const bool HAS_METALLICROUGHNESS_UV_TRANSFORM        = false;
layout (constant_id = 23) const bool MATERIAL_SHEEN                            = false;
layout (constant_id = 24) const bool HAS_SHEEN_COLOR_MAP                       = false;
layout (constant_id = 25) const bool HAS_SHEENCOLOR_UV_TRANSFORM               = false;
layout (constant_id = 26) const bool HAS_SHEEN_ROUGHNESS_MAP                   = false;
layout (constant_id = 27) const bool HAS_SHEENROUGHNESS_UV_TRANSFORM           = false;
layout (constant_id = 28) const bool MATERIAL_SPECULARGLOSSINESS               = false;
layout (constant_id = 29) const bool HAS_DIFFUSE_MAP                           = false;
layout (constant_id = 30) const bool HAS_DIFFUSE_UV_TRANSFORM                  = false;
layout (constant_id = 31) const bool HAS_SPECULAR_GLOSSINESS_MAP               = false;
layout (constant_id = 32) const bool HAS_SPECULARGLOSSINESS_UV_TRANSFORM       = false;
layout (constant_id = 33) const bool MATERIAL_CLEARCOAT                        = false;
layout (constant_id = 34) const bool HAS_CLEARCOAT_MAP                         = false;
layout (constant_id = 35) const bool HAS_CLEARCOAT_UV_TRANSFORM                = false;
layout (constant_id = 36) const bool HAS_CLEARCOAT_ROUGHNESS_MAP               = false;
layout (constant_id = 37) const bool HAS_CLEARCOATROUGHNESS_UV_TRANSFORM       = false;
layout (constant_id = 38) const bool HAS_CLEARCOAT_NORMAL_MAP                  = false;
layout (constant_id = 39) const bool HAS_CLEARCOATNORMAL_UV_TRANSFORM          = false;
layout (constant_id = 40) const bool MATERIAL_SPECULAR                         = false;
layout (constant_id = 41) const bool HAS_SPECULAR_MAP                          = false;
layout (constant_id = 42) const bool HAS_SPECULAR_UV_TRANSFORM                 = false;
layout (constant_id = 43) const bool HAS_SPECULAR_COLOR_MAP                    = false;
layout (constant_id = 44) const bool HAS_SPECULARCOLOR_UV_TRANSFORM            = false;
layout (constant_id = 45) const bool MATERIAL_TRANSMISSION                     = false;
layout (constant_id = 46) const bool HAS_TRANSMISSION_MAP                      = false;
layout (constant_id = 47) const bool HAS_TRANSMISSION_UV_TRANSFORM             = false;
layout (constant_id = 48) const bool MATERIAL_VOLUME                           = false;
layout (constant_id = 49) const bool HAS_THICKNESS_MAP                         = false;
layout (constant_id = 50) const bool HAS_THICKNESS_UV_TRANSFORM                = false;
layout (constant_id = 51) const bool MATERIAL_IRIDESCENCE                      = false;
layout (constant_id = 52) const bool HAS_IRIDESCENCE_MAP                       = false;
layout (constant_id = 53) const bool HAS_IRIDESCENCE_UV_TRANSFORM              = false;
layout (constant_id = 54) const bool HAS_IRIDESCENCE_THICKNESS_MAP             = false;
layout (constant_id = 55) const bool HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM     = false;
layout (constant_id = 56) const bool MATERIAL_DIFFUSE_TRANSMISSION             = false;
layout (constant_id = 57) const bool HAS_DIFFUSE_TRANSMISSION_MAP              = false;
layout (constant_id = 58) const bool HAS_DIFFUSETRANSMISSION_UV_TRANSFORM      = false;
layout (constant_id = 59) const bool HAS_DIFFUSE_TRANSMISSION_COLOR_MAP        = false;
layout (constant_id = 60) const bool HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM = false;
layout (constant_id = 61) const bool MATERIAL_ANISOTROPY                       = false;
layout (constant_id = 62) const bool HAS_ANISOTROPY_MAP                        = false;
layout (constant_id = 63) const bool HAS_ANISOTROPY_UV_TRANSFORM               = false;
layout (constant_id = 64) const bool MATERIAL_IOR                              = false;
layout (constant_id = 65) const bool MATERIAL_DISPERSION                       = false;
layout (constant_id = 66) const bool MATERIAL_EMISSIVE_STRENGTH                = false;


layout (push_constant) uniform Object {
    mat4 u_ModelMatrix;
    mat4 u_NormalMatrix;
} object;

layout (set = 3, binding = 0) uniform Material {
    // Metallic Roughness
    float u_MetallicFactor;
    float u_RoughnessFactor;
    vec4 u_BaseColorFactor;

    // Specular Glossiness
    vec3 u_SpecularFactor;
    vec4 u_DiffuseFactor;
    float u_GlossinessFactor;

    // Sheen
    float u_SheenRoughnessFactor;
    vec3 u_SheenColorFactor;

    // Clearcoat
    float u_ClearcoatFactor;
    float u_ClearcoatRoughnessFactor;

    // Specular
    vec3 u_KHR_materials_specular_specularColorFactor;
    float u_KHR_materials_specular_specularFactor;

    // Transmission
    float u_TransmissionFactor;

    // Volume
    float u_ThicknessFactor;
    vec3 u_AttenuationColor;
    float u_AttenuationDistance;

    // Iridescence
    float u_IridescenceFactor;
    float u_IridescenceIor;
    float u_IridescenceThicknessMinimum;
    float u_IridescenceThicknessMaximum;

    // Diffuse Transmission
    float u_DiffuseTransmissionFactor;
    vec3 u_DiffuseTransmissionColorFactor;

    // Emissive Strength
    float u_EmissiveStrength;

    // IOR
    float u_Ior;

    // Anisotropy
    vec3 u_Anisotropy;

    // Dispersion
    float u_Dispersion;

    // Alpha mode
    float u_AlphaCutoff;

    mat3 u_vertNormalUVTransform;
} material;

layout (set = 1, binding = 0) uniform Camera {
    mat4 u_ViewMatrix;
    mat4 u_ProjectionMatrix;
    mat4 u_ViewProjectionMatrix;
    vec3 u_Camera;
} camera;