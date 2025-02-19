layout (constant_id = 0)  const bool HAS_NORMAL_VEC3                = true;
layout (constant_id = 1)  const bool HAS_TANGENT_VEC4               = false;
layout (constant_id = 2)  const bool MATERIAL_METALLICROUGHNESS     = true;
layout (constant_id = 3)  const bool MATERIAL_SPECULARGLOSSINESS    = false;
layout (constant_id = 4)  const bool MATERIAL_CLEARCOAT             = false;
layout (constant_id = 5)  const bool MATERIAL_SHEEN                 = false;
layout (constant_id = 6)  const bool MATERIAL_SPECULAR              = false;
layout (constant_id = 7)  const bool MATERIAL_TRANSMISSION          = false;
layout (constant_id = 8)  const bool MATERIAL_VOLUME                = false;
layout (constant_id = 9)  const bool MATERIAL_IRIDESCENCE           = false;
layout (constant_id = 10) const bool MATERIAL_DIFFUSE_TRANSMISSION  = false;
layout (constant_id = 11) const bool MATERIAL_ANISOTROPY            = false;
layout (constant_id = 12) const bool MATERIAL_IOR                   = false;
layout (constant_id = 13) const bool MATERIAL_DISPERSION            = false;
layout (constant_id = 14) const bool MATERIAL_EMISSIVE_STRENGTH     = false;
layout (constant_id = 15) const bool MATERIAL_UNLIT                 = false;

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

#ifdef HAS_VERT_NORMAL_UV_TRANSFORM
    mat3 u_vertNormalUVTransform;
#endif
} material;

layout (set = 1, binding = 0) uniform Camera {
    mat4 u_ViewMatrix;
    mat4 u_ProjectionMatrix;
    mat4 u_ViewProjectionMatrix;
    vec3 u_Camera;
} camera;