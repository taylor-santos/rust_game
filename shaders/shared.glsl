layout (push_constant) uniform Object {
    mat4 u_ModelMatrix;
    mat4 u_NormalMatrix;
} object;

layout (set = 1, binding = 0) uniform Material {
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

layout (set = 2, binding = 0) uniform Camera {
    mat4 u_ViewProjectionMatrix;
    vec3 u_Camera;
} camera;