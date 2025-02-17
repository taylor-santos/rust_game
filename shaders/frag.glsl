#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_tangent;
layout(location = 2) in vec3 v_bitangent;
layout(location = 3) in vec2 v_texcoord;
layout(location = 4) in vec3 v_worldPos;

layout(location = 0) out vec4 f_color;

layout (push_constant) uniform PushConstants {
    vec3 light_direction;
    vec3 light_color;
    vec3 light_ambient;
    mat4 cam_viewproj;
    vec3 cam_position;
} push_constants;

#define ALPHA_BLEND  0
#define ALPHA_OPAQUE 1
#define ALPHA_MASK   2

layout (set = 0, binding = 0) uniform Material {
    float alphaCutoff;
    int alphaMode;
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    vec3 emissiveFactor;
    float emissiveStrength;
} material;

layout (set = 0, binding = 1) uniform sampler2D u_BaseColorTex;
layout (set = 0, binding = 2) uniform sampler2D u_NormalTex;
layout (set = 0, binding = 3) uniform sampler2D u_MetallicRoughnessTex;
layout (set = 0, binding = 4) uniform sampler2D u_EmissiveTex;

// -----------------------------------------------------------------------------
// Constants and helper functions for PBR
// -----------------------------------------------------------------------------
const float PI = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function
float DistributionGGX(float NdotH, float roughness)
{
    float alpha     = roughness * roughness;
    float alphaSqr  = alpha * alpha;
    float denom     = (NdotH * NdotH) * (alphaSqr - 1.0) + 1.0;
    return alphaSqr / (PI * denom * denom);
}

// Smith's Schlick-GGX geometry term
float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0; // Schlick-GGX approximation

    float G_V = NdotV / (NdotV * (1.0 - k) + k);
    float G_L = NdotL / (NdotL * (1.0 - k) + k);
    return G_V * G_L;
}

// Schlick Fresnel approximation
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    // FresnelSchlick = F0 + (1 - F0) * (1 - cosTheta)^5
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float hash12(vec2 p)
{
    // Some constants that produce decent distribution
    float h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

const float bayer[64] = float[64](
     0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
    48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
    12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, 38.0/64.0,
    60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
     3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
    51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
    15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, 37.0/64.0,
    63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
);

void ditherDiscard(float alpha)
{
    int x = int(mod(gl_FragCoord.x, 8));
    int y = int(mod(gl_FragCoord.y, 8));

    float threshold = bayer[y * 8 + x];
    float rnd = hash12(gl_FragCoord.xy) * (1.0 / 64.0);
    threshold = clamp(threshold + rnd, 0, 1);

    if (alpha <= threshold) {
        discard;
    }
}

void main()
{
    // -------------------------------------------------------------------------
    // 1. Sample material properties from textures & factor uniforms
    // -------------------------------------------------------------------------
    // Base color
    vec4 baseColorTex = texture(u_BaseColorTex, v_texcoord);
    if (material.alphaMode == ALPHA_MASK && baseColorTex.a < material.alphaCutoff) discard;
    if (material.alphaMode == ALPHA_BLEND) {
        ditherDiscard(baseColorTex.a);
    }

    // Multiply texture’s RGB by material’s baseColorFactor RGB (and also handle alpha if needed)
    vec3 baseColor = baseColorTex.rgb * material.baseColorFactor.rgb;
    // Optionally modulate alpha, if you have transparency:
    // float alpha = baseColorTex.a * material.baseColorFactor.a;

    // -------------------------------------------------------------------------
    // 2. Derive the normal from normal map (in tangent space)
    // -------------------------------------------------------------------------
    // The normal map is assumed to be in [0..1] range, so remap to [-1..1].
    vec3 normalSample = texture(u_NormalTex, v_texcoord).xyz * 2.0 - 1.0;

    // Construct TBN matrix from our world-space T, B, N
    mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
    // Transform the tangent-space normal to world space
    vec3 N = normalize(TBN * normalSample);

    // Metallic (B channel) and roughness (G channel)
    vec4 mrTex      = texture(u_MetallicRoughnessTex, v_texcoord);
    float metallic = mrTex.b * material.metallicFactor;
    float roughness = mrTex.g * material.roughnessFactor;

    // Emissive
    vec3 emissiveTex = texture(u_EmissiveTex, v_texcoord).rgb;
    vec3 emissive = emissiveTex * material.emissiveFactor * material.emissiveStrength;

    // -------------------------------------------------------------------------
    // 3. Compute the lighting terms
    // -------------------------------------------------------------------------
    // Camera/view direction (normalize worldPos -> camera vector)
    vec3 V = normalize(push_constants.cam_position - v_worldPos);

    // Light direction is given as 'light.direction' is "toward the fragment";
    // if it were "from the light," you might do `-light.direction`.
    vec3 L = normalize(push_constants.light_direction);

    // Half vector
    vec3 H = normalize(V + L);

    // Dot products
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // -------------------------------------------------------------------------
    // 4. Microfacet BRDF calculations
    // -------------------------------------------------------------------------
    // For metals, F0 is baseColor; for dielectrics, ~0.04
    // We blend between these using metallic factor.
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, baseColor, metallic);

    // Distribution term (D)
    float D = DistributionGGX(NdotH, roughness);

    // Geometry term (G)
    float G = GeometrySmith(NdotV, NdotL, roughness);

    // Fresnel term (F)
    vec3 F = FresnelSchlick(HdotV, F0);

    // Specular term
    vec3 numerator   = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular    = numerator / denominator;

    // kD is the diffuse contribution factor (energy conserved) = (1 - F) * (1 - metallic)
    // Metallic surfaces do not have a “traditional” diffuse term (they are highly reflective).
    vec3 kD = (1.0 - F) * (1.0 - metallic);

    // Lambertian diffuse = baseColor / π
    vec3 diffuse = kD * baseColor / PI;

    // Final contribution from this directional light
    vec3 directLight = (diffuse + specular) * push_constants.light_color * NdotL;

    // -------------------------------------------------------------------------
    // 5. Add ambient (simple constant ambient in this example) and emissive
    // -------------------------------------------------------------------------
    // Ambient is approximated as a simple constant color. In more advanced setups,
    // you would use image-based lighting (IBL) for more realism.
    vec3 ambient = push_constants.light_ambient * baseColor;

    // Combine all terms
    vec3 color = ambient + directLight + emissive;

    // -------------------------------------------------------------------------
    // 6. Output the final color
    // -------------------------------------------------------------------------
    // If you're using an sRGB framebuffer, consider applying gamma correction:
    // color = pow(color, vec3(1.0/2.2));

    f_color = vec4(color, 1.0);
}