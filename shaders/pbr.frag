//
// This fragment shader defines a reference implementation for Physically Based Shading of
// a microfacet surface material defined by a glTF model.
//
// References:
// [1] Real Shading in Unreal Engine 4
//     http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// [2] Physically Based Shading at Disney
//     http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] README.md - Environment Maps
//     https://github.com/KhronosGroup/glTF-WebGL-PBR/#environment-maps
// [4] "An Inexpensive BRDF Model for Physically based Rendering" by Christophe Schlick
//     https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf
// [5] "KHR_materials_clearcoat"
//     https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_materials_clearcoat

#version 450

#define DEBUG_NONE 0
#define DEBUG_NORMAL_SHADING 1
#define DEBUG_NORMAL_TEXTURE 2
#define DEBUG_NORMAL_GEOMETRY 3
#define DEBUG_TANGENT 4
#define DEBUG_BITANGENT 5
#define DEBUG_ALPHA 6
#define DEBUG_UV_0 7
#define DEBUG_UV_1 8
#define DEBUG_OCCLUSION 9
#define DEBUG_EMISSIVE 10
#define DEBUG_BASE_COLOR 11
#define DEBUG_ROUGHNESS 12
#define DEBUG_METALLIC 13
#define DEBUG_CLEARCOAT_FACTOR 14
#define DEBUG_CLEARCOAT_ROUGHNESS 15
#define DEBUG_CLEARCOAT_NORMAL 16
#define DEBUG_SHEEN_COLOR 17
#define DEBUG_SHEEN_ROUGHNESS 18
#define DEBUG_SPECULAR_FACTOR 19
#define DEBUG_SPECULAR_COLOR 20
#define DEBUG_TRANSMISSION_FACTOR 21
#define DEBUG_VOLUME_THICKNESS 22
#define DEBUG_DIFFUSE_TRANSMISSION_FACTOR 23
#define DEBUG_DIFFUSE_TRANSMISSION_COLOR_FACTOR 24
#define DEBUG_IRIDESCENCE_FACTOR 25
#define DEBUG_IRIDESCENCE_THICKNESS 26
#define DEBUG_ANISOTROPIC_STRENGTH 27
#define DEBUG_ANISOTROPIC_DIRECTION 28

precision highp float;

layout (location = 0) in vec3 v_Position;
layout (location = 1) in vec2 v_texcoord_0;
layout (location = 2) in vec2 v_texcoord_1;
layout (location = 3) in vec4 v_Color;
layout (location = 4) in vec3 v_Normal;
layout (location = 5) in mat3 v_TBN;

#include <shared.glsl>
#include <tonemapping.glsl>
#include <textures.glsl>
#include <functions.glsl>
#include <brdf.glsl>
#include <punctual.glsl>
#include <ibl.glsl>
#include <material_info.glsl>
#include <iridescence.glsl>

layout (location = 0) out vec4 g_finalColor;

void main()
{
    vec4 baseColor = getBaseColor();

    if (ALPHAMODE_OPAQUE) {
        baseColor.a = 1.0;
    }
    vec3 color = vec3(0);

    vec3 v = normalize(camera.u_Camera - v_Position);
    NormalInfo normalInfo = getNormalInfo(v);
    vec3 n = normalInfo.n;
    vec3 t = normalInfo.t;
    vec3 b = normalInfo.b;

    float NdotV = clampedDot(n, v);
    float TdotV = clampedDot(t, v);
    float BdotV = clampedDot(b, v);

    MaterialInfo materialInfo;
    materialInfo.baseColor = baseColor.rgb;

    // The default index of refraction of 1.5 yields a dielectric normal incidence reflectance of 0.04.
    materialInfo.ior = 1.5;
    materialInfo.f0_dielectric = vec3(0.04);
    materialInfo.specularWeight = 1.0;

    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    materialInfo.f90 = vec3(1.0);
    materialInfo.f90_dielectric = materialInfo.f90;

    if (MATERIAL_IOR) {
        materialInfo = getIorInfo(materialInfo);
    }

    if (MATERIAL_SPECULARGLOSSINESS) {
        materialInfo = getSpecularGlossinessInfo(materialInfo);
    }

    if (MATERIAL_METALLICROUGHNESS) {
        materialInfo = getMetallicRoughnessInfo(materialInfo);
    } else {
        materialInfo.metallic = 0.0;
    }

    if (MATERIAL_SHEEN) {
        materialInfo = getSheenInfo(materialInfo);
    }

    if (MATERIAL_CLEARCOAT) {
        materialInfo = getClearCoatInfo(materialInfo, normalInfo);
    }

    if (MATERIAL_SPECULAR) {
        materialInfo = getSpecularInfo(materialInfo);
    }

    if (MATERIAL_TRANSMISSION) {
        materialInfo = getTransmissionInfo(materialInfo);
    }

    if (MATERIAL_VOLUME) {
        materialInfo = getVolumeInfo(materialInfo);
    } else {
        materialInfo.thickness = 0.0;
        materialInfo.attenuationColor = vec3(1.0);
        materialInfo.attenuationDistance = 1./0.;
    }

    if (MATERIAL_IRIDESCENCE) {
        materialInfo = getIridescenceInfo(materialInfo);
    }

    if (MATERIAL_DIFFUSE_TRANSMISSION) {
        materialInfo = getDiffuseTransmissionInfo(materialInfo);
    }

    if (MATERIAL_ANISOTROPY) {
        materialInfo = getAnisotropyInfo(materialInfo, normalInfo);
    }

    materialInfo.perceptualRoughness = clamp(materialInfo.perceptualRoughness, 0.0, 1.0);
    materialInfo.metallic = clamp(materialInfo.metallic, 0.0, 1.0);

    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness.
    materialInfo.alphaRoughness = materialInfo.perceptualRoughness * materialInfo.perceptualRoughness;


    // LIGHTING
    vec3 f_specular_dielectric = vec3(0.0);
    vec3 f_specular_metal = vec3(0.0);
    vec3 f_diffuse = vec3(0.0);
    vec3 f_dielectric_brdf_ibl = vec3(0.0);
    vec3 f_metal_brdf_ibl = vec3(0.0);
    vec3 f_emissive = vec3(0.0);
    vec3 clearcoat_brdf = vec3(0.0);
    vec3 f_sheen = vec3(0.0);
    vec3 f_specular_transmission = vec3(0.0);
    vec3 f_diffuse_transmission = vec3(0.0);

    float clearcoatFactor = 0.0;
    vec3 clearcoatFresnel = vec3(0);

    float albedoSheenScaling = 1.0;
    float diffuseTransmissionThickness = 1.0;

    vec3 iridescenceFresnel_dielectric;
    vec3 iridescenceFresnel_metallic;
    if (MATERIAL_IRIDESCENCE) {
        iridescenceFresnel_dielectric = evalIridescence(1.0, materialInfo.iridescenceIor, NdotV, materialInfo.iridescenceThickness, materialInfo.f0_dielectric);
        iridescenceFresnel_metallic = evalIridescence(1.0, materialInfo.iridescenceIor, NdotV, materialInfo.iridescenceThickness, baseColor.rgb);

        if (materialInfo.iridescenceThickness == 0.0) {
            materialInfo.iridescenceFactor = 0.0;
        }
    }

    if (MATERIAL_DIFFUSE_TRANSMISSION) {
        if (MATERIAL_VOLUME) {
            diffuseTransmissionThickness = materialInfo.thickness *
                (length(vec3(object.u_ModelMatrix[0].xyz)) + length(vec3(object.u_ModelMatrix[1].xyz)) + length(vec3(object.u_ModelMatrix[2].xyz))) / 3.0;
        }
    }

    if (MATERIAL_CLEARCOAT) {
        clearcoatFactor = materialInfo.clearcoatFactor;
        clearcoatFresnel = F_Schlick(materialInfo.clearcoatF0, materialInfo.clearcoatF90, clampedDot(materialInfo.clearcoatNormal, v));
    }

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    const bool use_ibl = true;
#else
    const bool use_ibl = false;
#endif
    if (use_ibl || MATERIAL_TRANSMISSION) {
        f_diffuse = getDiffuseLight(n) * baseColor.rgb ;

        if (MATERIAL_DIFFUSE_TRANSMISSION) {
            vec3 diffuseTransmissionIBL = getDiffuseLight(-n) * materialInfo.diffuseTransmissionColorFactor;
            if (MATERIAL_VOLUME) {
                diffuseTransmissionIBL = applyVolumeAttenuation(diffuseTransmissionIBL, diffuseTransmissionThickness, materialInfo.attenuationColor, materialInfo.attenuationDistance);
            }
            f_diffuse = mix(f_diffuse, diffuseTransmissionIBL, materialInfo.diffuseTransmissionFactor);
        }

        if (MATERIAL_TRANSMISSION) {
            f_specular_transmission = getIBLVolumeRefraction(
                n, v,
                materialInfo.perceptualRoughness,
                baseColor.rgb, materialInfo.f0_dielectric, materialInfo.f90,
                v_Position, object.u_ModelMatrix, camera.u_ViewMatrix, camera.u_ProjectionMatrix,
                materialInfo.ior, materialInfo.thickness, materialInfo.attenuationColor, materialInfo.attenuationDistance, materialInfo.dispersion);
            f_diffuse = mix(f_diffuse, f_specular_transmission, materialInfo.transmissionFactor);
        }

        if (MATERIAL_ANISOTROPY) {
            f_specular_metal = getIBLRadianceAnisotropy(n, v, materialInfo.perceptualRoughness, materialInfo.anisotropyStrength, materialInfo.anisotropicB);
            f_specular_dielectric = f_specular_metal;
        } else {
            f_specular_metal = getIBLRadianceGGX(n, v, materialInfo.perceptualRoughness);
            f_specular_dielectric = f_specular_metal;
        }

        // Calculate fresnel mix for IBL

        vec3 f_metal_fresnel_ibl = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness, baseColor.rgb, 1.0);
        f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

        vec3 f_dielectric_fresnel_ibl = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness, materialInfo.f0_dielectric, materialInfo.specularWeight);
        f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric,  f_dielectric_fresnel_ibl);

        if (MATERIAL_IRIDESCENCE) {
            f_metal_brdf_ibl = mix(f_metal_brdf_ibl, f_specular_metal * iridescenceFresnel_metallic, materialInfo.iridescenceFactor);
            f_dielectric_brdf_ibl = mix(f_dielectric_brdf_ibl, rgb_mix(f_diffuse, f_specular_dielectric, iridescenceFresnel_dielectric), materialInfo.iridescenceFactor);
        }

        if (MATERIAL_CLEARCOAT) {
            clearcoat_brdf = getIBLRadianceGGX(materialInfo.clearcoatNormal, v, materialInfo.clearcoatRoughness);
        }

        if (MATERIAL_SHEEN) {
            f_sheen = getIBLRadianceCharlie(n, v, materialInfo.sheenRoughnessFactor, materialInfo.sheenColorFactor);
            albedoSheenScaling = 1.0 - max3(materialInfo.sheenColorFactor) * albedoSheenScalingLUT(NdotV, materialInfo.sheenRoughnessFactor);
        }

        color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, materialInfo.metallic);
        color = f_sheen + color * albedoSheenScaling;
        color = mix(color, clearcoat_brdf, clearcoatFactor * clearcoatFresnel);

        if (HAS_OCCLUSION_MAP) {
            float ao = 1.0;
            ao = texture(u_OcclusionSampler,  getOcclusionUV()).r;
            color = color * (1.0 + s.u_OcclusionStrength * (ao - 1.0));
        }

    }

    f_diffuse = vec3(0.0);
    f_specular_dielectric = vec3(0.0);
    f_specular_metal = vec3(0.0);
    vec3 f_dielectric_brdf = vec3(0.0);
    vec3 f_metal_brdf = vec3(0.0);

#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i)
    {
        Light light = u_Lights[i];

        vec3 pointToLight;
        if (light.type != LightType_Directional)
        {
            pointToLight = light.position - v_Position;
        }
        else
        {
            pointToLight = -light.direction;
        }

        // BSTF
        vec3 l = normalize(pointToLight);   // Direction from surface point to light
        vec3 h = normalize(l + v);          // Direction of the vector between l and v, called halfway vector
        float NdotL = clampedDot(n, l);
        float NdotV = clampedDot(n, v);
        float NdotH = clampedDot(n, h);
        float LdotH = clampedDot(l, h);
        float VdotH = clampedDot(v, h);

        vec3 dielectric_fresnel = F_Schlick(materialInfo.f0_dielectric * materialInfo.specularWeight, materialInfo.f90_dielectric, abs(VdotH));
        vec3 metal_fresnel = F_Schlick(baseColor.rgb, vec3(1.0), abs(VdotH));

        vec3 lightIntensity = getLighIntensity(light, pointToLight);

        vec3 l_diffuse = lightIntensity * NdotL * BRDF_lambertian(baseColor.rgb);
        vec3 l_specular_dielectric = vec3(0.0);
        vec3 l_specular_metal = vec3(0.0);
        vec3 l_dielectric_brdf = vec3(0.0);
        vec3 l_metal_brdf = vec3(0.0);
        vec3 l_clearcoat_brdf = vec3(0.0);
        vec3 l_sheen = vec3(0.0);
        float l_albedoSheenScaling = 1.0;


        if (MATERIAL_DIFFUSE_TRANSMISSION) {
            vec3 diffuse_btdf = lightIntensity * clampedDot(-n, l) * BRDF_lambertian(materialInfo.diffuseTransmissionColorFactor);

            if (MATERIAL_VOLUME) {
                diffuse_btdf = applyVolumeAttenuation(diffuse_btdf, diffuseTransmissionThickness, materialInfo.attenuationColor, materialInfo.attenuationDistance);
            }
            l_diffuse = mix(l_diffuse, diffuse_btdf, materialInfo.diffuseTransmissionFactor);
        }

        // BTDF (Bidirectional Transmittance Distribution Function)

        if (MATERIAL_TRANSMISSION) {
            // If the light ray travels through the geometry, use the point it exits the geometry again.
            // That will change the angle to the light source, if the material refracts the light ray.
            vec3 transmissionRay = getVolumeTransmissionRay(n, v, materialInfo.thickness, materialInfo.ior, object.u_ModelMatrix);
            pointToLight -= transmissionRay;
            l = normalize(pointToLight);

            vec3 transmittedLight = lightIntensity * getPunctualRadianceTransmission(n, v, l, materialInfo.alphaRoughness, materialInfo.f0_dielectric, materialInfo.f90, baseColor.rgb, materialInfo.ior);

            if (MATERIAL_VOLUME) {
                transmittedLight = applyVolumeAttenuation(transmittedLight, length(transmissionRay), materialInfo.attenuationColor, materialInfo.attenuationDistance);
            }
            l_diffuse = mix(l_diffuse, transmittedLight, materialInfo.transmissionFactor);
        }

        // Calculation of analytical light
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
        vec3 intensity = getLighIntensity(light, pointToLight);

        if (MATERIAL_ANISOTROPY) {
            l_specular_metal = intensity * NdotL * BRDF_specularGGXAnisotropy(materialInfo.alphaRoughness, materialInfo.anisotropyStrength, n, v, l, h, materialInfo.anisotropicT, materialInfo.anisotropicB);
            l_specular_dielectric = l_specular_metal;
        } else {
            l_specular_metal = intensity * NdotL * BRDF_specularGGX(materialInfo.alphaRoughness, NdotL, NdotV, NdotH);
            l_specular_dielectric = l_specular_metal;
        }

        l_metal_brdf = metal_fresnel * l_specular_metal;
        l_dielectric_brdf = mix(l_diffuse, l_specular_dielectric, dielectric_fresnel); // Do we need to handle vec3 fresnel here?

        if (MATERIAL_IRIDESCENCE) {
            l_metal_brdf = mix(l_metal_brdf, l_specular_metal * iridescenceFresnel_metallic, materialInfo.iridescenceFactor);
            l_dielectric_brdf = mix(l_dielectric_brdf, rgb_mix(l_diffuse, l_specular_dielectric, iridescenceFresnel_dielectric), materialInfo.iridescenceFactor);
        }

        if (MATERIAL_CLEARCOAT) {
            l_clearcoat_brdf = intensity * getPunctualRadianceClearCoat(materialInfo.clearcoatNormal, v, l, h, VdotH,
                materialInfo.clearcoatF0, materialInfo.clearcoatF90, materialInfo.clearcoatRoughness);
        }

        if (MATERIAL_SHEEN) {
            l_sheen = intensity * getPunctualRadianceSheen(materialInfo.sheenColorFactor, materialInfo.sheenRoughnessFactor, NdotL, NdotV, NdotH);
            l_albedoSheenScaling = min(1.0 - max3(materialInfo.sheenColorFactor) * albedoSheenScalingLUT(NdotV, materialInfo.sheenRoughnessFactor),
                1.0 - max3(materialInfo.sheenColorFactor) * albedoSheenScalingLUT(NdotL, materialInfo.sheenRoughnessFactor));
        }

        vec3 l_color = mix(l_dielectric_brdf, l_metal_brdf, materialInfo.metallic);
        l_color = l_sheen + l_color * l_albedoSheenScaling;
        l_color = mix(l_color, l_clearcoat_brdf, clearcoatFactor * clearcoatFresnel);
        color += l_color;
    }
#endif // USE_PUNCTUAL

    f_emissive = s.u_EmissiveFactor;
    if (MATERIAL_EMISSIVE_STRENGTH) {
        f_emissive *= material.u_EmissiveStrength;
    }
    if (HAS_EMISSIVE_MAP) {
        f_emissive *= texture(u_EmissiveSampler, getEmissiveUV()).rgb;
    }


    if (MATERIAL_UNLIT) {
        color = baseColor.rgb;
    } else {
        if (NOT_TRIANGLE && !HAS_NORMAL_VEC3) {
            //Points or Lines with no NORMAL attribute SHOULD be rendered without lighting and instead use the sum of the base color value and the emissive value.
            color = f_emissive + baseColor.rgb;
        } else {
            color = f_emissive * (1.0 - clearcoatFactor * clearcoatFresnel) + color;
        }
    }



    if (ALPHAMODE_MASK) {
        // Late discard to avoid sampling artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
        if (baseColor.a < material.u_AlphaCutoff)
        {
            discard;
        }
        baseColor.a = 1.0;
    }

#if DEBUG == DEBUG_NONE

#ifdef LINEAR_OUTPUT
    g_finalColor = vec4(color.rgb, baseColor.a);
#else
    g_finalColor = vec4(toneMap(color), baseColor.a);
#endif

#else
    // In case of missing data for a debug view, render a checkerboard.
    g_finalColor = vec4(1.0);
    {
        float frequency = 0.02;
        float gray = 0.9;

        vec2 v1 = step(0.5, fract(frequency * gl_FragCoord.xy));
        vec2 v2 = step(0.5, vec2(1.0) - fract(frequency * gl_FragCoord.xy));
        g_finalColor.rgb *= gray + v1.x * v1.y + v2.x * v2.y;
    }
#endif

    // Debug views:

    // Generic:
#if DEBUG == DEBUG_UV_0
    if (HAS_TEXCOORD_0_VEC2) {
        g_finalColor.rgb = vec3(v_texcoord_0, 0);
    }
#endif
#if DEBUG == DEBUG_UV_1
    if (HAS_TEXCOORD_1_VEC2) {
        g_finalColor.rgb = vec3(v_texcoord_1, 0);
    }
#endif
#if DEBUG == DEBUG_NORMAL_TEXTURE
    if (HAS_NORMAL_MAP) {
        g_finalColor.rgb = (normalInfo.ntex + 1.0) / 2.0;
    }
#endif
#if DEBUG == DEBUG_NORMAL_SHADING
    g_finalColor.rgb = (n + 1.0) / 2.0;
#endif
#if DEBUG == DEBUG_NORMAL_GEOMETRY
    g_finalColor.rgb = (normalInfo.ng + 1.0) / 2.0;
#endif
#if DEBUG == DEBUG_TANGENT
    g_finalColor.rgb = (normalInfo.t + 1.0) / 2.0;
#endif
#if DEBUG == DEBUG_BITANGENT
    g_finalColor.rgb = (normalInfo.b + 1.0) / 2.0;
#endif
#if DEBUG == DEBUG_ALPHA
    g_finalColor.rgb = vec3(baseColor.a);
#endif
#if DEBUG == DEBUG_OCCLUSION && defined(HAS_OCCLUSION_MAP)
    g_finalColor.rgb = vec3(ao);
#endif
#if DEBUG == DEBUG_EMISSIVE
    g_finalColor.rgb = linearTosRGB(f_emissive);
#endif

    // MR:
    if (MATERIAL_METALLICROUGHNESS) {
#if DEBUG == DEBUG_METALLIC
        g_finalColor.rgb = vec3(materialInfo.metallic);
#endif
#if DEBUG == DEBUG_ROUGHNESS
        g_finalColor.rgb = vec3(materialInfo.perceptualRoughness);
#endif
#if DEBUG == DEBUG_BASE_COLOR
        g_finalColor.rgb = linearTosRGB(materialInfo.baseColor);
#endif
    }

    // Clearcoat:
    if (MATERIAL_CLEARCOAT) {
#if DEBUG == DEBUG_CLEARCOAT_FACTOR
        g_finalColor.rgb = vec3(materialInfo.clearcoatFactor);
#endif
#if DEBUG == DEBUG_CLEARCOAT_ROUGHNESS
        g_finalColor.rgb = vec3(materialInfo.clearcoatRoughness);
#endif
#if DEBUG == DEBUG_CLEARCOAT_NORMAL
        g_finalColor.rgb = (materialInfo.clearcoatNormal + vec3(1)) / 2.0;
#endif
    }

    // Sheen:
    if (MATERIAL_SHEEN) {
#if DEBUG == DEBUG_SHEEN_COLOR
        g_finalColor.rgb = materialInfo.sheenColorFactor;
#endif
#if DEBUG == DEBUG_SHEEN_ROUGHNESS
        g_finalColor.rgb = vec3(materialInfo.sheenRoughnessFactor);
#endif
    }

    // Specular:
    if (MATERIAL_SPECULAR) {
#if DEBUG == DEBUG_SPECULAR_FACTOR
        g_finalColor.rgb = vec3(materialInfo.specularWeight);
#endif

#if DEBUG == DEBUG_SPECULAR_COLOR
        vec3 specularTexture = vec3(1.0);
        if (HAS_SPECULAR_COLOR_MAP) {
            specularTexture.rgb = texture(u_SpecularColorSampler, getSpecularColorUV()).rgb;
        }
        g_finalColor.rgb = material.u_KHR_materials_specular_specularColorFactor * specularTexture.rgb;
#endif
    }

    // Transmission, Volume:
    if (MATERIAL_TRANSMISSION) {
#if DEBUG == DEBUG_TRANSMISSION_FACTOR
        g_finalColor.rgb = vec3(materialInfo.transmissionFactor);
#endif
    }
    if (MATERIAL_VOLUME) {
#if DEBUG == DEBUG_VOLUME_THICKNESS
        g_finalColor.rgb = vec3(materialInfo.thickness / material.u_ThicknessFactor);
#endif
    }

    // Iridescence:
    if (MATERIAL_IRIDESCENCE) {
#if DEBUG == DEBUG_IRIDESCENCE_FACTOR
        g_finalColor.rgb = vec3(materialInfo.iridescenceFactor);
#endif
#if DEBUG == DEBUG_IRIDESCENCE_THICKNESS
        g_finalColor.rgb = vec3(materialInfo.iridescenceThickness / 1200.0);
#endif
    }

    // Anisotropy:
    if (MATERIAL_ANISOTROPY) {
#if DEBUG == DEBUG_ANISOTROPIC_STRENGTH
        g_finalColor.rgb = vec3(materialInfo.anisotropyStrength);
#endif
#if DEBUG == DEBUG_ANISOTROPIC_DIRECTION
        vec2 direction = vec2(1.0, 0.0);
        if (HAS_ANISOTROPY_MAP) {
            direction = texture(u_AnisotropySampler, getAnisotropyUV()).xy;
            direction = direction * 2.0 - vec2(1.0); // [0, 1] -> [-1, 1]
        }
        vec2 directionRotation = material.u_Anisotropy.xy; // cos(theta), sin(theta)
        mat2 rotationMatrix = mat2(directionRotation.x, directionRotation.y, -directionRotation.y, directionRotation.x);
        direction = (direction + vec2(1.0)) * 0.5; // [-1, 1] -> [0, 1]

        g_finalColor.rgb = vec3(direction, 0.0);
#endif
    }

    // Diffuse Transmission:
    if (MATERIAL_DIFFUSE_TRANSMISSION) {
#if DEBUG == DEBUG_DIFFUSE_TRANSMISSION_FACTOR
        g_finalColor.rgb = linearTosRGB(vec3(materialInfo.diffuseTransmissionFactor));
#endif
#if DEBUG == DEBUG_DIFFUSE_TRANSMISSION_COLOR_FACTOR
        g_finalColor.rgb = linearTosRGB(materialInfo.diffuseTransmissionColorFactor);
#endif
    }
}