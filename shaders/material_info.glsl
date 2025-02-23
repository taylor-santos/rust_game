struct MaterialInfo
{
    float ior;
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0_dielectric;

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])

    float fresnel_w;

    vec3 f90;                       // reflectance color at grazing angle
    vec3 f90_dielectric;
    float metallic;

    vec3 baseColor;

    float sheenRoughnessFactor;
    vec3 sheenColorFactor;

    vec3 clearcoatF0;
    vec3 clearcoatF90;
    float clearcoatFactor;
    vec3 clearcoatNormal;
    float clearcoatRoughness;

    // KHR_materials_specular 
    float specularWeight; // product of specularFactor and specularTexture.a

    float transmissionFactor;

    float thickness;
    vec3 attenuationColor;
    float attenuationDistance;

    // KHR_materials_iridescence
    float iridescenceFactor;
    float iridescenceIor;
    float iridescenceThickness;

    float diffuseTransmissionFactor;
    vec3 diffuseTransmissionColorFactor;

    // KHR_materials_anisotropy
    vec3 anisotropicT;
    vec3 anisotropicB;
    float anisotropyStrength;

    // KHR_materials_dispersion
    float dispersion;
};

// Get normal, tangent and bitangent vectors.
NormalInfo getNormalInfo(vec3 v)
{
    vec2 UV = getNormalUV();
    vec2 uv_dx = dFdx(UV);
    vec2 uv_dy = dFdy(UV);

    if (length(uv_dx) <= 1e-2) {
      uv_dx = vec2(1.0, 0.0);
    }

    if (length(uv_dy) <= 1e-2) {
      uv_dy = vec2(0.0, 1.0);
    }

    vec3 t_ = (uv_dy.t * dFdx(v_Position) - uv_dx.t * dFdy(v_Position)) /
        (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);

    vec3 n, t, b, ng;

    // Compute geometrical TBN:
    if (HAS_NORMAL_VEC3) {
        if (HAS_TANGENT_VEC4) {
            // Trivial TBN computation, present as vertex attribute.
            // Normalize eigenvectors as matrix is linearly interpolated.
            t = normalize(v_TBN[0]);
            b = normalize(v_TBN[1]);
            ng = normalize(v_TBN[2]);
        } else {
            // Normals are either present as vertex attributes or approximated.
            ng = normalize(v_Normal);
            t = normalize(t_ - ng * dot(ng, t_));
            b = cross(ng, t);
        }
    } else {
        ng = normalize(cross(dFdx(v_Position), dFdy(v_Position)));
        t = normalize(t_ - ng * dot(ng, t_));
        b = cross(ng, t);
    }

    if (NOT_TRIANGLE) {
        // For a back-facing surface, the tangential basis vectors are negated.
        if (gl_FrontFacing == false)
        {
            t *= -1.0;
            b *= -1.0;
            ng *= -1.0;
        }
    }

    // Compute normals:
    NormalInfo info;
    info.ng = ng;
    if (HAS_NORMAL_MAP) {
        info.ntex = texture(u_NormalSampler, UV).rgb * 2.0 - vec3(1.0);
        info.ntex *= vec3(s.u_NormalScale, s.u_NormalScale, 1.0);
        info.ntex = normalize(info.ntex);
        info.n = normalize(mat3(t, b, ng) * info.ntex);
    } else {
        info.n = ng;
    }
    info.t = t;
    info.b = b;
    return info;
}

vec3 getClearcoatNormal(NormalInfo normalInfo)
{
    if (HAS_CLEARCOAT_NORMAL_MAP) {
        vec3 n = texture(u_ClearcoatNormalSampler, getClearcoatNormalUV()).rgb * 2.0 - vec3(1.0);
        n *= vec3(s.u_ClearcoatNormalScale, s.u_ClearcoatNormalScale, 1.0);
        n = mat3(normalInfo.t, normalInfo.b, normalInfo.ng) * normalize(n);
        return n;
    } else {
        return normalInfo.ng;
    }
}

vec4 getBaseColor()
{
    vec4 baseColor = vec4(1);

    if (MATERIAL_SPECULARGLOSSINESS) {
        baseColor = material.u_DiffuseFactor;
    } else if (MATERIAL_METALLICROUGHNESS) {
        baseColor = material.u_BaseColorFactor;
    }

    if (HAS_DIFFUSE_MAP) {
        if (MATERIAL_SPECULARGLOSSINESS) {
            baseColor *= texture(u_DiffuseSampler, getDiffuseUV());
        }
    } else if (HAS_BASE_COLOR_MAP) {
        if (MATERIAL_METALLICROUGHNESS) {
            baseColor *= texture(u_BaseColorSampler, getBaseColorUV());
        }
    }

    return baseColor * getVertexColor();
}

MaterialInfo getSpecularGlossinessInfo(MaterialInfo info)
{
    info.f0_dielectric = material.u_SpecularFactor;
    info.perceptualRoughness = material.u_GlossinessFactor;

    if (HAS_SPECULAR_GLOSSINESS_MAP) {
        vec4 sgSample = texture(u_SpecularGlossinessSampler, getSpecularGlossinessUV());
        info.perceptualRoughness *= sgSample.a ; // glossiness to roughness
        info.f0_dielectric *= sgSample.rgb; // specular
    }

    info.perceptualRoughness = 1.0 - info.perceptualRoughness; // 1 - glossiness
    return info;
}

MaterialInfo getMetallicRoughnessInfo(MaterialInfo info)
{
    info.metallic = material.u_MetallicFactor;
    info.perceptualRoughness = material.u_RoughnessFactor;

    if (HAS_METALLIC_ROUGHNESS_MAP) {
        // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
        // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
        vec4 mrSample = texture(u_MetallicRoughnessSampler, getMetallicRoughnessUV());
        info.perceptualRoughness *= mrSample.g;
        info.metallic *= mrSample.b;
    }

    return info;
}

MaterialInfo getSheenInfo(MaterialInfo info)
{
    info.sheenColorFactor = material.u_SheenColorFactor;
    info.sheenRoughnessFactor = material.u_SheenRoughnessFactor;

    if (HAS_SHEEN_COLOR_MAP) {
        vec4 sheenColorSample = texture(u_SheenColorSampler, getSheenColorUV());
        info.sheenColorFactor *= sheenColorSample.rgb;
    }

    if (HAS_SHEEN_ROUGHNESS_MAP) {
        vec4 sheenRoughnessSample = texture(u_SheenRoughnessSampler, getSheenRoughnessUV());
        info.sheenRoughnessFactor *= sheenRoughnessSample.a;
    }
    return info;
}

MaterialInfo getSpecularInfo(MaterialInfo info)
{   
    vec4 specularTexture = vec4(1.0);
    if (HAS_SPECULAR_MAP) {
        specularTexture.a = texture(u_SpecularSampler, getSpecularUV()).a;
    }
    if (HAS_SPECULAR_COLOR_MAP) {
        specularTexture.rgb = texture(u_SpecularColorSampler, getSpecularColorUV()).rgb;
    }

    info.f0_dielectric = min(info.f0_dielectric * material.u_KHR_materials_specular_specularColorFactor * specularTexture.rgb, vec3(1.0));
    info.specularWeight = material.u_KHR_materials_specular_specularFactor * specularTexture.a;
    info.f90_dielectric = vec3(info.specularWeight);
    return info;
}

MaterialInfo getTransmissionInfo(MaterialInfo info)
{
    info.transmissionFactor = material.u_TransmissionFactor;

    if (HAS_TRANSMISSION_MAP) {
        vec4 transmissionSample = texture(u_TransmissionSampler, getTransmissionUV());
        info.transmissionFactor *= transmissionSample.r;
    }

    if (MATERIAL_DISPERSION) {
        info.dispersion = material.u_Dispersion;
    } else {
        info.dispersion = 0.0;
    }
    return info;
}

MaterialInfo getVolumeInfo(MaterialInfo info)
{
    info.thickness = material.u_ThicknessFactor;
    info.attenuationColor = material.u_AttenuationColor;
    info.attenuationDistance = material.u_AttenuationDistance;

    if (HAS_THICKNESS_MAP) {
        vec4 thicknessSample = texture(u_ThicknessSampler, getThicknessUV());
        info.thickness *= thicknessSample.g;
    }
    return info;
}

MaterialInfo getIridescenceInfo(MaterialInfo info)
{
    info.iridescenceFactor = material.u_IridescenceFactor;
    info.iridescenceIor = material.u_IridescenceIor;
    info.iridescenceThickness = material.u_IridescenceThicknessMaximum;

    if (HAS_IRIDESCENCE_MAP) {
        info.iridescenceFactor *= texture(u_IridescenceSampler, getIridescenceUV()).r;
    }

    if (HAS_IRIDESCENCE_THICKNESS_MAP) {
        float thicknessSampled = texture(u_IridescenceThicknessSampler, getIridescenceThicknessUV()).g;
        float thickness = mix(material.u_IridescenceThicknessMinimum, material.u_IridescenceThicknessMaximum, thicknessSampled);
        info.iridescenceThickness = thickness;
    }

    return info;
}

MaterialInfo getDiffuseTransmissionInfo(MaterialInfo info)
{
    info.diffuseTransmissionFactor = material.u_DiffuseTransmissionFactor;
    info.diffuseTransmissionColorFactor = material.u_DiffuseTransmissionColorFactor;

    if (HAS_DIFFUSE_TRANSMISSION_MAP) {
        info.diffuseTransmissionFactor *= texture(u_DiffuseTransmissionSampler, getDiffuseTransmissionUV()).a;
    }

    if (HAS_DIFFUSE_TRANSMISSION_COLOR_MAP) {
        info.diffuseTransmissionColorFactor *= texture(u_DiffuseTransmissionColorSampler, getDiffuseTransmissionColorUV()).rgb;
    }

    return info;
}

MaterialInfo getClearCoatInfo(MaterialInfo info, NormalInfo normalInfo)
{
    info.clearcoatFactor = material.u_ClearcoatFactor;
    info.clearcoatRoughness = material.u_ClearcoatRoughnessFactor;
    info.clearcoatF0 = vec3(pow((info.ior - 1.0) / (info.ior + 1.0), 2.0));
    info.clearcoatF90 = vec3(1.0);

    if (HAS_CLEARCOAT_MAP) {
        vec4 clearcoatSample = texture(u_ClearcoatSampler, getClearcoatUV());
        info.clearcoatFactor *= clearcoatSample.r;
    }

    if (HAS_CLEARCOAT_ROUGHNESS_MAP) {
        vec4 clearcoatSampleRoughness = texture(u_ClearcoatRoughnessSampler, getClearcoatRoughnessUV());
        info.clearcoatRoughness *= clearcoatSampleRoughness.g;
    }

    info.clearcoatNormal = getClearcoatNormal(normalInfo);
    info.clearcoatRoughness = clamp(info.clearcoatRoughness, 0.0, 1.0);
    return info;
}


MaterialInfo getIorInfo(MaterialInfo info)
{
    info.f0_dielectric = vec3(pow(( material.u_Ior - 1.0) /  (material.u_Ior + 1.0), 2.0));
    info.ior = material.u_Ior;
    return info;
}


MaterialInfo getAnisotropyInfo(MaterialInfo info, NormalInfo normalInfo)
{
    vec2 direction = vec2(1.0, 0.0);
    float strengthFactor = 1.0;
    if (HAS_ANISOTROPY_MAP) {
        vec3 anisotropySample = texture(u_AnisotropySampler, getAnisotropyUV()).xyz;
        direction = anisotropySample.xy * 2.0 - vec2(1.0);
        strengthFactor = anisotropySample.z;
    }
    vec2 directionRotation = material.u_Anisotropy.xy; // cos(theta), sin(theta)
    mat2 rotationMatrix = mat2(directionRotation.x, directionRotation.y, -directionRotation.y, directionRotation.x);
    direction = rotationMatrix * direction.xy;

    info.anisotropicT = mat3(normalInfo.t, normalInfo.b, normalInfo.n) * normalize(vec3(direction, 0.0));
    info.anisotropicB = cross(normalInfo.ng, info.anisotropicT);
    info.anisotropyStrength = clamp(material.u_Anisotropy.z * strengthFactor, 0.0, 1.0);
    return info;
}


float albedoSheenScalingLUT(float NdotV, float sheenRoughnessFactor)
{
    return texture(u_SheenELUT, vec2(NdotV, sheenRoughnessFactor)).r;
}
