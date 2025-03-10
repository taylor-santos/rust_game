use crate::material::{AlphaMode, Material};
use std::fmt::Formatter;
use vulkano::shader::SpecializationConstant;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/pbr.vert",
        include: ["shaders"],
        define: [
            // ("USE_INSTANCING", "1"),
            // ("USE_MORPHING", "1"),
            // ("USE_SKINNING", "1"),
        ],
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag",
        include: ["shaders"],
        define: [
            ("USE_IBL", "1"),
            // ("USE_PUNCTUAL", "1"),

            ("DEBUG", "DEBUG_NONE"),
            // ("DEBUG", "DEBUG_ALPHA"),
            // ("DEBUG", "DEBUG_ANISOTROPIC_DIRECTION"),
            // ("DEBUG", "DEBUG_ANISOTROPIC_STRENGTH"),
            // ("DEBUG", "DEBUG_BASE_COLOR"),
            // ("DEBUG", "DEBUG_BITANGENT"),
            // ("DEBUG", "DEBUG_CLEARCOAT_FACTOR"),
            // ("DEBUG", "DEBUG_CLEARCOAT_NORMAL"),
            // ("DEBUG", "DEBUG_CLEARCOAT_ROUGHNESS"),
            // ("DEBUG", "DEBUG_DIFFUSE_TRANSMISSION_COLOR_FACTOR"),
            // ("DEBUG", "DEBUG_DIFFUSE_TRANSMISSION_FACTOR"),
            // ("DEBUG", "DEBUG_IRIDESCENCE_FACTOR"),
            // ("DEBUG", "DEBUG_IRIDESCENCE_THICKNESS"),
            // ("DEBUG", "DEBUG_METALLIC"),
            // ("DEBUG", "DEBUG_NONE"),
            // ("DEBUG", "DEBUG_NORMAL_GEOMETRY"),
            // ("DEBUG", "DEBUG_NORMAL_SHADING"),
            // ("DEBUG", "DEBUG_NORMAL_TEXTURE"),
            // ("DEBUG", "DEBUG_OCCLUSION"),
            // ("DEBUG", "DEBUG_ROUGHNESS"),
            // ("DEBUG", "DEBUG_SHEEN_COLOR"),
            // ("DEBUG", "DEBUG_SHEEN_ROUGHNESS"),
            // ("DEBUG", "DEBUG_SPECULAR_COLOR"),
            // ("DEBUG", "DEBUG_SPECULAR_FACTOR"),
            // ("DEBUG", "DEBUG_TANGENT"),
            // ("DEBUG", "DEBUG_TRANSMISSION_FACTOR"),
            // ("DEBUG", "DEBUG_UV_0"),
            // ("DEBUG", "DEBUG_UV_1"),
            // ("DEBUG", "DEBUG_VOLUME_THICKNESS"),
        ],
    }
}

pub mod cubemap_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/cubemap.vert",
        include: ["shaders"],
    }
}

pub mod cubemap_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/cubemap.frag",
        include: ["shaders"],
    }
}

pub mod tonemap_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/tonemap.vert",
        include: ["shaders"],
    }
}

pub mod tonemap_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/tonemap.frag",
        include: ["shaders"],
        define: [
            ("TONEMAP_KHR_PBR_NEUTRAL", "1"),
            // ("TONEMAP_ACES_HILL", "1"),
            // ("TONEMAP_ACES_HILL_EXPOSURE_BOOST", "1"),
            // ("TONEMAP_ACES_NARKOWICZ", "1"),
            // ("LINEAR_OUTPUT", "1"),
        ]
    }
}

macro_rules! unwrap {
    ($base:ident $( $rest:tt )*) => {{
        #[allow(clippy::redundant_closure_call)]
        (|| Some(unwrap_chain!($base $( $rest )*)) )().unwrap_or_default().into()
    }};
}

macro_rules! unwrap_chain {
    // Match a method call with a trailing '?': .method(...)?
    ($acc:tt . $method:ident ( $( $args:tt )* ) ? $( $rest:tt )*) => {
        unwrap_chain!(( $acc.$method( $( $args )* ).as_ref()? ) $( $rest )*)
    };
    // Match a normal method call: .method(...)
    ($acc:tt . $method:ident ( $( $args:tt )* ) $( $rest:tt )*) => {
        unwrap_chain!(( $acc.$method( $( $args )* ) ) $( $rest )*)
    };
    // Match a field with a trailing '?': .field?
    ($acc:tt . $field:ident ? $( $rest:tt )+) => {
        unwrap_chain!( $acc.$field.as_ref()? $( $rest )*)
    };
    // When an optional field access is encountered with a leading '?': ? . field
    ($acc:tt ? . $field:ident $( $rest:tt )*) => {
        unwrap_chain!( $acc.as_ref()? . $field $( $rest )*)
    };
    // When a normal field access is encountered: .field
    ($acc:tt . $field:ident $( $rest:tt )*) => {
        unwrap_chain!(( $acc.$field ) $( $rest )*)
    };
    // Handle a trailing '?' after any expression.
    ($acc:tt ?) => {
        unwrap_chain!(( $acc?.clone() ))
    };
    // Base case: no more tokens; simply return the accumulator.
    ($e:tt) => { $e };
}

impl From<&Material> for fs::MatSamplers {
    fn from(mat: &Material) -> Self {
        #[allow(clippy::useless_conversion)]
        Self {
            u_NormalScale: unwrap!(mat.normal_texture?.scale),
            u_NormalUVSet: unwrap!(mat.normal_texture?.info.tex_coord()),
            u_NormalUVTransform: unwrap!(mat.normal_texture?.info.transform?),

            u_EmissiveFactor: unwrap!(mat.emissive_factor),
            u_EmissiveUVSet: unwrap!(mat.emissive_texture?.tex_coord()),
            u_EmissiveUVTransform: unwrap!(mat.emissive_texture?.transform?),

            u_OcclusionStrength: unwrap!(mat.occlusion_texture?.strength),
            u_OcclusionUVSet: unwrap!(mat.occlusion_texture?.info.tex_coord()),
            u_OcclusionUVTransform: unwrap!(mat.occlusion_texture?.info.transform?),

            u_BaseColorUVSet: unwrap!(mat.pbr_metallic_roughness.base_color_texture?.tex_coord()),
            u_BaseColorUVTransform: unwrap!(
                mat.pbr_metallic_roughness.base_color_texture?.transform?
            ),

            u_MetallicRoughnessUVSet: unwrap!(mat
                .pbr_metallic_roughness
                .metallic_roughness_texture?
                .tex_coord()),
            u_MetallicRoughnessUVTransform: unwrap!(
                mat.pbr_metallic_roughness
                    .metallic_roughness_texture?
                    .transform?
            ),

            u_SheenColorUVSet: unwrap!(mat.sheen?.color_texture?.tex_coord()),
            u_SheenColorUVTransform: unwrap!(mat.sheen?.color_texture?.transform?),
            u_SheenRoughnessUVSet: unwrap!(mat.sheen?.roughness_texture?.tex_coord()),
            u_SheenRoughnessUVTransform: unwrap!(mat.sheen?.roughness_texture?.transform?),

            u_DiffuseUVSet: unwrap!(mat.pbr_specular_glossiness?.diffuse_texture?.tex_coord()),
            u_DiffuseUVTransform: unwrap!(mat.pbr_specular_glossiness?.diffuse_texture?.transform?),

            u_SpecularGlossinessUVSet: unwrap!(mat
                .pbr_specular_glossiness?
                .specular_glossiness_texture?
                .tex_coord()),
            u_SpecularGlossinessUVTransform: unwrap!(
                mat.pbr_specular_glossiness?
                    .specular_glossiness_texture?
                    .transform?
            ),

            u_ClearcoatUVSet: unwrap!(mat.clearcoat?.texture?.tex_coord()),
            u_ClearcoatUVTransform: unwrap!(mat.clearcoat?.texture?.transform?),
            u_ClearcoatRoughnessUVSet: unwrap!(mat.clearcoat?.roughness_texture?.tex_coord()),
            u_ClearcoatRoughnessUVTransform: unwrap!(mat.clearcoat?.roughness_texture?.transform?),
            u_ClearcoatNormalUVSet: unwrap!(mat.clearcoat?.normal_texture?.info.tex_coord()),
            u_ClearcoatNormalUVTransform: unwrap!(mat.clearcoat?.normal_texture?.info.transform?),
            u_ClearcoatNormalScale: unwrap!(mat.clearcoat?.normal_texture?.scale),

            u_SpecularUVSet: unwrap!(mat.specular?.texture?.tex_coord()),
            u_SpecularUVTransform: unwrap!(mat.specular?.texture?.transform?),
            u_SpecularColorUVSet: unwrap!(mat.specular?.color_texture?.tex_coord()),
            u_SpecularColorUVTransform: unwrap!(mat.specular?.color_texture?.transform?),

            u_TransmissionUVSet: unwrap!(mat.transmission?.texture?.tex_coord()),
            u_TransmissionUVTransform: unwrap!(mat.transmission?.texture?.transform?),

            u_ThicknessUVSet: unwrap!(mat.volume?.thickness_texture?.tex_coord()),
            u_ThicknessUVTransform: unwrap!(mat.volume?.thickness_texture?.transform?),

            u_IridescenceUVSet: unwrap!(mat.iridescence?.texture?.tex_coord()),
            u_IridescenceUVTransform: unwrap!(mat.iridescence?.texture?.transform?),
            u_IridescenceThicknessUVSet: unwrap!(mat.iridescence?.thickness_texture?.tex_coord()),
            u_IridescenceThicknessUVTransform: unwrap!(
                mat.iridescence?.thickness_texture?.transform?
            ),

            u_DiffuseTransmissionUVSet: unwrap!(mat.diffuse_transmission?.texture?.tex_coord()),
            u_DiffuseTransmissionUVTransform: unwrap!(
                mat.diffuse_transmission?.texture?.transform?
            ),
            u_DiffuseTransmissionColorUVSet: unwrap!(mat
                .diffuse_transmission?
                .color_texture?
                .tex_coord()),
            u_DiffuseTransmissionColorUVTransform: unwrap!(
                mat.diffuse_transmission?.color_texture?.transform?
            ),

            u_AnisotropyUVSet: unwrap!(mat.anisotropy?.texture?.tex_coord()),
            u_AnisotropyUVTransform: unwrap!(mat.anisotropy?.texture?.transform?),
        }
    }
}

impl From<&Material> for fs::Material {
    #[allow(non_snake_case)]
    fn from(mat: &Material) -> Self {
        Self {
            u_MetallicFactor: unwrap!(mat.pbr_metallic_roughness.metallic_factor),
            u_RoughnessFactor: unwrap!(mat.pbr_metallic_roughness.roughness_factor),
            u_BaseColorFactor: unwrap!(mat.pbr_metallic_roughness.base_color_factor),
            u_SpecularFactor: unwrap!(mat.pbr_specular_glossiness?.specular_factor),
            u_DiffuseFactor: unwrap!(mat.pbr_specular_glossiness?.diffuse_factor),
            u_GlossinessFactor: unwrap!(mat.pbr_specular_glossiness?.glossiness_factor),
            u_SheenRoughnessFactor: unwrap!(mat.sheen?.roughness_factor),
            u_SheenColorFactor: unwrap!(mat.sheen?.color_factor),
            u_ClearcoatFactor: unwrap!(mat.clearcoat?.factor),
            u_ClearcoatRoughnessFactor: unwrap!(mat.clearcoat?.roughness_factor),
            u_KHR_materials_specular_specularColorFactor: unwrap!(mat.specular?.color_factor),
            u_KHR_materials_specular_specularFactor: unwrap!(mat.specular?.factor),
            u_TransmissionFactor: unwrap!(mat.transmission?.factor),
            u_ThicknessFactor: unwrap!(mat.volume?.thickness_factor),
            u_AttenuationColor: unwrap!(mat.volume?.attenuation_color),
            u_AttenuationDistance: unwrap!(mat.volume?.attenuation_distance),
            u_IridescenceFactor: unwrap!(mat.iridescence?.factor),
            u_IridescenceIor: unwrap!(mat.iridescence?.ior),
            u_IridescenceThicknessMinimum: unwrap!(mat.iridescence?.thickness_minimum),
            u_IridescenceThicknessMaximum: unwrap!(mat.iridescence?.thickness_maximum),
            u_DiffuseTransmissionFactor: unwrap!(mat.diffuse_transmission?.factor),
            u_DiffuseTransmissionColorFactor: unwrap!(mat.diffuse_transmission?.color_factor),
            u_EmissiveStrength: unwrap!(mat.emissive_strength?),
            u_Ior: unwrap!(mat.ior?),
            u_Anisotropy: unwrap!(mat.anisotropy?),
            u_Dispersion: unwrap!(mat.dispersion?),
            u_AlphaCutoff: unwrap!(mat.alpha_cutoff),
            u_vertNormalUVTransform: unwrap!(mat.normal_texture?.info.transform?),
        }
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct SpecializationConstants {
    pub object_constants: ObjectSpecializationConstants,
    pub material_constants: MaterialSpecializationConstants,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct ObjectSpecializationConstants {
    pub HAS_NORMAL_VEC3: bool,
    pub HAS_TANGENT_VEC4: bool,
    pub HAS_TEXCOORD_0_VEC2: bool,
    pub HAS_TEXCOORD_1_VEC2: bool,
    pub HAS_COLOR_0_VEC3: bool,
    pub HAS_COLOR_0_VEC4: bool,
    pub NOT_TRIANGLE: bool,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct MaterialSpecializationConstants {
    pub DOUBLE_SIDED: bool,
    pub ALPHAMODE_OPAQUE: bool,
    pub ALPHAMODE_MASK: bool,
    pub ALPHAMODE_BLEND: bool,
    pub MATERIAL_UNLIT: bool,
    pub HAS_NORMAL_MAP: bool,
    pub HAS_NORMAL_UV_TRANSFORM: bool,
    pub HAS_VERT_NORMAL_UV_TRANSFORM: bool,
    pub HAS_EMISSIVE_MAP: bool,
    pub HAS_EMISSIVE_UV_TRANSFORM: bool,
    pub HAS_OCCLUSION_MAP: bool,
    pub HAS_OCCLUSION_UV_TRANSFORM: bool,
    pub HAS_BASE_COLOR_MAP: bool,
    pub HAS_BASECOLOR_UV_TRANSFORM: bool,
    pub MATERIAL_METALLICROUGHNESS: bool,
    pub HAS_METALLIC_ROUGHNESS_MAP: bool,
    pub HAS_METALLICROUGHNESS_UV_TRANSFORM: bool,
    pub MATERIAL_SHEEN: bool,
    pub HAS_SHEEN_COLOR_MAP: bool,
    pub HAS_SHEENCOLOR_UV_TRANSFORM: bool,
    pub HAS_SHEEN_ROUGHNESS_MAP: bool,
    pub HAS_SHEENROUGHNESS_UV_TRANSFORM: bool,
    pub MATERIAL_SPECULARGLOSSINESS: bool,
    pub HAS_DIFFUSE_MAP: bool,
    pub HAS_DIFFUSE_UV_TRANSFORM: bool,
    pub HAS_SPECULAR_GLOSSINESS_MAP: bool,
    pub HAS_SPECULARGLOSSINESS_UV_TRANSFORM: bool,
    pub MATERIAL_CLEARCOAT: bool,
    pub HAS_CLEARCOAT_MAP: bool,
    pub HAS_CLEARCOAT_UV_TRANSFORM: bool,
    pub HAS_CLEARCOAT_ROUGHNESS_MAP: bool,
    pub HAS_CLEARCOATROUGHNESS_UV_TRANSFORM: bool,
    pub HAS_CLEARCOAT_NORMAL_MAP: bool,
    pub HAS_CLEARCOATNORMAL_UV_TRANSFORM: bool,
    pub MATERIAL_SPECULAR: bool,
    pub HAS_SPECULAR_MAP: bool,
    pub HAS_SPECULAR_UV_TRANSFORM: bool,
    pub HAS_SPECULAR_COLOR_MAP: bool,
    pub HAS_SPECULARCOLOR_UV_TRANSFORM: bool,
    pub MATERIAL_TRANSMISSION: bool,
    pub HAS_TRANSMISSION_MAP: bool,
    pub HAS_TRANSMISSION_UV_TRANSFORM: bool,
    pub MATERIAL_VOLUME: bool,
    pub HAS_THICKNESS_MAP: bool,
    pub HAS_THICKNESS_UV_TRANSFORM: bool,
    pub MATERIAL_IRIDESCENCE: bool,
    pub HAS_IRIDESCENCE_MAP: bool,
    pub HAS_IRIDESCENCE_UV_TRANSFORM: bool,
    pub HAS_IRIDESCENCE_THICKNESS_MAP: bool,
    pub HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM: bool,
    pub MATERIAL_DIFFUSE_TRANSMISSION: bool,
    pub HAS_DIFFUSE_TRANSMISSION_MAP: bool,
    pub HAS_DIFFUSETRANSMISSION_UV_TRANSFORM: bool,
    pub HAS_DIFFUSE_TRANSMISSION_COLOR_MAP: bool,
    pub HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM: bool,
    pub MATERIAL_ANISOTROPY: bool,
    pub HAS_ANISOTROPY_MAP: bool,
    pub HAS_ANISOTROPY_UV_TRANSFORM: bool,
    pub MATERIAL_IOR: bool,
    pub MATERIAL_DISPERSION: bool,
    pub MATERIAL_EMISSIVE_STRENGTH: bool,
}

#[derive(Eq, PartialEq, Debug)]
pub enum RenderType {
    Opaque,
    Translucent,
    Transmissive,
}

impl MaterialSpecializationConstants {
    pub const fn render_type(&self) -> RenderType {
        if self.MATERIAL_TRANSMISSION {
            RenderType::Transmissive
        } else if self.ALPHAMODE_BLEND {
            RenderType::Translucent
        } else {
            RenderType::Opaque
        }
    }
}

impl From<SpecializationConstants> for Vec<(u32, SpecializationConstant)> {
    fn from(constants: SpecializationConstants) -> Self {
        [
            constants.object_constants.HAS_NORMAL_VEC3,
            constants.object_constants.HAS_TANGENT_VEC4,
            constants.object_constants.HAS_TEXCOORD_0_VEC2,
            constants.object_constants.HAS_TEXCOORD_1_VEC2,
            constants.object_constants.HAS_COLOR_0_VEC3,
            constants.object_constants.HAS_COLOR_0_VEC4,
            constants.object_constants.NOT_TRIANGLE,
            constants.material_constants.ALPHAMODE_OPAQUE,
            constants.material_constants.ALPHAMODE_MASK,
            constants.material_constants.ALPHAMODE_BLEND,
            constants.material_constants.MATERIAL_UNLIT,
            constants.material_constants.HAS_NORMAL_MAP,
            constants.material_constants.HAS_NORMAL_UV_TRANSFORM,
            constants.material_constants.HAS_VERT_NORMAL_UV_TRANSFORM,
            constants.material_constants.HAS_EMISSIVE_MAP,
            constants.material_constants.HAS_EMISSIVE_UV_TRANSFORM,
            constants.material_constants.HAS_OCCLUSION_MAP,
            constants.material_constants.HAS_OCCLUSION_UV_TRANSFORM,
            constants.material_constants.HAS_BASE_COLOR_MAP,
            constants.material_constants.HAS_BASECOLOR_UV_TRANSFORM,
            constants.material_constants.MATERIAL_METALLICROUGHNESS,
            constants.material_constants.HAS_METALLIC_ROUGHNESS_MAP,
            constants
                .material_constants
                .HAS_METALLICROUGHNESS_UV_TRANSFORM,
            constants.material_constants.MATERIAL_SHEEN,
            constants.material_constants.HAS_SHEEN_COLOR_MAP,
            constants.material_constants.HAS_SHEENCOLOR_UV_TRANSFORM,
            constants.material_constants.HAS_SHEEN_ROUGHNESS_MAP,
            constants.material_constants.HAS_SHEENROUGHNESS_UV_TRANSFORM,
            constants.material_constants.MATERIAL_SPECULARGLOSSINESS,
            constants.material_constants.HAS_DIFFUSE_MAP,
            constants.material_constants.HAS_DIFFUSE_UV_TRANSFORM,
            constants.material_constants.HAS_SPECULAR_GLOSSINESS_MAP,
            constants
                .material_constants
                .HAS_SPECULARGLOSSINESS_UV_TRANSFORM,
            constants.material_constants.MATERIAL_CLEARCOAT,
            constants.material_constants.HAS_CLEARCOAT_MAP,
            constants.material_constants.HAS_CLEARCOAT_UV_TRANSFORM,
            constants.material_constants.HAS_CLEARCOAT_ROUGHNESS_MAP,
            constants
                .material_constants
                .HAS_CLEARCOATROUGHNESS_UV_TRANSFORM,
            constants.material_constants.HAS_CLEARCOAT_NORMAL_MAP,
            constants
                .material_constants
                .HAS_CLEARCOATNORMAL_UV_TRANSFORM,
            constants.material_constants.MATERIAL_SPECULAR,
            constants.material_constants.HAS_SPECULAR_MAP,
            constants.material_constants.HAS_SPECULAR_UV_TRANSFORM,
            constants.material_constants.HAS_SPECULAR_COLOR_MAP,
            constants.material_constants.HAS_SPECULARCOLOR_UV_TRANSFORM,
            constants.material_constants.MATERIAL_TRANSMISSION,
            constants.material_constants.HAS_TRANSMISSION_MAP,
            constants.material_constants.HAS_TRANSMISSION_UV_TRANSFORM,
            constants.material_constants.MATERIAL_VOLUME,
            constants.material_constants.HAS_THICKNESS_MAP,
            constants.material_constants.HAS_THICKNESS_UV_TRANSFORM,
            constants.material_constants.MATERIAL_IRIDESCENCE,
            constants.material_constants.HAS_IRIDESCENCE_MAP,
            constants.material_constants.HAS_IRIDESCENCE_UV_TRANSFORM,
            constants.material_constants.HAS_IRIDESCENCE_THICKNESS_MAP,
            constants
                .material_constants
                .HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM,
            constants.material_constants.MATERIAL_DIFFUSE_TRANSMISSION,
            constants.material_constants.HAS_DIFFUSE_TRANSMISSION_MAP,
            constants
                .material_constants
                .HAS_DIFFUSETRANSMISSION_UV_TRANSFORM,
            constants
                .material_constants
                .HAS_DIFFUSE_TRANSMISSION_COLOR_MAP,
            constants
                .material_constants
                .HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM,
            constants.material_constants.MATERIAL_ANISOTROPY,
            constants.material_constants.HAS_ANISOTROPY_MAP,
            constants.material_constants.HAS_ANISOTROPY_UV_TRANSFORM,
            constants.material_constants.MATERIAL_IOR,
            constants.material_constants.MATERIAL_DISPERSION,
            constants.material_constants.MATERIAL_EMISSIVE_STRENGTH,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.into()))
        .collect()
    }
}

impl std::fmt::Display for ObjectSpecializationConstants {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut sep = "";
        let delim = " ";
        if self.HAS_NORMAL_VEC3 {
            write!(f, "{sep}HAS_NORMAL_VEC3")?;
            sep = delim;
        }
        if self.HAS_TANGENT_VEC4 {
            write!(f, "{sep}HAS_TANGENT_VEC4")?;
            sep = delim;
        }
        if self.HAS_TEXCOORD_0_VEC2 {
            write!(f, "{sep}HAS_TEXCOORD_0_VEC2")?;
            sep = delim;
        }
        if self.HAS_TEXCOORD_1_VEC2 {
            write!(f, "{sep}HAS_TEXCOORD_1_VEC2")?;
            sep = delim;
        }
        if self.HAS_COLOR_0_VEC3 {
            write!(f, "{sep}HAS_COLOR_0_VEC3")?;
            sep = delim;
        }
        if self.HAS_COLOR_0_VEC4 {
            write!(f, "{sep}HAS_COLOR_0_VEC4")?;
            sep = delim;
        }
        if self.NOT_TRIANGLE {
            write!(f, "{sep}NOT_TRIANGLE")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for MaterialSpecializationConstants {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut sep = "";
        let delim = " ";
        if self.ALPHAMODE_OPAQUE {
            write!(f, "{sep}ALPHAMODE_OPAQUE")?;
            sep = delim;
        }
        if self.ALPHAMODE_MASK {
            write!(f, "{sep}ALPHAMODE_MASK")?;
            sep = delim;
        }
        if self.ALPHAMODE_BLEND {
            write!(f, "{sep}ALPHAMODE_BLEND")?;
            sep = delim;
        }
        if self.MATERIAL_UNLIT {
            write!(f, "{sep}MATERIAL_UNLIT")?;
            sep = delim;
        }
        if self.HAS_NORMAL_MAP {
            write!(f, "{sep}HAS_NORMAL_MAP")?;
            sep = delim;
        }
        if self.HAS_NORMAL_UV_TRANSFORM {
            write!(f, "{sep}HAS_NORMAL_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_VERT_NORMAL_UV_TRANSFORM {
            write!(f, "{sep}HAS_VERT_NORMAL_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_EMISSIVE_MAP {
            write!(f, "{sep}HAS_EMISSIVE_MAP")?;
            sep = delim;
        }
        if self.HAS_EMISSIVE_UV_TRANSFORM {
            write!(f, "{sep}HAS_EMISSIVE_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_OCCLUSION_MAP {
            write!(f, "{sep}HAS_OCCLUSION_MAP")?;
            sep = delim;
        }
        if self.HAS_OCCLUSION_UV_TRANSFORM {
            write!(f, "{sep}HAS_OCCLUSION_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_BASE_COLOR_MAP {
            write!(f, "{sep}HAS_BASE_COLOR_MAP")?;
            sep = delim;
        }
        if self.HAS_BASECOLOR_UV_TRANSFORM {
            write!(f, "{sep}HAS_BASECOLOR_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_METALLICROUGHNESS {
            write!(f, "{sep}MATERIAL_METALLICROUGHNESS")?;
            sep = delim;
        }
        if self.HAS_METALLIC_ROUGHNESS_MAP {
            write!(f, "{sep}HAS_METALLIC_ROUGHNESS_MAP")?;
            sep = delim;
        }
        if self.HAS_METALLICROUGHNESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_METALLICROUGHNESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_SHEEN {
            write!(f, "{sep}MATERIAL_SHEEN")?;
            sep = delim;
        }
        if self.HAS_SHEEN_COLOR_MAP {
            write!(f, "{sep}HAS_SHEEN_COLOR_MAP")?;
            sep = delim;
        }
        if self.HAS_SHEENCOLOR_UV_TRANSFORM {
            write!(f, "{sep}HAS_SHEENCOLOR_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_SHEEN_ROUGHNESS_MAP {
            write!(f, "{sep}HAS_SHEEN_ROUGHNESS_MAP")?;
            sep = delim;
        }
        if self.HAS_SHEENROUGHNESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_SHEENROUGHNESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_SPECULARGLOSSINESS {
            write!(f, "{sep}MATERIAL_SPECULARGLOSSINESS")?;
            sep = delim;
        }
        if self.HAS_DIFFUSE_MAP {
            write!(f, "{sep}HAS_DIFFUSE_MAP")?;
            sep = delim;
        }
        if self.HAS_DIFFUSE_UV_TRANSFORM {
            write!(f, "{sep}HAS_DIFFUSE_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_SPECULAR_GLOSSINESS_MAP {
            write!(f, "{sep}HAS_SPECULAR_GLOSSINESS_MAP")?;
            sep = delim;
        }
        if self.HAS_SPECULARGLOSSINESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_SPECULARGLOSSINESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_CLEARCOAT {
            write!(f, "{sep}MATERIAL_CLEARCOAT")?;
            sep = delim;
        }
        if self.HAS_CLEARCOAT_MAP {
            write!(f, "{sep}HAS_CLEARCOAT_MAP")?;
            sep = delim;
        }
        if self.HAS_CLEARCOAT_UV_TRANSFORM {
            write!(f, "{sep}HAS_CLEARCOAT_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_CLEARCOAT_ROUGHNESS_MAP {
            write!(f, "{sep}HAS_CLEARCOAT_ROUGHNESS_MAP")?;
            sep = delim;
        }
        if self.HAS_CLEARCOATROUGHNESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_CLEARCOATROUGHNESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_CLEARCOAT_NORMAL_MAP {
            write!(f, "{sep}HAS_CLEARCOAT_NORMAL_MAP")?;
            sep = delim;
        }
        if self.HAS_CLEARCOATNORMAL_UV_TRANSFORM {
            write!(f, "{sep}HAS_CLEARCOATNORMAL_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_SPECULAR {
            write!(f, "{sep}MATERIAL_SPECULAR")?;
            sep = delim;
        }
        if self.HAS_SPECULAR_MAP {
            write!(f, "{sep}HAS_SPECULAR_MAP")?;
            sep = delim;
        }
        if self.HAS_SPECULAR_UV_TRANSFORM {
            write!(f, "{sep}HAS_SPECULAR_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_SPECULAR_COLOR_MAP {
            write!(f, "{sep}HAS_SPECULAR_COLOR_MAP")?;
            sep = delim;
        }
        if self.HAS_SPECULARCOLOR_UV_TRANSFORM {
            write!(f, "{sep}HAS_SPECULARCOLOR_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_TRANSMISSION {
            write!(f, "{sep}MATERIAL_TRANSMISSION")?;
            sep = delim;
        }
        if self.HAS_TRANSMISSION_MAP {
            write!(f, "{sep}HAS_TRANSMISSION_MAP")?;
            sep = delim;
        }
        if self.HAS_TRANSMISSION_UV_TRANSFORM {
            write!(f, "{sep}HAS_TRANSMISSION_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_VOLUME {
            write!(f, "{sep}MATERIAL_VOLUME")?;
            sep = delim;
        }
        if self.HAS_THICKNESS_MAP {
            write!(f, "{sep}HAS_THICKNESS_MAP")?;
            sep = delim;
        }
        if self.HAS_THICKNESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_THICKNESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_IRIDESCENCE {
            write!(f, "{sep}MATERIAL_IRIDESCENCE")?;
            sep = delim;
        }
        if self.HAS_IRIDESCENCE_MAP {
            write!(f, "{sep}HAS_IRIDESCENCE_MAP")?;
            sep = delim;
        }
        if self.HAS_IRIDESCENCE_UV_TRANSFORM {
            write!(f, "{sep}HAS_IRIDESCENCE_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_IRIDESCENCE_THICKNESS_MAP {
            write!(f, "{sep}HAS_IRIDESCENCE_THICKNESS_MAP")?;
            sep = delim;
        }
        if self.HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM {
            write!(f, "{sep}HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_DIFFUSE_TRANSMISSION {
            write!(f, "{sep}MATERIAL_DIFFUSE_TRANSMISSION")?;
            sep = delim;
        }
        if self.HAS_DIFFUSE_TRANSMISSION_MAP {
            write!(f, "{sep}HAS_DIFFUSE_TRANSMISSION_MAP")?;
            sep = delim;
        }
        if self.HAS_DIFFUSETRANSMISSION_UV_TRANSFORM {
            write!(f, "{sep}HAS_DIFFUSETRANSMISSION_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.HAS_DIFFUSE_TRANSMISSION_COLOR_MAP {
            write!(f, "{sep}HAS_DIFFUSE_TRANSMISSION_COLOR_MAP")?;
            sep = delim;
        }
        if self.HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM {
            write!(f, "{sep}HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_ANISOTROPY {
            write!(f, "{sep}MATERIAL_ANISOTROPY")?;
            sep = delim;
        }
        if self.HAS_ANISOTROPY_MAP {
            write!(f, "{sep}HAS_ANISOTROPY_MAP")?;
            sep = delim;
        }
        if self.HAS_ANISOTROPY_UV_TRANSFORM {
            write!(f, "{sep}HAS_ANISOTROPY_UV_TRANSFORM")?;
            sep = delim;
        }
        if self.MATERIAL_IOR {
            write!(f, "{sep}MATERIAL_IOR")?;
            sep = delim;
        }
        if self.MATERIAL_DISPERSION {
            write!(f, "{sep}MATERIAL_DISPERSION")?;
            sep = delim;
        }
        if self.MATERIAL_EMISSIVE_STRENGTH {
            write!(f, "{sep}MATERIAL_EMISSIVE_STRENGTH")?;
        }
        Ok(())
    }
}

macro_rules! some {
    ($base:ident $( $rest:tt )*) => {{
        #[allow(clippy::redundant_closure_call)]
        (|| some_chain!($base $( $rest )*).as_ref() )().is_some()
    }};
}

macro_rules! some_chain {
    // When an optional field access is encountered: “? . field”
    ($acc:tt ? . $field:ident $( $rest:tt )*) => {
        some_chain!(( $acc.as_ref()? . $field ) $( $rest )*)
    };
    // When a normal field access is encountered: “. field”
    ($acc:tt . $field:ident $( $rest:tt )*) => {
        some_chain!(( $acc.$field ) $( $rest )*)
    };
    // Base case: no more tokens; simply return the accumulator.
    ($e:tt) => { $e };
}

impl From<&Material> for MaterialSpecializationConstants {
    #[allow(non_snake_case)]
    fn from(mat: &Material) -> Self {
        let MATERIAL_SPECULARGLOSSINESS = some!(mat.pbr_specular_glossiness);
        let MATERIAL_SPECULAR = some!(mat.specular);
        let MATERIAL_METALLICROUGHNESS = !MATERIAL_SPECULARGLOSSINESS;

        Self {
            DOUBLE_SIDED: mat.double_sided,
            ALPHAMODE_OPAQUE: mat.alpha_mode == AlphaMode::Opaque,
            ALPHAMODE_MASK: mat.alpha_mode == AlphaMode::Mask,
            ALPHAMODE_BLEND: mat.alpha_mode == AlphaMode::Blend,
            MATERIAL_UNLIT: mat.unlit,
            HAS_NORMAL_MAP: some!(mat.normal_texture),
            HAS_NORMAL_UV_TRANSFORM: some!(mat.normal_texture?.info.transform),
            HAS_VERT_NORMAL_UV_TRANSFORM: some!(mat.normal_texture?.info.transform),
            HAS_EMISSIVE_MAP: some!(mat.emissive_texture),
            HAS_EMISSIVE_UV_TRANSFORM: some!(mat.emissive_texture?.transform),
            HAS_OCCLUSION_MAP: some!(mat.occlusion_texture),
            HAS_OCCLUSION_UV_TRANSFORM: some!(mat.occlusion_texture?.info.transform),
            HAS_BASE_COLOR_MAP: some!(mat.pbr_metallic_roughness.base_color_texture),
            HAS_BASECOLOR_UV_TRANSFORM: some!(
                mat.pbr_metallic_roughness.base_color_texture?.transform
            ),
            MATERIAL_METALLICROUGHNESS,
            HAS_METALLIC_ROUGHNESS_MAP: some!(
                mat.pbr_metallic_roughness.metallic_roughness_texture
            ),
            HAS_METALLICROUGHNESS_UV_TRANSFORM: some!(
                mat.pbr_metallic_roughness
                    .metallic_roughness_texture?
                    .transform
            ),
            MATERIAL_SHEEN: some!(mat.sheen),
            HAS_SHEEN_COLOR_MAP: some!(mat.sheen?.color_texture),
            HAS_SHEENCOLOR_UV_TRANSFORM: some!(mat.sheen?.color_texture?.transform),
            HAS_SHEEN_ROUGHNESS_MAP: some!(mat.sheen?.roughness_texture),
            HAS_SHEENROUGHNESS_UV_TRANSFORM: some!(mat.sheen?.roughness_texture?.transform),
            MATERIAL_SPECULARGLOSSINESS,
            HAS_DIFFUSE_MAP: some!(mat.pbr_specular_glossiness?.diffuse_texture),
            HAS_DIFFUSE_UV_TRANSFORM: some!(
                mat.pbr_specular_glossiness?.diffuse_texture?.transform
            ),
            HAS_SPECULAR_GLOSSINESS_MAP: some!(
                mat.pbr_specular_glossiness?.specular_glossiness_texture
            ),
            HAS_SPECULARGLOSSINESS_UV_TRANSFORM: some!(
                mat.pbr_specular_glossiness?
                    .specular_glossiness_texture?
                    .transform
            ),
            MATERIAL_CLEARCOAT: some!(mat.clearcoat),
            HAS_CLEARCOAT_MAP: some!(mat.clearcoat?.texture),
            HAS_CLEARCOAT_UV_TRANSFORM: some!(mat.clearcoat?.texture?.transform),
            HAS_CLEARCOAT_ROUGHNESS_MAP: some!(mat.clearcoat?.roughness_texture),
            HAS_CLEARCOATROUGHNESS_UV_TRANSFORM: some!(mat.clearcoat?.roughness_texture?.transform),
            HAS_CLEARCOAT_NORMAL_MAP: some!(mat.clearcoat?.normal_texture),
            HAS_CLEARCOATNORMAL_UV_TRANSFORM: some!(mat.clearcoat?.normal_texture?.info.transform),
            MATERIAL_SPECULAR,
            HAS_SPECULAR_MAP: some!(mat.specular?.texture),
            HAS_SPECULAR_UV_TRANSFORM: some!(mat.specular?.texture?.transform),
            HAS_SPECULAR_COLOR_MAP: some!(mat.specular?.color_texture),
            HAS_SPECULARCOLOR_UV_TRANSFORM: some!(mat.specular?.color_texture?.transform),
            MATERIAL_TRANSMISSION: some!(mat.transmission),
            HAS_TRANSMISSION_MAP: some!(mat.transmission?.texture),
            HAS_TRANSMISSION_UV_TRANSFORM: some!(mat.transmission?.texture?.transform),
            MATERIAL_VOLUME: some!(mat.volume),
            HAS_THICKNESS_MAP: some!(mat.volume?.thickness_texture),
            HAS_THICKNESS_UV_TRANSFORM: some!(mat.volume?.thickness_texture?.transform),
            MATERIAL_IRIDESCENCE: some!(mat.iridescence),
            HAS_IRIDESCENCE_MAP: some!(mat.iridescence?.texture),
            HAS_IRIDESCENCE_UV_TRANSFORM: some!(mat.iridescence?.texture?.transform),
            HAS_IRIDESCENCE_THICKNESS_MAP: some!(mat.iridescence?.thickness_texture),
            HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM: some!(
                mat.iridescence?.thickness_texture?.transform
            ),
            MATERIAL_DIFFUSE_TRANSMISSION: some!(mat.diffuse_transmission),
            HAS_DIFFUSE_TRANSMISSION_MAP: some!(mat.diffuse_transmission?.texture),
            HAS_DIFFUSETRANSMISSION_UV_TRANSFORM: some!(
                mat.diffuse_transmission?.texture?.transform
            ),
            HAS_DIFFUSE_TRANSMISSION_COLOR_MAP: some!(mat.diffuse_transmission?.color_texture),
            HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM: some!(
                mat.diffuse_transmission?.color_texture?.transform
            ),
            MATERIAL_ANISOTROPY: some!(mat.anisotropy),
            HAS_ANISOTROPY_MAP: some!(mat.anisotropy?.texture),
            HAS_ANISOTROPY_UV_TRANSFORM: some!(mat.anisotropy?.texture?.transform),
            MATERIAL_IOR: some!(mat.ior),
            MATERIAL_DISPERSION: some!(mat.dispersion),
            MATERIAL_EMISSIVE_STRENGTH: some!(mat.emissive_strength),
        }
    }
}
