use cgmath::num_traits::Float;
use cgmath::{Matrix3, Rad};
use vulkano::padded::Padded;

#[derive(Debug)]
pub struct Material {
    pub name: Option<String>,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub normal_texture: Option<NormalTexture>,
    pub occlusion_texture: Option<OcclusionTexture>,
    pub emissive_texture: Option<TextureInfo>,
    pub emissive_factor: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: AlphaCutoff,
    pub double_sided: bool,
    // Extensions
    pub pbr_specular_glossiness: Option<PbrSpecularGlossiness>,
    pub anisotropy: Option<Anisotropy>,
    pub clearcoat: Option<Clearcoat>,
    pub diffuse_transmission: Option<DiffuseTransmission>,
    pub dispersion: Option<Dispersion>,
    pub emissive_strength: Option<EmissiveStrength>,
    pub ior: Option<Ior>,
    pub iridescence: Option<Iridescence>,
    pub sheen: Option<Sheen>,
    pub specular: Option<Specular>,
    pub transmission: Option<Transmission>,
    pub unlit: bool,
    pub volume: Option<Volume>,
}

impl Default for Material {
    fn default() -> Material {
        Self {
            name: None,
            pbr_metallic_roughness: Default::default(),
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            emissive_factor: [0.0; 3],
            alpha_mode: Default::default(),
            alpha_cutoff: Default::default(),
            double_sided: false,
            pbr_specular_glossiness: None,
            anisotropy: None,
            clearcoat: None,
            diffuse_transmission: None,
            dispersion: None,
            emissive_strength: None,
            ior: None,
            iridescence: None,
            sheen: None,
            specular: None,
            transmission: None,
            unlit: false,
            volume: None,
        }
    }
}

#[derive(Debug)]
pub struct PbrMetallicRoughness {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<TextureInfo>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<TextureInfo>,
}

impl Default for PbrMetallicRoughness {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0; 4],
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Debug)]
pub struct PbrSpecularGlossiness {
    pub diffuse_factor: [f32; 4],
    pub diffuse_texture: Option<TextureInfo>,
    pub specular_factor: [f32; 3],
    pub glossiness_factor: f32,
    pub specular_glossiness_texture: Option<TextureInfo>,
}

impl Default for PbrSpecularGlossiness {
    fn default() -> Self {
        Self {
            diffuse_factor: [1.0; 4],
            diffuse_texture: None,
            specular_factor: [1.0; 3],
            glossiness_factor: 1.0,
            specular_glossiness_texture: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Anisotropy {
    pub strength: f32,
    pub rotation: f32,
    pub texture: Option<TextureInfo>,
}

impl Default for Anisotropy {
    fn default() -> Self {
        Self {
            strength: 0.0,
            rotation: 0.0,
            texture: None,
        }
    }
}

impl From<Anisotropy> for [f32; 3] {
    fn from(value: Anisotropy) -> Self {
        [value.rotation.cos(), value.rotation.sin(), value.strength]
    }
}

#[derive(Debug)]
pub struct Clearcoat {
    pub factor: f32,
    pub texture: Option<TextureInfo>,
    pub roughness_factor: f32,
    pub roughness_texture: Option<TextureInfo>,
    pub normal_texture: Option<NormalTexture>,
}

impl Default for Clearcoat {
    fn default() -> Self {
        Self {
            factor: 0.0,
            texture: None,
            roughness_factor: 0.0,
            roughness_texture: None,
            normal_texture: None,
        }
    }
}

#[derive(Debug)]
pub struct DiffuseTransmission {
    pub factor: f32,
    pub texture: Option<TextureInfo>,
    pub color_factor: [f32; 3],
    pub color_texture: Option<TextureInfo>,
}

impl Default for DiffuseTransmission {
    fn default() -> Self {
        Self {
            factor: 0.0,
            texture: None,
            color_factor: [1.0; 3],
            color_texture: None,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct AlphaCutoff(f32);

impl Default for AlphaCutoff {
    fn default() -> Self {
        Self(0.5)
    }
}

impl From<f32> for AlphaCutoff {
    fn from(v: f32) -> Self {
        Self(v)
    }
}

impl From<AlphaCutoff> for f32 {
    fn from(value: AlphaCutoff) -> Self {
        value.0
    }
}

impl<const N: usize> From<AlphaCutoff> for Padded<f32, N> {
    fn from(a: AlphaCutoff) -> Self {
        a.0.into()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dispersion(pub f32);

impl Default for Dispersion {
    fn default() -> Self {
        Self(0.0)
    }
}

impl From<f32> for Dispersion {
    fn from(v: f32) -> Self {
        Self(v)
    }
}

impl From<Dispersion> for f32 {
    fn from(value: Dispersion) -> Self {
        value.0
    }
}

impl<const N: usize> From<Dispersion> for Padded<f32, N> {
    fn from(value: Dispersion) -> Self {
        value.0.into()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmissiveStrength(f32);

impl Default for EmissiveStrength {
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<f32> for EmissiveStrength {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<EmissiveStrength> for f32 {
    fn from(value: EmissiveStrength) -> Self {
        value.0
    }
}

impl<const N: usize> From<EmissiveStrength> for Padded<f32, N> {
    fn from(e: EmissiveStrength) -> Self {
        e.0.into()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ior(f32);

impl Default for Ior {
    fn default() -> Self {
        Self(1.5)
    }
}

impl From<f32> for Ior {
    fn from(v: f32) -> Self {
        Self(v)
    }
}

impl From<Ior> for f32 {
    fn from(value: Ior) -> Self {
        value.0
    }
}

impl<const N: usize> From<Ior> for Padded<f32, N> {
    fn from(ior: Ior) -> Self {
        ior.0.into()
    }
}

#[derive(Debug)]
pub struct Iridescence {
    pub factor: f32,
    pub texture: Option<TextureInfo>,
    pub ior: f32,
    pub thickness_minimum: f32,
    pub thickness_maximum: f32,
    pub thickness_texture: Option<TextureInfo>,
}

impl Default for Iridescence {
    fn default() -> Self {
        Self {
            factor: 0.0,
            texture: None,
            ior: 1.3,
            thickness_minimum: 100.0,
            thickness_maximum: 400.0,
            thickness_texture: None,
        }
    }
}

#[derive(Debug)]
pub struct Sheen {
    pub color_factor: [f32; 3],
    pub color_texture: Option<TextureInfo>,
    pub roughness_factor: f32,
    pub roughness_texture: Option<TextureInfo>,
}

impl Default for Sheen {
    fn default() -> Self {
        Self {
            color_factor: [0.0; 3],
            color_texture: None,
            roughness_factor: 0.0,
            roughness_texture: None,
        }
    }
}

#[derive(Debug)]
pub struct Specular {
    pub factor: f32,
    pub texture: Option<TextureInfo>,
    pub color_factor: [f32; 3],
    pub color_texture: Option<TextureInfo>,
}

impl Default for Specular {
    fn default() -> Self {
        Self {
            factor: 1.0,
            texture: None,
            color_factor: [1.0; 3],
            color_texture: None,
        }
    }
}

#[derive(Debug)]
pub struct Transmission {
    pub factor: f32,
    pub texture: Option<TextureInfo>,
}

impl Default for Transmission {
    fn default() -> Self {
        Self {
            factor: 0.0,
            texture: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Volume {
    pub thickness_factor: f32,
    pub thickness_texture: Option<TextureInfo>,
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            thickness_factor: 0.0,
            thickness_texture: None,
            attenuation_distance: f32::infinity(),
            attenuation_color: [1.0; 3],
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord: Option<u32>,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
            tex_coord: None,
        }
    }
}

impl From<TextureTransform> for Matrix3<f32> {
    fn from(value: TextureTransform) -> Self {
        let translation = Matrix3::from_translation(value.offset.into());
        let rotation = Matrix3::from_angle_z(Rad(-value.rotation));
        let scale = Matrix3::from_nonuniform_scale(value.scale[0], value.scale[1]);

        translation * rotation * scale
    }
}

impl From<TextureTransform> for [vulkano::padded::Padded<[f32; 3], 4usize>; 3] {
    fn from(value: TextureTransform) -> Self {
        let mat: Matrix3<_> = value.into();
        [
            Padded(mat.x.into()),
            Padded(mat.y.into()),
            Padded(mat.z.into()),
        ]
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct TextureInfo {
    pub texture: Texture,
    pub transform: Option<TextureTransform>,
}

impl TextureInfo {
    pub fn tex_coord(&self) -> i32 {
        self.transform
            .as_ref()
            .and_then(|t| t.tex_coord)
            .unwrap_or(self.texture.tex_coord) as i32
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Texture {
    pub index: usize,
    pub tex_coord: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NormalTexture {
    pub info: TextureInfo,
    pub scale: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OcclusionTexture {
    pub info: TextureInfo,
    pub strength: f32,
}
