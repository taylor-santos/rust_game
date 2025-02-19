use cgmath::{Matrix, Matrix3, Rad, SquareMatrix};
use gltf::material::AlphaMode;
use vulkano::padded::Padded;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Material {
    pub name: Option<String>,
    pub alpha_cutoff: Option<f32>,
    pub alpha_mode: AlphaMode,
    pub base_color_texture: Option<TextureInfo>,
    pub normal_texture: Option<ScaledTexture>,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub pbr_specular_glossiness: Option<PbrSpecularGlossiness>,
    pub unlit: bool,
    pub variants: Vec<String>,
    pub volume: Option<Volume>,
    pub specular: Option<Specular>,
    pub transmission: Option<Transmission>,
    pub ior: Option<f32>,
    pub emissive_strength: Option<f32>,
    pub emissive_texture: Option<TextureInfo>,
    pub emissive_factor: [f32; 3],
    pub occlusion_texture: Option<ScaledTexture>,
    pub sheen_roughness_factor: Option<f32>,
    pub sheen_color_factor: Option<[f32; 3]>,
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct TextureTransform {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub rotation: f32,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            rotation: 0.0,
        }
    }
}

impl Into<Matrix3<f32>> for TextureTransform {
    fn into(self) -> Matrix3<f32> {
        let translation = Matrix3::from_translation(self.offset.into());
        let rotation = Matrix3::from_angle_z(Rad(self.rotation));
        let scale = Matrix3::from_nonuniform_scale(self.scale[0], self.scale[1]);
        let mat = translation * rotation * scale;
        // mat.transpose()
        mat
    }
}

impl Into<[vulkano::padded::Padded<[f32; 3], 4usize>; 3]> for TextureTransform {
    fn into(self) -> [Padded<[f32; 3], 4>; 3] {
        let mat: Matrix3<_> = self.into();
        [
            Padded(mat.x.into()),
            Padded(mat.y.into()),
            Padded(mat.z.into()),
        ]
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TextureInfo {
    pub texture: Texture,
    pub transform: Option<TextureTransform>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Texture {
    pub index: usize,
    pub tex_coord: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ScaledTexture {
    pub info: TextureInfo,
    pub scale: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PbrMetallicRoughness {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<TextureInfo>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PbrSpecularGlossiness {
    pub diffuse_factor: [f32; 4],
    pub specular_factor: [f32; 3],
    pub glossiness_factor: f32,
    pub diffuse_texture: Option<Texture>,
    pub specular_glossiness_texture: Option<Texture>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Volume {
    pub thickness_factor: f32,
    pub thickness_texture: Option<Texture>,
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Specular {
    pub specular_factor: f32,
    pub specular_texture: Option<Texture>,
    pub specular_color_factor: [f32; 3],
    pub specular_color_texture: Option<Texture>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Transmission {
    pub transmission_factor: f32,
    pub transmission_texture: Option<Texture>,
}
