use gltf::material::AlphaMode;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Material {
    pub name: Option<String>,
    pub alpha_cutoff: Option<f32>,
    pub alpha_mode: AlphaMode,
    pub base_color_texture: Option<Texture>,
    pub normal_texture: Option<NormalTexture>,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub pbr_specular_glossiness: Option<PbrSpecularGlossiness>,
    pub unlit: bool,
    pub texture_transform: Option<TextureTransform>,
    pub variants: Vec<String>,
    pub volume: Option<Volume>,
    pub specular: Option<Specular>,
    pub transmission: Option<Transmission>,
    pub ior: Option<f32>,
    pub emissive_strength: Option<f32>,
    pub emissive_texture: Option<Texture>,
    pub emissive_factor: [f32; 3],
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Texture {
    pub index: usize,
    pub tex_coord: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NormalTexture {
    pub texture: Texture,
    pub scale: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PbrMetallicRoughness {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Texture>,
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
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord: Option<u32>,
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
