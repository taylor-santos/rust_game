use cgmath::{Vector3, Vector4};
use gltf::image::{Data, Format};
use gltf::Error;
use std::time::Instant;

type TextureID = usize;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Material {
    pub name: Option<String>,

    pub base_color_factor: Vector4<f32>,
    pub base_color_texture: Option<TextureID>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<TextureID>,

    pub normal_texture: Option<TextureID>,
    pub normal_scale: Option<f32>,

    pub occlusion_texture: Option<TextureID>,
    pub occlusion_strength: Option<f32>,

    pub emissive_factor: Vector3<f32>,
    pub emissive_texture: Option<TextureID>,

    pub alpha_cutoff: Option<f32>,
    pub alpha_mode: gltf::material::AlphaMode,

    pub double_sided: bool,
}

pub struct Mesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<u32>,
    pub mat_idx: usize,
}

type Texture = Data;
pub(crate) type TextureFormat = Format;

pub fn load_gltf(path: &str) -> Result<(Vec<Mesh>, Vec<Texture>, Vec<Material>), Error> {
    let mut start_time = Instant::now();
    let (doc, buffers, textures) = gltf::import(path)?;
    println!("Loaded gltf in {:?}", start_time.elapsed());
    start_time = Instant::now();

    let materials = doc
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            Material {
                name: mat.name().map(Into::into),

                base_color_factor: pbr.base_color_factor().into(),
                base_color_texture: pbr
                    .base_color_texture()
                    .map(|tex| tex.texture().source().index()),
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                metallic_roughness_texture: pbr
                    .metallic_roughness_texture()
                    .map(|tex| tex.texture().source().index()),

                normal_texture: mat
                    .normal_texture()
                    .map(|tex| tex.texture().source().index()),
                normal_scale: mat.normal_texture().map(|tex| tex.scale()),

                occlusion_texture: mat
                    .occlusion_texture()
                    .map(|tex| tex.texture().source().index()),
                occlusion_strength: mat.occlusion_texture().map(|tex| tex.strength()),

                emissive_factor: mat.emissive_factor().into(),
                emissive_texture: mat
                    .emissive_texture()
                    .map(|tex| tex.texture().source().index()),

                alpha_cutoff: mat.alpha_cutoff(),
                alpha_mode: mat.alpha_mode(),

                double_sided: mat.double_sided(),
            }
        })
        .collect::<Vec<_>>();
    println!(
        "Loaded {} materials in {:?}",
        materials.len(),
        start_time.elapsed()
    );
    start_time = Instant::now();

    let meshes = doc
        .meshes()
        .map(|mesh| mesh.primitives())
        .map(|prims| {
            prims.map(|prim| {
                let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader
                    .read_positions()
                    .unwrap()
                    .flatten()
                    .collect::<Vec<_>>();
                let normals = reader.read_normals().unwrap().flatten().collect::<Vec<_>>();
                let texcoords = reader
                    .read_tex_coords(0)
                    .unwrap()
                    .into_f32()
                    .flatten()
                    .collect::<Vec<_>>();
                let indices = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .collect::<Vec<_>>();
                let mat_idx = prim.material().index().unwrap();

                Mesh {
                    positions,
                    normals,
                    texcoords,
                    indices,
                    mat_idx,
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    println!(
        "Loaded {} meshes in {:?}",
        meshes.len(),
        start_time.elapsed()
    );

    Ok((meshes, textures, materials))
}
