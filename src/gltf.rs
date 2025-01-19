use crate::material::{Material, NormalTexture, PbrMetallicRoughness, Texture};
use gltf;
use gltf::image::{Data, Format};
use gltf::texture::Info;
use gltf::Error;
use std::time::Instant;

pub struct Mesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<u32>,
    pub mat_idx: usize,
}

impl From<gltf::Material<'_>> for Material {
    fn from(mat: gltf::Material<'_>) -> Material {
        let pbr_metallic_roughness = mat.pbr_metallic_roughness();
        let pbr_metallic_roughness_data = PbrMetallicRoughness {
            base_color_factor: pbr_metallic_roughness.base_color_factor(),
            metallic_factor: pbr_metallic_roughness.metallic_factor(),
            roughness_factor: pbr_metallic_roughness.roughness_factor(),
            metallic_roughness_texture: pbr_metallic_roughness
                .metallic_roughness_texture()
                .map(|info| info.into()),
        };

        Material {
            name: mat.name().map(String::from),
            base_color_texture: pbr_metallic_roughness.base_color_texture().map(Into::into),
            normal_texture: mat.normal_texture().map(Into::into),
            pbr_metallic_roughness: pbr_metallic_roughness_data,
            pbr_specular_glossiness: None, // Handle this when the extension is supported
            unlit: mat.unlit(),
            texture_transform: None, // Handle this when the extension is supported
            variants: vec![],        // Handle this when the extension is supported
            volume: None,            // Handle this when the extension is supported
            specular: None,          // Handle this when the extension is supported
            transmission: None,      // Handle this when the extension is supported
            ior: mat.ior(),
            emissive_strength: mat.emissive_strength(),
            emissive_texture: mat.emissive_texture().map(|info| info.into()),
            emissive_factor: mat.emissive_factor(),
        }
    }
}

impl From<Info<'_>> for Texture {
    fn from(info: Info<'_>) -> Self {
        Self {
            index: info.texture().source().index(),
            tex_coord: info.tex_coord(),
        }
    }
}

impl From<gltf::material::NormalTexture<'_>> for NormalTexture {
    fn from(normal_texture: gltf::material::NormalTexture<'_>) -> Self {
        Self {
            texture: Texture {
                index: normal_texture.texture().source().index(),
                tex_coord: normal_texture.tex_coord(),
            },
            scale: normal_texture.scale(),
        }
    }
}

pub(crate) type TextureData = Data;
pub(crate) type TextureFormat = Format;

pub fn load_gltf(path: &str) -> Result<(Vec<Mesh>, Vec<TextureData>, Vec<Material>), Error> {
    let mut start_time = Instant::now();
    let (doc, buffers, textures) = gltf::import(path)?;
    println!("Loaded gltf in {:?}", start_time.elapsed());
    start_time = Instant::now();

    let materials = doc.materials().map(Into::into).collect::<Vec<Material>>();
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
                    .read_tex_coords(0) // TODO: support multiple TEXCOORDs
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
