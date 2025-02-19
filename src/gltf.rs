use crate::material::{
    Material, PbrMetallicRoughness, ScaledTexture, Texture, TextureInfo, TextureTransform,
};
use cgmath::{Matrix3, Matrix4, Point2, SquareMatrix, Vector2};
use gltf::image::{Data, Format};
use gltf::json::Value;
use gltf::scene::Transform;
use gltf::texture::Info;
use gltf::{texture, Error};
use rayon::prelude::*;
use std::iter::Map;
use std::time::Instant;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct CombinedVertex {
    #[format(R32G32B32_SFLOAT)]
    a_position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    a_normal: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    a_tangent: [f32; 4],
    #[format(R32G32_SFLOAT)]
    a_texcoord_0: [f32; 2],
    #[format(R32G32_SFLOAT)]
    a_texcoord_1: [f32; 2],
}

pub struct Primitive {
    pub vertices: Vec<CombinedVertex>,
    pub indices: Vec<u32>,
    pub mat_idx: usize,
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

pub struct Object {
    pub transform: Matrix4<f32>,
    pub mesh_idx: usize,
}

pub(crate) type TextureData = Data;
pub(crate) type TextureFormat = Format;

pub struct Gltf {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<TextureData>,
    pub materials: Vec<Material>,
    pub objects: Vec<Object>,
}

impl mikktspace::Geometry for Primitive {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let tri = self.indices[face * 3 + vert] as usize;
        self.vertices[tri].a_position
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let tri = self.indices[face * 3 + vert] as usize;
        self.vertices[tri].a_normal
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let tri = self.indices[face * 3 + vert] as usize;
        self.vertices[tri].a_texcoord_0
    }

    fn set_tangent_encoded(&mut self, mut _tangent: [f32; 4], _face: usize, _vert: usize) {
        let tri = self.indices[_face * 3 + _vert] as usize;
        _tangent[3] = -_tangent[3];
        self.vertices[tri].a_tangent = _tangent;
    }
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

        let alpha_cutoff = mat.alpha_cutoff();
        let alpha_mode = mat.alpha_mode();

        let mut sheen_roughness_factor = None;
        let mut sheen_color_factor = None;

        mat.extensions()
            .into_iter()
            .flatten()
            .for_each(|(key, value)| match key.as_str() {
                "KHR_materials_sheen" => {
                    use gltf::json::Value;
                    if let Value::Object(items) = value {
                        for (key, value) in items {
                            match (key.as_str(), value) {
                                ("sheenRoughnessFactor", Value::Number(n)) => {
                                    sheen_roughness_factor = Some(n.as_f64().unwrap() as f32);
                                }
                                ("sheenColorFactor", v) => {
                                    sheen_color_factor = Some(unwrap_vec(v).unwrap())
                                }
                                _ => panic!("Unrecognized key {}: {:?}", key, value),
                            }
                        }
                    } else {
                        panic!("Unrecognized value type for {}: {:?}", key, value);
                    }
                }
                _ => panic!("Unsupported KHR extension: {}", key),
            });

        // mat.normal_texture().map(|norm| {
        //     norm.
        // })

        let normal_texture = mat.normal_texture().map(Into::into);

        Material {
            name: mat.name().map(String::from),
            alpha_cutoff,
            alpha_mode,
            base_color_texture: pbr_metallic_roughness.base_color_texture().map(Into::into),
            normal_texture,
            pbr_metallic_roughness: pbr_metallic_roughness_data,
            pbr_specular_glossiness: None, // Handle this when the extension is supported
            unlit: mat.unlit(),
            variants: vec![],   // Handle this when the extension is supported
            volume: None,       // Handle this when the extension is supported
            specular: None,     // Handle this when the extension is supported
            transmission: None, // Handle this when the extension is supported
            ior: mat.ior(),
            emissive_strength: mat.emissive_strength(),
            emissive_texture: mat.emissive_texture().map(Into::into),
            emissive_factor: mat.emissive_factor(),
            occlusion_texture: mat.occlusion_texture().map(Into::into),
            sheen_roughness_factor,
            sheen_color_factor,
        }
    }
}

impl From<Info<'_>> for TextureInfo {
    fn from(info: Info<'_>) -> Self {
        Self {
            texture: Texture {
                index: info.texture().source().index(),
                tex_coord: info.tex_coord(),
            },
            transform: info.texture_transform().map(|t| TextureTransform {
                scale: t.scale(),
                offset: t.offset(),
                rotation: t.rotation(),
            }),
        }
    }
}

fn unwrap_vec<const N: usize>(value: &gltf::json::Value) -> Option<[f32; N]> {
    value
        .as_array()?
        .into_iter()
        .map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Option<Vec<_>>>()?
        .try_into()
        .ok()
}

trait ScalableTexture {
    fn tex_coord(&self) -> u32;
    fn extensions(&self) -> Option<&serde_json::map::Map<String, Value>>;
    fn texture(&self) -> texture::Texture<'_>;
    fn scale(&self) -> f32;
}

impl ScalableTexture for gltf::material::NormalTexture<'_> {
    fn tex_coord(&self) -> u32 {
        self.tex_coord()
    }

    fn extensions(&self) -> Option<&serde_json::map::Map<String, Value>> {
        self.extensions()
    }

    fn texture(&self) -> gltf::Texture<'_> {
        self.texture()
    }

    fn scale(&self) -> f32 {
        self.scale()
    }
}

impl ScalableTexture for gltf::material::OcclusionTexture<'_> {
    fn tex_coord(&self) -> u32 {
        self.tex_coord()
    }

    fn extensions(&self) -> Option<&serde_json::Map<String, Value>> {
        self.extensions()
    }

    fn texture(&self) -> gltf::Texture<'_> {
        self.texture()
    }

    fn scale(&self) -> f32 {
        self.strength()
    }
}

impl<T: ScalableTexture> From<T> for ScaledTexture {
    fn from(texture: T) -> Self {
        let mut transform = None;
        let mut tex_coord = texture.tex_coord();
        texture
            .extensions()
            .into_iter()
            .flatten()
            .for_each(|(key, value)| match key.as_str() {
                "KHR_texture_transform" => {
                    use gltf::json::Value;
                    let mut trs = TextureTransform::default();
                    if let Value::Object(items) = value {
                        for (key, value) in items {
                            match key.as_str() {
                                "offset" => trs.offset = unwrap_vec(value).unwrap(),
                                "scale" => trs.scale = unwrap_vec(value).unwrap(),
                                "rotation" => trs.rotation = value.as_f64().unwrap() as f32,
                                "texCoord" => tex_coord = value.as_i64().unwrap() as u32,
                                _ => panic!(
                                    "Unexpected KHR_texture_transform key: {}: {:?}",
                                    key, value
                                ),
                            }
                        }
                    }
                    transform = Some(trs);
                }
                _ => panic!("Unsupported KHR extension: {}", key),
            });

        let texture_info = TextureInfo {
            texture: Texture {
                index: texture.texture().index(),
                tex_coord,
            },
            transform,
        };

        Self {
            info: texture_info,
            scale: texture.scale(),
        }
    }
}

pub fn load_gltf(path: &str) -> Result<Gltf, Error> {
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

    let nodes = doc
        .nodes()
        .map(|node| {
            (
                Matrix4::from(node.transform().matrix()),
                node.mesh().map(|m| m.index()),
                node.children()
                    .map(|child| child.index())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let mut objects = Vec::new();
    {
        let mut stack = doc
            .scenes()
            .flat_map(|scene| {
                scene
                    .nodes()
                    .map(|node| (node.index(), Matrix4::identity()))
            })
            .collect::<Vec<_>>();

        while let Some((node_id, parent_transform)) = stack.pop() {
            let (node_transform, mesh_idx, children) = nodes[node_id].clone();

            let transform = parent_transform * node_transform;

            if let Some(mesh_idx) = mesh_idx {
                objects.push(Object {
                    transform,
                    mesh_idx,
                });
            }

            for child_id in children {
                stack.push((child_id, transform));
            }
        }
    }

    let meshes = doc
        .meshes()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|mesh| {
            let primitives = mesh
                .primitives()
                .collect::<Vec<_>>()
                .par_iter()
                .map(|prim| {
                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                    let positions = reader
                        .read_positions()
                        .unwrap()
                        .flatten()
                        .collect::<Vec<_>>();

                    let num_verts = positions.len();

                    let normals = reader.read_normals().unwrap().flatten().collect::<Vec<_>>();
                    let texcoords0 = reader
                        .read_tex_coords(0) // TODO: support multiple TEXCOORDs
                        .map(|t| t.into_f32().flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 2]);
                    let texcoords1 = reader
                        .read_tex_coords(1) // TODO: support multiple TEXCOORDs
                        .map(|t| t.into_f32().flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 2]);

                    let opt_tangents = reader
                        .read_tangents()
                        .map(|t| t.flatten().collect::<Vec<_>>());

                    let need_tangents = opt_tangents.is_none();
                    let tangents = opt_tangents.unwrap_or_else(|| vec![0.; num_verts * 4]);

                    let indices = reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .collect::<Vec<_>>();

                    let vertices = positions
                        .par_chunks_exact(3)
                        .zip(normals.par_chunks_exact(3))
                        .zip(tangents.par_chunks_exact(4))
                        .zip(texcoords0.par_chunks_exact(2))
                        .zip(texcoords1.par_chunks_exact(2))
                        .map(|((((position, normal), tangent), texcoord0), texcoord1)| {
                            CombinedVertex {
                                a_position: position.try_into().unwrap(),
                                a_normal: normal.try_into().unwrap(),
                                a_tangent: tangent.try_into().unwrap(),
                                a_texcoord_0: texcoord0.try_into().unwrap(),
                                a_texcoord_1: texcoord1.try_into().unwrap(),
                            }
                        })
                        .collect();

                    let mat_idx = prim.material().index().unwrap();

                    let mut prim = Primitive {
                        vertices,
                        indices,
                        mat_idx,
                    };

                    if need_tangents {
                        let timer = Instant::now();
                        mikktspace::generate_tangents(&mut prim);
                        println!("Generated {} tangents in {:?}", num_verts, timer.elapsed());
                    }

                    prim
                })
                .collect();
            Mesh { primitives }
        })
        .collect::<Vec<_>>();

    println!(
        "Loaded {} meshes in {:?}",
        meshes
            .iter()
            .map(|mesh| mesh.primitives.len())
            .sum::<usize>(),
        start_time.elapsed()
    );

    Ok(Gltf {
        meshes,
        textures,
        materials,
        objects,
    })
}
