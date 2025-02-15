use crate::material::{Material, NormalTexture, PbrMetallicRoughness, Texture};
use cgmath::{Matrix4, SquareMatrix};
use gltf::image::{Data, Format};
use gltf::texture::Info;
use gltf::Error;
use std::time::Instant;

pub struct Primitive {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub tangents: Vec<f32>,
    pub texcoords: Vec<f32>,
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
        let v0 = self.positions[tri * 3];
        let v1 = self.positions[tri * 3 + 1];
        let v2 = self.positions[tri * 3 + 2];
        [v0, v1, v2]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let tri = self.indices[face * 3 + vert] as usize;
        let n0 = self.normals[tri * 3];
        let n1 = self.normals[tri * 3 + 1];
        let n2 = self.normals[tri * 3 + 2];
        [n0, n1, n2]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let tri = self.indices[face * 3 + vert] as usize;
        let uv0 = self.texcoords[tri * 2];
        let uv1 = self.texcoords[tri * 2 + 1];
        [uv0, uv1]
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bi_tangent: [f32; 3],
        _f_mag_s: f32,
        _f_mag_t: f32,
        _bi_tangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let tri = self.indices[face * 3 + vert] as usize;
        self.tangents[tri * 3] = tangent[0];
        self.tangents[tri * 3 + 1] = tangent[1];
        self.tangents[tri * 3 + 2] = tangent[2];
    }

    fn set_tangent_encoded(&mut self, _tangent: [f32; 4], _face: usize, _vert: usize) {
        let tri = self.indices[_face * 3 + _vert] as usize;
        self.tangents[tri * 3] = _tangent[0];
        self.tangents[tri * 3 + 1] = _tangent[1];
        self.tangents[tri * 3 + 2] = _tangent[2];
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
        .map(|mesh| {
            let primitives = mesh
                .primitives()
                .map(|prim| {
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

                    let opt_tangents = reader
                        .read_tangents()
                        .map(|t| t.flatten().collect::<Vec<_>>());

                    let indices = reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .collect::<Vec<_>>();
                    let mat_idx = prim.material().index().unwrap();

                    let num_verts = positions.len();
                    let mut prim = Primitive {
                        positions,
                        normals,
                        tangents: Vec::new(),
                        texcoords,
                        indices,
                        mat_idx,
                    };

                    match opt_tangents {
                        Some(tangents) => {
                            prim.tangents = tangents;
                        }
                        None => {
                            prim.tangents =
                                vec![[0f32; 3]; num_verts].into_iter().flatten().collect();
                            let start_time = Instant::now();
                            mikktspace::generate_tangents(&mut prim);
                            println!("Generated tangents in {:?}", start_time.elapsed());
                        }
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
