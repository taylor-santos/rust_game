use crate::material::*;
use crate::shader::*;
use cgmath::{Matrix4, SquareMatrix};
use gltf::image::{Data, Format};
use gltf::json::Value;
use gltf::mesh::util::ReadColors;
use gltf::texture::Info;
use gltf::Error;
use rayon::iter::Either;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex, Debug)]
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
    #[format(R32G32B32A32_SFLOAT)]
    a_color_0: [f32; 4],
}

#[derive(Debug)]
pub struct Primitive {
    pub vertices: Vec<CombinedVertex>,
    pub indices: Vec<u32>,
    pub mat_idx: usize,
    pub spec_const: ObjectSpecializationConstants,
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
    pub texture_maps: Vec<TextureMap>,
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
        // convert coordinate system handedness to respect output format of MikkTSpace
        _tangent[3] = -_tangent[3];
        self.vertices[tri].a_tangent = _tangent;
    }
}

impl From<gltf::Material<'_>> for Material {
    fn from(mat: gltf::Material<'_>) -> Self {
        let mut extensions = mat.extensions().cloned();

        let mat = Self {
            name: mat.name().map(Into::into),
            pbr_metallic_roughness: mat.pbr_metallic_roughness().into(),
            normal_texture: mat.normal_texture().map(Into::into),
            occlusion_texture: mat.occlusion_texture().map(Into::into),
            emissive_texture: mat.emissive_texture().map(Into::into),
            emissive_factor: mat.emissive_factor(),
            alpha_mode: mat.alpha_mode().into(),
            alpha_cutoff: mat.alpha_cutoff().map(Into::into).unwrap_or_default(),
            double_sided: mat.double_sided(),
            pbr_specular_glossiness: mat.pbr_specular_glossiness().map(Into::into),
            anisotropy: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_anisotropy"))
                .map(Into::into),
            clearcoat: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_clearcoat"))
                .map(Into::into),
            diffuse_transmission: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_diffuse_transmission"))
                .map(Into::into),
            dispersion: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_dispersion"))
                .map(Into::into),
            emissive_strength: mat.emissive_strength().map(Into::into),
            ior: mat.ior().map(Into::into),
            iridescence: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_iridescence"))
                .map(Into::into),
            sheen: extensions
                .as_mut()
                .and_then(|m| m.remove("KHR_materials_sheen"))
                .map(Into::into),
            specular: mat.specular().map(Into::into),
            transmission: mat.transmission().map(Into::into),
            unlit: mat.unlit(),
            volume: mat.volume().map(Into::into),
        };

        if let Some(extensions) = extensions {
            if !extensions.is_empty() {
                eprintln!("WARN: Unhandled extension: {:?}", &extensions);
            }
        }

        mat
    }
}

impl From<gltf::material::PbrMetallicRoughness<'_>> for PbrMetallicRoughness {
    fn from(mr: gltf::material::PbrMetallicRoughness) -> Self {
        Self {
            base_color_factor: mr.base_color_factor(),
            base_color_texture: mr.base_color_texture().map(Into::into),
            metallic_factor: mr.metallic_factor(),
            roughness_factor: mr.roughness_factor(),
            metallic_roughness_texture: mr.metallic_roughness_texture().map(Into::into),
        }
    }
}

impl From<gltf::material::AlphaMode> for AlphaMode {
    fn from(alpha_mode: gltf::material::AlphaMode) -> Self {
        match alpha_mode {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        }
    }
}

impl From<gltf::material::PbrSpecularGlossiness<'_>> for PbrSpecularGlossiness {
    fn from(sg: gltf::material::PbrSpecularGlossiness) -> Self {
        Self {
            diffuse_factor: sg.diffuse_factor(),
            diffuse_texture: sg.diffuse_texture().map(Into::into),
            specular_factor: sg.specular_factor(),
            glossiness_factor: sg.glossiness_factor(),
            specular_glossiness_texture: sg.specular_glossiness_texture().map(Into::into),
        }
    }
}

impl From<gltf::material::Specular<'_>> for Specular {
    fn from(s: gltf::material::Specular) -> Self {
        Self {
            factor: s.specular_factor(),
            texture: s.specular_texture().map(Into::into),
            color_factor: s.specular_color_factor(),
            color_texture: s.specular_color_texture().map(Into::into),
        }
    }
}

fn to_array<const N: usize>(value: Value) -> [f32; N] {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

impl From<Value> for Sheen {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut sheen = Sheen::default();

        if let Some(color_factor) = map.remove("sheenColorFactor").map(to_array) {
            sheen.color_factor = color_factor;
        }
        sheen.color_texture = map.remove("sheenColorTexture").map(Into::into);
        if let Some(roughness_factor) = map
            .remove("sheenRoughnessFactor")
            .map(|t| t.as_f64().unwrap() as f32)
        {
            sheen.roughness_factor = roughness_factor;
        }
        sheen.roughness_texture = map.remove("sheenRoughnessTexture").map(Into::into);

        if !map.is_empty() {
            eprintln!("WARN: Unhandled KHR_materials_sheen values: {:?}", &map);
        }

        sheen
    }
}

impl From<Value> for Clearcoat {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut clearcoat = Clearcoat::default();

        if let Some(factor) = map
            .remove("clearcoatFactor")
            .map(|f| f.as_f64().unwrap() as f32)
        {
            clearcoat.factor = factor;
        }
        clearcoat.texture = map.remove("clearcoatTexture").map(Into::into);
        if let Some(roughness) = map
            .remove("clearcoatRoughnessFactor")
            .map(|r| r.as_f64().unwrap() as f32)
        {
            clearcoat.roughness_factor = roughness;
        }
        clearcoat.roughness_texture = map.remove("clearcoatRoughnessTexture").map(Into::into);
        clearcoat.normal_texture = map.remove("clearcoatNormalTexture").map(Into::into);

        if !map.is_empty() {
            eprintln!("WARN: Unhandled KHR_materials_clearcoat values: {:?}", &map);
        }

        clearcoat
    }
}

impl From<Value> for Anisotropy {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut anisotropy = Anisotropy::default();

        if let Some(strength) = map
            .remove("anisotropyStrength")
            .map(|s| s.as_f64().unwrap() as f32)
        {
            anisotropy.strength = strength;
        }
        if let Some(rotation) = map
            .remove("anisotropyRotation")
            .map(|r| r.as_f64().unwrap() as f32)
        {
            anisotropy.rotation = rotation;
        }
        anisotropy.texture = map.remove("anisotropyTexture").map(Into::into);

        if !map.is_empty() {
            eprintln!(
                "WARN: Unhandled KHR_materials_anisotropy values: {:?}",
                &map
            );
        }

        anisotropy
    }
}

impl From<Value> for Iridescence {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut iridescence = Iridescence::default();

        if let Some(factor) = map
            .remove("iridescenceFactor")
            .map(|f| f.as_f64().unwrap() as f32)
        {
            iridescence.factor = factor;
        }
        iridescence.texture = map.remove("iridescenceTexture").map(Into::into);
        if let Some(ior) = map
            .remove("iridescenceIor")
            .map(|i| i.as_f64().unwrap() as f32)
        {
            iridescence.ior = ior;
        }
        if let Some(thickness_minimum) = map
            .remove("iridescenceThicknessMinimum")
            .map(|m| m.as_f64().unwrap() as f32)
        {
            iridescence.thickness_minimum = thickness_minimum;
        }
        if let Some(thickness_maximum) = map
            .remove("iridescenceThicknessMaximum")
            .map(|m| m.as_f64().unwrap() as f32)
        {
            iridescence.thickness_maximum = thickness_maximum;
        }
        iridescence.thickness_texture = map.remove("iridescenceThicknessTexture").map(Into::into);

        if !map.is_empty() {
            eprintln!(
                "WARN: Unhandled KHR_materials_iridescence values: {:?}",
                &map
            );
        }

        iridescence
    }
}

impl From<Value> for Dispersion {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut dispersion = Dispersion::default();

        if let Some(d) = map.remove("dispersion").map(|d| d.as_f64().unwrap() as f32) {
            dispersion = d.into();
        }

        if !map.is_empty() {
            eprintln!(
                "WARN: Unhandled KHR_materials_dispersion values: {:?}",
                &map
            );
        }

        dispersion
    }
}

impl From<Value> for DiffuseTransmission {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut diffuse_transmission = DiffuseTransmission::default();

        if let Some(factor) = map
            .remove("diffuseTransmissionFactor")
            .map(|f| f.as_f64().unwrap() as f32)
        {
            diffuse_transmission.factor = factor;
        }
        diffuse_transmission.texture = map.remove("diffuseTransmissionTexture").map(Into::into);
        if let Some(color_factor) = map.remove("diffuseTransmissionColorFactor").map(to_array) {
            diffuse_transmission.color_factor = color_factor;
        }
        diffuse_transmission.color_texture = map
            .remove("diffuseTransmissionColorTexture")
            .map(Into::into);

        if !map.is_empty() {
            eprintln!(
                "WARN: Unhandled KHR_materials_diffuse_transmission values: {:?}",
                &map
            );
        }

        diffuse_transmission
    }
}

impl From<Value> for TextureTransform {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let mut texture_transform = TextureTransform::default();

        if let Some(offset) = map.remove("offset").map(to_array) {
            texture_transform.offset = offset;
        }
        if let Some(rotation) = map.remove("rotation").map(|v| v.as_f64().unwrap() as f32) {
            texture_transform.rotation = rotation;
        }
        if let Some(scale) = map.remove("scale").map(to_array) {
            texture_transform.scale = scale;
        }
        texture_transform.tex_coord = map.remove("texCoord").map(|v| v.as_i64().unwrap() as u32);

        if !map.is_empty() {
            eprintln!("WARN: Unhandled KHR_texture_transform values: {:?}", &map);
        }

        texture_transform
    }
}

impl From<Value> for TextureInfo {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let index = map.remove("index").unwrap().as_i64().unwrap() as usize;
        let tex_coord = map
            .remove("texCoord")
            .map(|t| t.as_i64().unwrap() as u32)
            .unwrap_or(0);
        let mut extensions = map
            .remove("extensions")
            .map(|e| e.as_object().unwrap().clone());

        let transform = extensions
            .as_mut()
            .and_then(|e| e.remove("KHR_texture_transform"))
            .map(Into::into);

        if let Some(extensions) = extensions {
            if !extensions.is_empty() {
                eprintln!("WARN: Unhandled extension: {:?}", &extensions);
            }
        }

        if !map.is_empty() {
            eprintln!("WARN: Unhandled texture values: {:?}", &map);
        }

        TextureInfo {
            texture: Texture { index, tex_coord },
            transform,
        }
    }
}

impl From<Value> for NormalTexture {
    fn from(value: Value) -> Self {
        let mut map = value.as_object().unwrap().clone();

        let index = map
            .remove("index")
            .and_then(|v| v.as_i64())
            .expect("Normal Texture requires index value") as usize;
        let tex_coord = map
            .remove("texCoord")
            .map(|v| v.as_i64().unwrap() as u32)
            .unwrap_or(0);
        let scale = map
            .remove("scale")
            .map(|v| v.as_f64().unwrap() as f32)
            .unwrap_or(1.0);

        let mut extensions = map
            .remove("extensions")
            .map(|e| e.as_object().unwrap().clone());

        let transform = extensions
            .as_mut()
            .and_then(|e| e.remove("KHR_texture_transform"))
            .map(Into::into);

        if let Some(extensions) = extensions {
            if !extensions.is_empty() {
                eprintln!("WARN: Unhandled extension: {:?}", &extensions);
            }
        }

        if !map.is_empty() {
            eprintln!("WARN: Unhandled normal texture values: {:?}", &map);
        }

        Self {
            info: TextureInfo {
                texture: Texture { index, tex_coord },
                transform,
            },
            scale,
        }
    }
}

impl From<gltf::material::Transmission<'_>> for Transmission {
    fn from(t: gltf::material::Transmission) -> Self {
        Self {
            factor: t.transmission_factor(),
            texture: t.transmission_texture().map(Into::into),
        }
    }
}

impl From<gltf::material::Volume<'_>> for Volume {
    fn from(v: gltf::material::Volume) -> Self {
        Self {
            thickness_factor: v.thickness_factor(),
            thickness_texture: v.thickness_texture().map(Into::into),
            attenuation_distance: v.attenuation_distance(),
            attenuation_color: v.attenuation_color(),
        }
    }
}

impl From<gltf::material::NormalTexture<'_>> for NormalTexture {
    fn from(texture: gltf::material::NormalTexture) -> Self {
        Self {
            scale: texture.scale(),
            info: TextureInfo {
                texture: Texture {
                    index: texture.texture().index(),
                    tex_coord: texture.tex_coord(),
                },
                transform: texture
                    .extension_value("KHR_texture_transform")
                    .cloned()
                    .map(Into::into),
            },
        }
    }
}

impl From<gltf::material::OcclusionTexture<'_>> for OcclusionTexture {
    fn from(texture: gltf::material::OcclusionTexture) -> Self {
        Self {
            strength: texture.strength(),
            info: TextureInfo {
                texture: Texture {
                    index: texture.texture().index(),
                    tex_coord: texture.tex_coord(),
                },
                transform: texture
                    .extension_value("KHR_texture_transform")
                    .cloned()
                    .map(Into::into),
            },
        }
    }
}

impl From<Info<'_>> for TextureInfo {
    fn from(info: Info<'_>) -> Self {
        Self {
            texture: Texture {
                index: info.texture().index(),
                tex_coord: info.tex_coord(),
            },
            transform: info.texture_transform().map(|t| TextureTransform {
                scale: t.scale(),
                offset: t.offset(),
                rotation: t.rotation(),
                tex_coord: t.tex_coord(),
            }),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum SrgbUsage {
    // glTF
    BaseColor,
    Emissive,
    // KHR_materials_pbrSpecularGlossiness
    Diffuse,
    // KHR_materials_diffuse_transmission
    DiffuseTransmissionColor,
    // KHR_materials_sheen
    SheenColor,
    // KHR_materials_specular
    SpecularColor,
}

#[derive(Debug, PartialEq)]
pub enum LinearUsage {
    // glTF
    Normal,
    MetallicRoughness,
    Occlusion,
    // KHR_materials_pbrSpecularGlossiness
    SpecularGlossiness,
    // KHR_materials_anisotropy
    Anisotropy,
    // KHR_materials_clearcoat
    Clearcoat,
    ClearcoatRoughness,
    ClearcoatNormal,
    // KHR_materials_diffuse_transmission
    DiffuseTransmission,
    // KHR_materials_iridescence
    Iridescence,
    IridescenceThickness,
    // KHR_materials_sheen
    SheenRoughness,
    // KHR_materials_specular
    Specular,
    // KHR_materials_transmission
    Transmission,
    // KHR_materials_volume
    Thickness,
}

type TextureUsage = Either<SrgbUsage, LinearUsage>;

pub struct TextureMap {
    pub index: usize,
    pub usage: Option<TextureUsage>,
}

pub fn load_gltf<P: AsRef<Path>>(path: P) -> Result<Gltf, Error> {
    let mut start_time = Instant::now();
    let f = File::open(&path)?;
    let reader = BufReader::new(f);
    let gltf::Gltf { document, blob } = gltf::Gltf::from_reader_without_validation(reader)?;
    let base = path.as_ref().parent().unwrap_or_else(|| Path::new("./"));
    let buffers = gltf::import_buffers(&document, Some(base), blob)?;

    let textures = gltf::import_images(&document, Some(base), &buffers)?;

    println!("Loaded gltf in {:?}", start_time.elapsed());
    start_time = Instant::now();

    let mut texture_maps: Vec<_> = document
        .textures()
        .map(|tex| TextureMap {
            index: tex.source().unwrap().index(),
            usage: None,
        })
        .collect();

    let materials: Vec<Material> = document.materials().map(Into::into).collect();
    println!(
        "Loaded {} materials in {:?}",
        materials.len(),
        start_time.elapsed()
    );

    let mut add_texture_usage = |idx: usize, usage: TextureUsage| {
        let tex = &mut texture_maps[idx];
        if let Some(old_usage) = &tex.usage {
            if old_usage.is_left() != usage.is_left() {
                eprintln!(
                    "WARN: Texture {} used in both sRGB and Linear contexts",
                    idx
                );
            }
        } else {
            tex.usage = Some(usage);
        }
    };

    materials.iter().for_each(|mat| {
        use Either::*;
        use LinearUsage::*;
        use SrgbUsage::*;
        if let Some(t) = &mat.pbr_metallic_roughness.base_color_texture {
            add_texture_usage(t.texture.index, Left(BaseColor));
        }
        if let Some(t) = &mat.pbr_metallic_roughness.metallic_roughness_texture {
            add_texture_usage(t.texture.index, Right(MetallicRoughness));
        }
        if let Some(t) = &mat.normal_texture {
            add_texture_usage(t.info.texture.index, Right(Normal));
        }
        if let Some(t) = &mat.occlusion_texture {
            add_texture_usage(t.info.texture.index, Right(Occlusion));
        }
        if let Some(t) = &mat.emissive_texture {
            add_texture_usage(t.texture.index, Left(Emissive));
        }
        if let Some(pbr_specular_glossiness) = &mat.pbr_specular_glossiness {
            if let Some(t) = &pbr_specular_glossiness.diffuse_texture {
                add_texture_usage(t.texture.index, Left(Diffuse));
            }
            if let Some(t) = &pbr_specular_glossiness.specular_glossiness_texture {
                add_texture_usage(t.texture.index, Right(SpecularGlossiness));
            }
        }
        if let Some(t) = &mat.anisotropy.as_ref().and_then(|a| a.texture) {
            add_texture_usage(t.texture.index, Right(Anisotropy));
        }
        if let Some(clearcoat) = &mat.clearcoat {
            if let Some(t) = &clearcoat.texture {
                add_texture_usage(t.texture.index, Right(Clearcoat));
            }
            if let Some(t) = &clearcoat.roughness_texture {
                add_texture_usage(t.texture.index, Right(ClearcoatRoughness));
            }
            if let Some(t) = &clearcoat.normal_texture {
                add_texture_usage(t.info.texture.index, Right(ClearcoatNormal));
            }
        }
        if let Some(diffuse_transmission) = &mat.diffuse_transmission {
            if let Some(t) = &diffuse_transmission.texture {
                add_texture_usage(t.texture.index, Right(DiffuseTransmission));
            }
            if let Some(t) = &diffuse_transmission.color_texture {
                add_texture_usage(t.texture.index, Left(DiffuseTransmissionColor));
            }
        }
        if let Some(iridescence) = &mat.iridescence {
            if let Some(t) = &iridescence.texture {
                add_texture_usage(t.texture.index, Right(Iridescence));
            }
            if let Some(t) = &iridescence.thickness_texture {
                add_texture_usage(t.texture.index, Right(IridescenceThickness));
            }
        }
        if let Some(sheen) = &mat.sheen {
            if let Some(t) = &sheen.color_texture {
                add_texture_usage(t.texture.index, Left(SheenColor));
            }
            if let Some(t) = &sheen.roughness_texture {
                add_texture_usage(t.texture.index, Right(SheenRoughness));
            }
        }
        if let Some(specular) = &mat.specular {
            if let Some(t) = &specular.texture {
                add_texture_usage(t.texture.index, Right(Specular));
            }
            if let Some(t) = &specular.color_texture {
                add_texture_usage(t.texture.index, Left(SpecularColor));
            }
        }
        if let Some(t) = &mat.transmission.as_ref().and_then(|t| t.texture) {
            add_texture_usage(t.texture.index, Right(Transmission));
        }
        if let Some(t) = &mat.volume.as_ref().and_then(|v| v.thickness_texture) {
            add_texture_usage(t.texture.index, Right(Thickness));
        }
    });

    for texture in &texture_maps {
        if texture.usage.is_none() {
            eprintln!("WARN: texture {} unused", texture.index);
        }
    }

    start_time = Instant::now();

    let nodes = document
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
        let mut stack = document
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

    let meshes = document
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

                    let normals = reader.read_normals();
                    let tangents = reader.read_tangents();
                    let texcoords0 = reader.read_tex_coords(0);
                    let texcoords1 = reader.read_tex_coords(1);
                    let colors = reader.read_colors(0);

                    let (color3, color4) = match colors {
                        None => (false, false),
                        Some(ReadColors::RgbU8(_))
                        | Some(ReadColors::RgbU16(_))
                        | Some(ReadColors::RgbF32(_)) => (true, false),
                        Some(ReadColors::RgbaU8(_))
                        | Some(ReadColors::RgbaU16(_))
                        | Some(ReadColors::RgbaF32(_)) => (false, true),
                    };

                    // If tangents are not provided by the model, they can be generated using the MikkTSpace algorithm.
                    // This requires the presence of vertex normals and texcoords.
                    let should_generate_tangents =
                        tangents.is_none() && normals.is_some() && texcoords0.is_some();

                    let spec_constants = ObjectSpecializationConstants {
                        HAS_NORMAL_VEC3: normals.is_some(),
                        HAS_TANGENT_VEC4: tangents.is_some() || should_generate_tangents,
                        HAS_TEXCOORD_0_VEC2: texcoords0.is_some(),
                        HAS_TEXCOORD_1_VEC2: texcoords1.is_some(),
                        HAS_COLOR_0_VEC3: color3,
                        HAS_COLOR_0_VEC4: color4,
                        NOT_TRIANGLE: false,
                    };

                    let positions = reader
                        .read_positions()
                        .unwrap()
                        .flatten()
                        .collect::<Vec<_>>();

                    let num_verts = positions.len();

                    let normals = normals
                        .map(|n| n.flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 3]);
                    let texcoords0 = texcoords0
                        .map(|t| t.into_f32().flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 2]);
                    let texcoords1 = texcoords1
                        .map(|t| t.into_f32().flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 2]);
                    let colors = colors
                        .map(|c| c.into_rgba_f32().flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 4]);
                    let tangents = tangents
                        .map(|t| t.flatten().collect::<Vec<_>>())
                        .unwrap_or_else(|| vec![0.; num_verts * 4]);

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
                        .zip(colors.par_chunks_exact(4))
                        .map(
                            |(((((position, normal), tangent), texcoord0), texcoord1), color)| {
                                CombinedVertex {
                                    a_position: position.try_into().unwrap(),
                                    a_normal: normal.try_into().unwrap(),
                                    a_tangent: tangent.try_into().unwrap(),
                                    a_texcoord_0: texcoord0.try_into().unwrap(),
                                    a_texcoord_1: texcoord1.try_into().unwrap(),
                                    a_color_0: color.try_into().unwrap(),
                                }
                            },
                        )
                        .collect();

                    let mat_idx = prim.material().index().unwrap();

                    let mut prim = Primitive {
                        vertices,
                        indices,
                        mat_idx,
                        spec_const: spec_constants,
                    };

                    if should_generate_tangents {
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
        texture_maps,
        materials,
        objects,
    })
}
