// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.
//
// This version of the triangle example is written using dynamic rendering instead of render pass
// and framebuffer objects. If your device does not support Vulkan 1.3 or the
// `khr_dynamic_rendering` extension, or if you want to see how to support older versions, see the
// original triangle example.

use crate::camera::FirstPersonCamera;
use crate::gltf::{load_gltf, CombinedVertex, CubemapVertex, Gltf, Object, Scene, TextureFormat};
use crate::gui::Gui;
use crate::material::{AlphaMode, Material};
use crate::renderer::{Renderer, RendererContext};
use crate::shader::{
    cubemap_fs, cubemap_vs, fs, vs, MaterialSpecializationConstants, ObjectSpecializationConstants,
    RenderType, SpecializationConstants,
};
use c_str_macro::c_str;
use cgmath::num_traits::Float;
use cgmath::{EuclideanSpace, Matrix, Matrix4, MetricSpace, SquareMatrix};
use image::{ColorType, DynamicImage, ImageBuffer, ImageReader};
use imgui::sys::{
    igDockSpaceOverViewport, igGetMainViewport, ImGuiDockNodeFlags_PassthruCentralNode,
};
use imgui::{Condition, DragDropFlags, StyleColor, TableFlags, TreeNodeFlags, WindowFlags};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use ktx2::SupercompressionScheme;
use rayon::iter::Either;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f32::consts::PI;
use std::ptr::null;
use std::time::{Duration, Instant};
use std::{error::Error, sync::Arc};
use vulkano::buffer::allocator::SubbufferAllocator;
use vulkano::command_buffer::{
    BlitImageInfo, BufferImageCopy, CopyBufferInfo, CopyBufferToImageInfo, ImageBlit,
    PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::{ClearValue, Format};
use vulkano::half::f16;
use vulkano::image::sampler::SamplerAddressMode::{ClampToEdge, MirroredRepeat, Repeat};
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    max_mip_levels, mip_level_extent, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout,
    ImageSubresourceLayers, ImageType,
};
use vulkano::padded::Padded;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::shader::{DescriptorBindingRequirements, ShaderModule, ShaderStages};
use vulkano::swapchain::PresentMode;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::Queue,
    image::{view::ImageView, Image, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    sync::{self, GpuFuture},
    DeviceSize,
};
use vulkano_util::window::WindowDescriptor;
use winit::event::{DeviceEvent, DeviceId, ElementState, Event, MouseButton, StartCause};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::CursorGrabMode;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

mod camera;
mod gltf;
mod gui;
mod material;
mod renderer;
mod shader;
mod transform;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    event_loop.run_app(&mut app)
}

struct Skybox {
    pub lambertian: Arc<ImageView>,
    pub ggx: Arc<ImageView>,
    pub charlie: Arc<ImageView>,
    pub lut_ggx: Arc<ImageView>,
    pub lut_charlie: Arc<ImageView>,
    pub lut_sheen_e: Arc<ImageView>,
}

struct Samplers {
    mirror_sampler_mipmap: Arc<Sampler>,
    clamp_sampler_no_mipmap: Arc<Sampler>,
    wrap_sampler_mipmap: Arc<Sampler>,
}

struct App {
    renderer: Renderer,
    vertex_buffer: Subbuffer<[CombinedVertex]>,
    index_buffer: Subbuffer<[u32]>,
    scene: Scene,
    mesh_infos: Vec<MeshDrawInfo>,
    prim_infos: Vec<PrimitiveDrawInfo>,
    materials: Vec<Material>,
    mat_prims: Vec<HashSet<usize>>,
    textures: Vec<Arc<ImageView>>,
    null_texture: Arc<ImageView>,
    skyboxes: Skybox,
    samplers: Samplers,
    camera: FirstPersonCamera,
    input_state: InputState,
    gui: Gui,
    imgui_ctx: imgui::Context,
    last_frame: Instant,
    rcx: Option<RenderContext>,
}

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    mouse_dx: f64,
    mouse_dy: f64,
    cursor_confined: bool,
}

struct RenderContext {
    cubemap_pipeline: Arc<GraphicsPipeline>,
    cubemap_index_buffer: Subbuffer<[u32]>,
    cubemap_vertex_buffer: Subbuffer<[CubemapVertex]>,
    cubemap_index_count: u32,
    intermediate_image_view: Arc<ImageView>,
    const_set: Arc<DescriptorSet>,
    skybox_set: Arc<DescriptorSet>,
    framebuffer_set: Arc<DescriptorSet>,
    viewport: Viewport,
    frame_times: VecDeque<Instant>,
    imgui_platform: WinitPlatform,
    imgui_renderer: imgui_vulkano_renderer::Renderer,
}

#[allow(clippy::too_many_arguments)]
fn upload_image<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pixels: I,
    extent: [u32; 3],
    mip_levels: u32,
    array_layers: u32,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    format: Format,
) -> Result<Arc<ImageView>, impl Error>
where
    I::IntoIter: ExactSizeIterator,
{
    let upload_buffer = create_buffer(
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue,
        BufferUsage::TRANSFER_SRC,
        pixels,
    );

    let flags = if array_layers == 6 {
        ImageCreateFlags::CUBE_COMPATIBLE
    } else {
        ImageCreateFlags::empty()
    };

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            flags,
            array_layers,
            image_type: ImageType::Dim2d,
            mip_levels,
            format,
            extent,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let regions = {
        let mut buffer_offset = 0;
        let mut mip_width = extent[0];
        let mut mip_height = extent[1];
        (0..mip_levels)
            .map(|mip_level| {
                let region = BufferImageCopy {
                    buffer_offset,
                    image_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level,
                        array_layers: 0..array_layers,
                    },
                    image_extent: [mip_width, mip_height, 1],
                    ..Default::default()
                };

                buffer_offset +=
                    format.block_size() * DeviceSize::from(array_layers * mip_width * mip_height);
                // Each successive Mip level is 4x smaller than the last, each dimension must be divided by 2
                mip_width /= 2;
                mip_height /= 2;

                region
            })
            .collect()
    };

    builder.copy_buffer_to_image(CopyBufferToImageInfo {
        regions,
        ..CopyBufferToImageInfo::buffer_image(upload_buffer, image.clone())
    })?;

    let view_type = if array_layers == 6 {
        ImageViewType::Cube
    } else {
        ImageViewType::Dim2d
    };

    let create_info = ImageViewCreateInfo {
        view_type,
        ..ImageViewCreateInfo::from_image(&image)
    };

    ImageView::new(image, create_info)
}

fn load_ktx2(
    path: &str,
    image_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
) -> Result<Arc<ImageView>, Box<dyn Error>> {
    let buf = std::fs::read(path)?;
    let reader = ktx2::Reader::new(buf)?;
    let header = reader.header();
    assert_eq!(header.face_count, 6, "Must be a cubemap");
    assert_eq!(
        header.format,
        Some(ktx2::Format::R16G16B16A16_SFLOAT),
        "Only supports RGBA_SFLOAT_16"
    );

    let pixel_bytes = header.type_size * 4;
    let face_pixels = header.pixel_width * header.pixel_height;
    let face_bytes = face_pixels * pixel_bytes;
    let mip0_bytes = face_bytes * header.face_count;

    let total_bytes = (0..header.level_count)
        .map(|level| {
            mip0_bytes >> (level * 2) // Each mip level is 4x smaller than the last
        })
        .sum::<u32>();

    let bytes: Vec<u8> = reader.levels().fold(
        Vec::with_capacity(total_bytes as usize),
        |mut bytes, level| {
            let data = match header.supercompression_scheme {
                None => level.to_vec(),
                Some(SupercompressionScheme::Zstandard) => zstd::decode_all(level).unwrap(),
                Some(scheme) => panic!("Unsupported compression scheme: {scheme:?}"),
            };
            bytes.extend(data);
            bytes
        },
    );
    assert_eq!(
        bytes.len(),
        total_bytes as usize,
        "Mip levels did not add up to the expected number of bytes",
    );

    let extent: [u32; 3] = [header.pixel_width, header.pixel_height, 1];

    Ok(upload_image(
        image_builder,
        bytes,
        extent,
        header.level_count,
        header.face_count,
        memory_allocator,
        command_buffer_allocator,
        queue,
        Format::R16G16B16A16_SFLOAT,
    )?)
}

fn load_png(
    path: &str,
    image_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
) -> Result<Arc<ImageView>, Box<dyn Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let color = img.color();
    let width = img.width();
    let height = img.height();
    match color {
        ColorType::Rgb8 => {
            let pixels = img.to_rgba8().as_raw().to_owned();
            Ok(upload_image(
                image_builder,
                pixels,
                [width, height, 1],
                1,
                1,
                memory_allocator,
                command_buffer_allocator,
                queue,
                Format::R8G8B8A8_UNORM,
            )?)
        }
        ColorType::Rgb16 => {
            let pixels = img.to_rgba16().as_raw().to_owned();
            Ok(upload_image(
                image_builder,
                pixels,
                [width, height, 1],
                1,
                1,
                memory_allocator,
                command_buffer_allocator,
                queue,
                Format::R16G16B16A16_UNORM,
            )?)
        }
        _ => panic!("Unsupported color type {color:?}"),
    }
}

impl App {
    fn new() -> Self {
        let renderer = Renderer::new();

        let Gltf {
            meshes,
            textures,
            texture_maps,
            mut materials,
            objects,
        } = load_gltf("models/DragonDispersion.glb").expect("Couldn't load gltf model");

        let timer = Instant::now();

        let (vert_count, index_count) = meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .map(|prim| (prim.vertices.len(), prim.indices.len()))
            .reduce(|(v1, i1), (v2, i2)| (v1 + v2, i1 + i2))
            .unwrap();

        let (vertex_buffer, index_buffer, mesh_infos, prim_infos, mat_prims) = {
            let mut combined_verts = Vec::with_capacity(vert_count / 3);
            let mut combined_indices = Vec::with_capacity(index_count);
            let mut mesh_infos = Vec::new();
            let mut prim_infos = Vec::new();
            let mut prims_offset = 0;
            let mut needs_default_mat = false;
            let mut mat_prims: Vec<_> = std::iter::repeat(HashSet::new())
                .take(materials.len() + 1 /* add 1 for default mat */)
                .collect();
            for mesh in meshes {
                let prims_count = mesh.primitives.len();
                for prim in mesh.primitives {
                    let mat_idx = prim.mat_idx.map_or_else(
                        || {
                            needs_default_mat = true;
                            materials.len()
                        },
                        |idx| idx,
                    );
                    mat_prims[mat_idx].insert(prim_infos.len());

                    prim_infos.push(PrimitiveDrawInfo {
                        index_offset: combined_indices.len() as u32,
                        vertex_offset: combined_verts.len() as i32,
                        index_count: prim.indices.len() as u32,
                        mat_idx,
                        spec_const: prim.spec_const,
                        object_ids: HashSet::new(),
                    });

                    combined_verts.extend(prim.vertices);
                    combined_indices.extend(prim.indices);
                }
                mesh_infos.push(MeshDrawInfo {
                    prims_offset,
                    prims_count,
                });
                prims_offset += prims_count;
            }

            if needs_default_mat {
                materials.push(Material::default());
            }

            for (obj_idx, object) in objects.iter().enumerate() {
                if let Some(mesh_idx) = object.mesh_idx {
                    let mesh_info = &mesh_infos[mesh_idx];
                    for prim in prim_infos
                        .iter_mut()
                        .skip(mesh_info.prims_offset)
                        .take(mesh_info.prims_count)
                    {
                        prim.object_ids.insert(obj_idx);
                    }
                }
            }

            let vertex_buffer = create_buffer(
                renderer.allocators.memory.clone(),
                renderer.allocators.command_buffer.clone(),
                renderer.context.graphics_queue(),
                BufferUsage::VERTEX_BUFFER,
                combined_verts,
            );

            let index_buffer = create_buffer(
                renderer.allocators.memory.clone(),
                renderer.allocators.command_buffer.clone(),
                renderer.context.graphics_queue(),
                BufferUsage::INDEX_BUFFER,
                combined_indices,
            );

            println!("Combined vertex data in {:?}", timer.elapsed());

            (
                vertex_buffer,
                index_buffer,
                mesh_infos,
                prim_infos,
                mat_prims,
            )
        };

        let mut image_builder = AutoCommandBufferBuilder::primary(
            renderer.allocators.command_buffer.clone(),
            renderer.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let textures = {
            let timer = Instant::now();

            let textures: Vec<_> = texture_maps
                .into_par_iter()
                .map(|texture_map| {
                    let texture = &textures[texture_map.index];
                    let is_srgb = texture_map.usage.as_ref().is_some_and(Either::is_left);

                    let (pixels, format) = match texture.format {
                        TextureFormat::R16 => {
                            if is_srgb {
                                let pixels = texture
                                    .pixels
                                    .chunks_exact(2)
                                    .map(|chunk| {
                                        (u16::from_le_bytes([chunk[0], chunk[1]]) / 257) as u8
                                    })
                                    .collect();
                                (pixels, Format::R8_SRGB)
                            } else {
                                (texture.pixels.clone(), Format::R16_UNORM)
                            }
                        }
                        TextureFormat::R8G8B8A8 => (
                            texture.pixels.clone(),
                            if is_srgb {
                                Format::R8G8B8A8_SRGB
                            } else {
                                Format::R8G8B8A8_UNORM
                            },
                        ),
                        TextureFormat::R8G8B8 => {
                            let pixels = DynamicImage::ImageRgb8(
                                ImageBuffer::from_raw(
                                    texture.width,
                                    texture.height,
                                    texture.pixels.clone(),
                                )
                                .unwrap(),
                            )
                            .to_rgba8()
                            .into_raw();

                            (
                                pixels,
                                if is_srgb {
                                    Format::R8G8B8A8_SRGB
                                } else {
                                    Format::R8G8B8A8_UNORM
                                },
                            )
                        }
                        TextureFormat::R8 => (
                            texture.pixels.clone(),
                            if is_srgb {
                                Format::R8_SRGB
                            } else {
                                Format::R8_UNORM
                            },
                        ),
                        TextureFormat::R16G16B16A16 => {
                            if is_srgb {
                                let pixels = texture
                                    .pixels
                                    .chunks_exact(2)
                                    .map(|chunk| {
                                        (u16::from_le_bytes([chunk[0], chunk[1]]) / 257) as u8
                                    })
                                    .collect();
                                (pixels, Format::R8G8B8A8_SRGB)
                            } else {
                                (texture.pixels.clone(), Format::R16G16B16A16_UNORM)
                            }
                        }
                        TextureFormat::R16G16B16 => {
                            let pixels = texture
                                .pixels
                                .chunks_exact(2)
                                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();
                            let pixels = DynamicImage::ImageRgb16(
                                ImageBuffer::from_raw(texture.width, texture.height, pixels)
                                    .unwrap(),
                            )
                            .to_rgba16()
                            .into_raw();

                            if is_srgb {
                                let pixels = pixels
                                    .into_par_iter()
                                    .map(|p16| (p16 / 257) as u8)
                                    .collect();
                                (pixels, Format::R8G8B8A8_SRGB)
                            } else {
                                let pixels =
                                    pixels.into_par_iter().flat_map(u16::to_le_bytes).collect();
                                (pixels, Format::R16G16B16A16_UNORM)
                            }
                        }
                        _ => panic!("unsupported texture format: {:?}", texture.format),
                    };

                    let extent: [u32; 3] = [texture.width, texture.height, 1];

                    (pixels, extent, format)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(pixels, extent, format)| {
                    upload_image(
                        &mut image_builder,
                        pixels,
                        extent,
                        1,
                        1,
                        &renderer.allocators.memory,
                        &renderer.allocators.command_buffer,
                        renderer.context.graphics_queue(),
                        format,
                    )
                    .unwrap()
                })
                .collect();

            println!(
                "Uploaded {} textures in {:?}",
                textures.len(),
                timer.elapsed()
            );

            textures
        };

        let null_texture = {
            let pixel = vec![0u8, 0, 0, 255]; // RGBA black
            let extent: [u32; 3] = [1, 1, 1]; // 1x1 texture

            upload_image(
                &mut image_builder,
                pixel,
                extent,
                1,
                1,
                &renderer.allocators.memory,
                &renderer.allocators.command_buffer,
                renderer.context.graphics_queue(),
                Format::R8G8B8A8_UNORM,
            )
            .unwrap()
        };

        let skyboxes = {
            let env = "helipad";

            let lambertian = load_ktx2(
                format!("textures/{env}/lambertian/diffuse.ktx2").as_str(),
                &mut image_builder,
                &renderer.allocators.memory,
                &renderer.allocators.command_buffer,
                renderer.context.graphics_queue(),
            )
            .unwrap();
            let ggx = load_ktx2(
                format!("textures/{env}/ggx/specular.ktx2").as_str(),
                &mut image_builder,
                &renderer.allocators.memory,
                &renderer.allocators.command_buffer,
                renderer.context.graphics_queue(),
            )
            .unwrap();

            let charlie = load_ktx2(
                format!("textures/{env}/charlie/sheen.ktx2").as_str(),
                &mut image_builder,
                &renderer.allocators.memory,
                &renderer.allocators.command_buffer,
                renderer.context.graphics_queue(),
            )
            .unwrap();

            // let lut_charlie = load_png(
            //     "textures/lut_charlie.png",
            //     &mut image_builder,
            //     renderer.allocators.memory.clone(),
            //     renderer.allocators.command_buffer.clone(),
            //     renderer.context.graphics_queue().clone(),
            // )
            // .unwrap();

            // let lut_ggx = load_png(
            //     "textures/lut_ggx.png",
            //     &mut image_builder,
            //     renderer.allocators.memory.clone(),
            //     renderer.allocators.command_buffer.clone(),
            //     renderer.context.graphics_queue().clone(),
            // )
            // .unwrap();

            let lut_charlie = {
                let bytes: Vec<_> = include_bytes!("../textures/lut_charlie.bin")
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits)
                    })
                    .collect();
                upload_image(
                    &mut image_builder,
                    bytes,
                    [1024, 1024, 1],
                    1,
                    1,
                    &renderer.allocators.memory,
                    &renderer.allocators.command_buffer,
                    renderer.context.graphics_queue(),
                    Format::R16G16B16A16_SFLOAT,
                )
                .unwrap()
            };

            let lut_ggx = {
                let bytes: Vec<_> = include_bytes!("../textures/lut_ggx.bin")
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits)
                    })
                    .collect();
                upload_image(
                    &mut image_builder,
                    bytes,
                    [1024, 1024, 1],
                    1,
                    1,
                    &renderer.allocators.memory,
                    &renderer.allocators.command_buffer,
                    renderer.context.graphics_queue(),
                    Format::R16G16B16A16_SFLOAT,
                )
                .unwrap()
            };

            let lut_sheen_e = load_png(
                "textures/lut_sheen_E.png",
                &mut image_builder,
                &renderer.allocators.memory,
                &renderer.allocators.command_buffer,
                renderer.context.graphics_queue(),
            )
            .unwrap();

            Skybox {
                lambertian,
                ggx,
                charlie,
                lut_ggx,
                lut_charlie,
                lut_sheen_e,
            }
        };

        image_builder
            .build()
            .unwrap()
            .execute(renderer.context.graphics_queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let clamp_sampler_no_mipmap = Sampler::new(
            renderer.context.device().clone(),
            SamplerCreateInfo {
                address_mode: [ClampToEdge, ClampToEdge, Repeat],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .expect("Couldn't create sampler");

        let mirror_sampler_mipmap = Sampler::new(
            renderer.context.device().clone(),
            SamplerCreateInfo {
                address_mode: [MirroredRepeat, MirroredRepeat, Repeat],
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .expect("Couldn't create sampler");

        let wrap_sampler_mipmap = Sampler::new(
            renderer.context.device().clone(),
            SamplerCreateInfo {
                address_mode: [Repeat; 3],
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .expect("Couldn't create sampler");

        let samplers = Samplers {
            mirror_sampler_mipmap,
            clamp_sampler_no_mipmap,
            wrap_sampler_mipmap,
        };

        let camera = FirstPersonCamera::new();

        let input_state = InputState::default();

        let gui = Gui {
            selected_object: (!objects.is_empty()).then_some(0),
            selected_material: None,
        };

        let mut imgui_ctx = imgui::Context::create();
        imgui_ctx.io_mut().config_flags |=
            imgui::ConfigFlags::DOCKING_ENABLE | imgui::ConfigFlags::VIEWPORTS_ENABLE;

        let scene = Scene::new(objects);

        let last_frame = Instant::now();

        Self {
            renderer,
            vertex_buffer,
            index_buffer,
            scene,
            mesh_infos,
            prim_infos,
            materials,
            mat_prims,
            textures,
            null_texture,
            skyboxes,
            samplers,
            camera,
            input_state,
            gui,
            imgui_ctx,
            last_frame,
            rcx: None,
        }
    }

    fn set_cursor_confinement(&mut self, confined: bool) {
        let window = self.renderer.windows.get_primary_window().unwrap();
        if confined {
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
                .unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
        window.set_cursor_visible(!confined);
        self.input_state.cursor_confined = confined;
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            cursor_confined: false,
        }
    }
}

fn build_cubemap_pipeline(
    device: &Arc<Device>,
    swapchain_format: Format,
    vert: &Arc<ShaderModule>,
    frag: &Arc<ShaderModule>,
    layout: Arc<PipelineLayout>,
) -> Arc<GraphicsPipeline> {
    // First, we load the shaders that the pipeline will use: the vertex shader and the
    // fragment shader.
    //
    // A Vulkan shader can in theory contain multiple entry points, so we have to specify
    // which one.

    let vs = vert.entry_point("main").unwrap();
    let fs = frag.entry_point("main").unwrap();

    // Automatically generate a vertex input state from the vertex shader's input
    // interface, that takes a single vertex buffer containing `Vertex` structs.
    let vertex_input_state = CubemapVertex::per_vertex().definition(&vs).unwrap();

    // Make a list of the shader stages that the pipeline will have.
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let subpass = PipelineRenderingCreateInfo {
        color_attachment_formats: vec![Some(swapchain_format)],
        depth_attachment_format: Some(Format::D32_SFLOAT),
        ..Default::default()
    };

    let depth_stencil_state = DepthStencilState {
        depth: Some(DepthState {
            write_enable: false,
            compare_op: CompareOp::LessOrEqual,
        }),
        ..Default::default()
    };

    let color_blend_state = ColorBlendState::with_attachment_states(
        subpass.color_attachment_formats.len() as u32,
        ColorBlendAttachmentState::default(),
    );

    // Finally, create the pipeline.
    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            // How vertex data is read from the vertex buffers into the vertex shader.
            vertex_input_state: Some(vertex_input_state),
            // How vertices are arranged into primitive shapes. The default primitive shape
            // is a triangle.
            input_assembly_state: Some(InputAssemblyState::default()),
            // How primitives are transformed and clipped to fit the framebuffer. We use a
            // resizable viewport, set to draw over the entire window.
            viewport_state: Some(ViewportState::default()),
            // How polygons are culled and converted into a raster of pixels. The default
            // value does not perform any culling.
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            // How multiple fragment shader samples are converted to a single pixel value.
            // The default value does not perform any multisampling.
            multisample_state: Some(MultisampleState::default()),
            // How pixel values are combined with the values already present in the
            // framebuffer. The default value overwrites the old value with the new one,
            // without any blending.
            color_blend_state: Some(color_blend_state),
            depth_stencil_state: Some(depth_stencil_state),
            // Dynamic states allows us to specify parts of the pipeline settings when
            // recording the command buffer, before we perform drawing. Here, we specify
            // that the viewport should be dynamic.
            dynamic_state: std::iter::once(DynamicState::Viewport).collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn build_material_texture_sets(
    mat: &Material,
    uniform_buffer_allocator: &SubbufferAllocator,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline_layout: &Arc<PipelineLayout>,
    textures: &[Arc<ImageView>],
    null_texture: &Arc<ImageView>,
    sampler: &Arc<Sampler>,
) -> (Arc<DescriptorSet>, Arc<DescriptorSet>) {
    let material_set = {
        let mat_uniform: fs::Material = mat.into();
        let subbuffer = uniform_buffer_allocator.allocate_sized().unwrap();
        *subbuffer.write().unwrap() = mat_uniform;
        WriteDescriptorSet::buffer(0, subbuffer)
    };

    let mat_sampler_set = {
        let mat_samplers: fs::MatSamplers = mat.into();
        let subbuffer = uniform_buffer_allocator.allocate_sized().unwrap();
        *subbuffer.write().unwrap() = mat_samplers;
        WriteDescriptorSet::buffer(1, subbuffer)
    };

    let textures = [
        // (set = 4, binding = 0) u_NormalSampler
        mat.normal_texture.as_ref().map(|t| &t.info),
        // (set = 4, binding = 1) u_EmissiveSampler
        mat.emissive_texture.as_ref(),
        // (set = 4, binding = 2) u_OcclusionSampler
        mat.occlusion_texture.as_ref().map(|t| &t.info),
        // (set = 4, binding = 3) u_BaseColorSampler
        mat.pbr_metallic_roughness.base_color_texture.as_ref(),
        // (set = 4, binding = 4) u_MetallicRoughnessSampler
        mat.pbr_metallic_roughness
            .metallic_roughness_texture
            .as_ref(),
        // (set = 4, binding = 5) u_DiffuseSampler
        mat.pbr_specular_glossiness
            .as_ref()
            .and_then(|t| t.diffuse_texture.as_ref()),
        // (set = 4, binding = 6) u_SpecularGlossinessSampler
        mat.pbr_specular_glossiness
            .as_ref()
            .and_then(|t| t.specular_glossiness_texture.as_ref()),
        // (set = 4, binding = 7) u_ClearcoatSampler
        mat.clearcoat.as_ref().and_then(|c| c.texture.as_ref()),
        // (set = 4, binding = 8) u_ClearcoatRoughnessSampler
        mat.clearcoat
            .as_ref()
            .and_then(|c| c.roughness_texture.as_ref()),
        // (set = 4, binding = 9) u_ClearcoatNormalSampler
        mat.clearcoat
            .as_ref()
            .and_then(|c| c.normal_texture.as_ref())
            .map(|t| &t.info),
        // (set = 4, binding = 10) u_SheenColorSampler
        mat.sheen.as_ref().and_then(|s| s.color_texture.as_ref()),
        // (set = 4, binding = 11) u_SheenRoughnessSampler
        mat.sheen
            .as_ref()
            .and_then(|s| s.roughness_texture.as_ref()),
        // (set = 4, binding = 12) u_SpecularSampler
        mat.specular.as_ref().and_then(|s| s.texture.as_ref()),
        // (set = 4, binding = 13) u_SpecularColorSampler
        mat.specular.as_ref().and_then(|s| s.color_texture.as_ref()),
        // (set = 4, binding = 14) u_TransmissionSampler
        mat.transmission.as_ref().and_then(|t| t.texture.as_ref()),
        // (set = 4, binding = 15) u_ThicknessSampler
        mat.volume
            .as_ref()
            .and_then(|v| v.thickness_texture.as_ref()),
        // (set = 4, binding = 16) u_IridescenceSampler
        mat.iridescence.as_ref().and_then(|i| i.texture.as_ref()),
        // (set = 4, binding = 17) u_IridescenceThicknessSampler
        mat.iridescence
            .as_ref()
            .and_then(|i| i.thickness_texture.as_ref()),
        // (set = 4, binding = 18) u_DiffuseTransmissionSampler
        mat.diffuse_transmission
            .as_ref()
            .and_then(|d| d.texture.as_ref()),
        // (set = 4, binding = 19) u_DiffuseTransmissionColorSampler
        mat.diffuse_transmission
            .as_ref()
            .and_then(|d| d.color_texture.as_ref()),
        // (set = 4, binding = 20) u_AnisotropySampler
        mat.anisotropy.as_ref().and_then(|a| a.texture.as_ref()),
    ]
    .map(|opt_t| opt_t.map(|t| &textures[t.texture.index]));

    let material_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        pipeline_layout.set_layouts().get(3).unwrap().clone(),
        [material_set, mat_sampler_set],
        [],
    )
    .unwrap();

    let texture_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        pipeline_layout.set_layouts().get(4).unwrap().clone(),
        textures
            .into_iter()
            .map(|t| t.unwrap_or(null_texture))
            .enumerate()
            .map(|(idx, texture)| {
                WriteDescriptorSet::image_view_sampler(idx as u32, texture.clone(), sampler.clone())
            }),
        [],
    )
    .unwrap();

    (material_set, texture_set)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.renderer.windows.primary_window_id() {
            self.renderer.windows.remove_renderer(primary_window_id);
        }

        let window_id = self.renderer.windows.create_window(
            event_loop,
            &self.renderer.context,
            &WindowDescriptor {
                present_mode: PresentMode::Immediate,
                width: 1920.,
                height: 1080.,
                scale_factor_override: Some(1.0),
                ..Default::default()
            },
            |info| {
                // Framebuffer needs TRANSFER_SRC so that it can be blitted onto the TransmissionFramebufferSampler
                info.image_usage |= ImageUsage::TRANSFER_SRC;
            },
        );

        let new_rcx = RendererContext::new(&mut self.renderer, window_id);

        let window_renderer = self.renderer.windows.get_primary_renderer_mut().unwrap();

        let window_size = window_renderer.window().inner_size();

        let mut imgui_platform = WinitPlatform::new(&mut self.imgui_ctx);
        let imgui_io = self.imgui_ctx.io_mut();

        imgui_platform.attach_window(imgui_io, window_renderer.window(), HiDpiMode::Default);

        let imgui_renderer = imgui_vulkano_renderer::Renderer::init(
            &mut self.imgui_ctx,
            self.renderer.context.device().clone(),
            self.renderer.context.graphics_queue().clone(),
            window_renderer.swapchain_format(),
            None,
            None,
        )
        .unwrap();

        let mip_levels = max_mip_levels([window_size.width, window_size.height, 1]);
        let intermediate_image = Image::new(
            self.renderer.allocators.memory.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: window_renderer.swapchain_format(),
                extent: [window_size.width, window_size.height, 1],
                mip_levels,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let intermediate_image_view = ImageView::new_default(intermediate_image).unwrap();

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let frame_times = {
            let mut v = VecDeque::new();
            v.push_back(Instant::now());
            v
        };

        let mat_prim_map = {
            let mut mat_prim_map = vec![Vec::new(); self.materials.len()];
            for (prim_idx, prim) in self.prim_infos.iter().enumerate() {
                let mat_idx = prim.mat_idx;
                mat_prim_map[mat_idx].push(prim_idx);
            }
            mat_prim_map
        };

        let const_set = {
            #[allow(clippy::useless_conversion)]
            let uniform = fs::Constants {
                u_MipCount: min(
                    self.skyboxes.ggx.image().mip_levels() as i32,
                    self.skyboxes.charlie.image().mip_levels() as i32,
                )
                .into(),
                u_FramebufferMipCount: (max_mip_levels([window_size.width, window_size.height, 1])
                    as i32)
                    .into(),
                u_EnvRotation: [
                    Padded([0f32, 0., -1.]),
                    Padded([0., 1., 0.]),
                    Padded([1., 0., 0.]),
                ]
                .into(),
                u_EnvIntensity: 1.0.into(),
                u_Exposure: 1.0.into(),
            };
            let subbuffer = self
                .renderer
                .allocators
                .uniform_buffer
                .allocate_sized()
                .unwrap();
            *subbuffer.write().unwrap() = uniform;

            WriteDescriptorSet::buffer(0, subbuffer)
        };

        let skybox_set = {
            [
                (
                    &self.skyboxes.lambertian,
                    &self.samplers.wrap_sampler_mipmap,
                ),
                (&self.skyboxes.ggx, &self.samplers.wrap_sampler_mipmap),
                (&self.skyboxes.charlie, &self.samplers.wrap_sampler_mipmap),
                (
                    &self.skyboxes.lut_ggx,
                    &self.samplers.clamp_sampler_no_mipmap,
                ),
                (
                    &self.skyboxes.lut_charlie,
                    &self.samplers.clamp_sampler_no_mipmap,
                ),
                (
                    &self.skyboxes.lut_sheen_e,
                    &self.samplers.clamp_sampler_no_mipmap,
                ),
            ]
            .into_iter()
            .enumerate()
            .map(|(idx, (texture, sampler))| {
                WriteDescriptorSet::image_view_sampler(idx as u32, texture.clone(), sampler.clone())
            })
        };

        let swapchain_format = window_renderer.swapchain_format();

        let vertex_shader = vs::load(self.renderer.context.device().clone()).unwrap();
        let fragment_shader = fs::load(self.renderer.context.device().clone()).unwrap();

        self.materials
            .iter()
            .enumerate()
            .for_each(|(mat_idx, mat)| {
                let (material_set, texture_set) = build_material_texture_sets(
                    mat,
                    &self.renderer.allocators.uniform_buffer,
                    &self.renderer.allocators.descriptor_set,
                    &new_rcx.pipeline_layout,
                    &self.textures,
                    &self.null_texture,
                    &self.samplers.wrap_sampler_mipmap,
                );

                for prim_idx in mat_prim_map[mat_idx].iter().copied() {
                    let material_constants = mat.into();
                    let prim = &self.prim_infos[prim_idx];

                    self.renderer.add_pipeline(
                        material_constants,
                        prim.spec_const,
                        new_rcx.pipeline_layout.clone(),
                        vertex_shader.clone(),
                        fragment_shader.clone(),
                        material_set.clone(),
                        texture_set.clone(),
                        mat_idx,
                        prim_idx,
                    );
                }
            });

        let const_set = DescriptorSet::new(
            self.renderer.allocators.descriptor_set.clone(),
            #[allow(clippy::get_first)]
            new_rcx
                .pipeline_layout
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [const_set],
            [],
        )
        .unwrap();
        let skybox_set = DescriptorSet::new(
            self.renderer.allocators.descriptor_set.clone(),
            new_rcx
                .pipeline_layout
                .set_layouts()
                .get(2)
                .unwrap()
                .clone(),
            skybox_set.clone(),
            [],
        )
        .unwrap();

        // Create descriptor set for the intermediate image (set 5)
        let framebuffer_set = {
            let layout = new_rcx
                .pipeline_layout
                .set_layouts()
                .get(5)
                .unwrap()
                .clone();
            let write_set = WriteDescriptorSet::image_view_sampler(
                0,
                intermediate_image_view.clone(),
                self.samplers.mirror_sampler_mipmap.clone(),
            );
            DescriptorSet::new(
                self.renderer.allocators.descriptor_set.clone(),
                layout,
                [write_set],
                [],
            )
            .unwrap()
        };

        let cubemap_pipeline = {
            let cubemap_layout = {
                let bindings = [
                    // set = 0
                    vec![
                        DescriptorType::UniformBuffer, // binding = 0 uniform Constants
                    ],
                    // set = 1
                    vec![
                        DescriptorType::UniformBuffer, // binding = 0 uniform Camera
                    ],
                    // set = 2
                    vec![DescriptorType::CombinedImageSampler; 6], // IBL Samplers
                ];

                PipelineLayout::new(
                    self.renderer.context.device().clone(),
                    PipelineLayoutCreateInfo {
                        set_layouts: bindings
                            .into_iter()
                            .map(|set| {
                                DescriptorSetLayout::new(
                                    self.renderer.context.device().clone(),
                                    DescriptorSetLayoutCreateInfo {
                                        bindings: set
                                            .into_iter()
                                            .enumerate()
                                            .map(|(idx, binding)| {
                                                (
                                                    idx as u32,
                                                    (&DescriptorBindingRequirements {
                                                        descriptor_types: vec![binding],
                                                        descriptor_count: Some(1),
                                                        stages: ShaderStages::all_graphics(),
                                                        ..Default::default()
                                                    })
                                                        .into(),
                                                )
                                            })
                                            .collect(),
                                        ..Default::default()
                                    },
                                )
                                .unwrap()
                            })
                            .collect::<Vec<_>>(),
                        ..Default::default()
                    },
                )
                .unwrap()
            };
            let vs = cubemap_vs::load(self.renderer.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = cubemap_fs::load(self.renderer.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let color_blend_state = ColorBlendState::with_attachment_states(1, Default::default());
            let depth_stencil_state = DepthStencilState::default();

            renderer::build_pipeline::<CubemapVertex>(
                self.renderer.context.device().clone(),
                swapchain_format,
                cubemap_layout,
                vs,
                fs,
                color_blend_state,
                depth_stencil_state,
                CullMode::None,
            )
        };

        let (cubemap_vertex_buffer, cubemap_index_buffer, cubemap_index_count) = {
            let vertices: Vec<CubemapVertex> = [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
            .into_iter()
            .map(Into::into)
            .collect();

            let indices: Vec<u32> = [
                1, 2, 0, 2, 3, 0, 6, 2, 1, 1, 5, 6, 6, 5, 4, 4, 7, 6, 6, 3, 2, 7, 3, 6, 3, 7, 0, 7,
                4, 0, 5, 1, 0, 4, 5, 0u32,
            ]
            .into_iter()
            .collect();

            let index_count = indices.len() as u32;

            let vertex_buffer = create_buffer(
                self.renderer.allocators.memory.clone(),
                self.renderer.allocators.command_buffer.clone(),
                self.renderer.context.graphics_queue(),
                BufferUsage::VERTEX_BUFFER,
                vertices,
            );

            let index_buffer = create_buffer(
                self.renderer.allocators.memory.clone(),
                self.renderer.allocators.command_buffer.clone(),
                self.renderer.context.graphics_queue(),
                BufferUsage::INDEX_BUFFER,
                indices,
            );

            (vertex_buffer, index_buffer, index_count)
        };

        /*
        println!("Rendering {} material variants:", pipelines.len());
        for (idx, (k, v)) in pipeline_map.iter().enumerate() {
            println!(" {}) {}", idx + 1, k);
            println!("\tWith {} object variants:", v.keys().len());
            for (idx, k) in v.keys().enumerate() {
                println!("\t {}) {}", idx + 1, k);
            }
        }
         */

        self.renderer.rcx.replace(new_rcx);

        self.rcx.replace(RenderContext {
            cubemap_pipeline,
            cubemap_index_buffer,
            cubemap_vertex_buffer,
            cubemap_index_count,
            intermediate_image_view,
            const_set,
            skybox_set,
            framebuffer_set,
            viewport,
            frame_times,
            imgui_platform,
            imgui_renderer,
        });
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        #[allow(clippy::single_match)]
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.input_state.cursor_confined {
                    self.input_state.mouse_dx += dx;
                    self.input_state.mouse_dy += dy;
                }
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_renderer = self.renderer.windows.get_primary_renderer_mut().unwrap();

        let rcx = self.rcx.as_mut().unwrap();
        let new_rcx = self.renderer.rcx.as_mut().unwrap();

        let imgui_io = self.imgui_ctx.io_mut();
        rcx.imgui_platform.handle_event::<()>(
            imgui_io,
            window_renderer.window(),
            &Event::WindowEvent {
                window_id,
                event: event.clone(),
            },
        );

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => window_renderer.resize(),
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                ..
            } => {
                if imgui_io.want_capture_mouse {
                    return;
                }
                self.set_cursor_confinement(true);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if imgui_io.want_capture_keyboard {
                    return;
                }
                if self.input_state.cursor_confined {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        let pressed = match event.state {
                            ElementState::Pressed => true,
                            ElementState::Released => false,
                        };
                        match code {
                            KeyCode::Escape => self.set_cursor_confinement(false),
                            KeyCode::KeyW => self.input_state.forward = pressed,
                            KeyCode::KeyA => self.input_state.left = pressed,
                            KeyCode::KeyS => self.input_state.backward = pressed,
                            KeyCode::KeyD => self.input_state.right = pressed,
                            KeyCode::KeyQ => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let window_size = window_renderer.window().inner_size();

                let frame_start = Instant::now();
                let delta_t = frame_start
                    .duration_since(*rcx.frame_times.back().unwrap())
                    .as_secs_f32();

                let total_time = frame_start.duration_since(*rcx.frame_times.front().unwrap());
                if total_time.as_secs_f32() >= 1.0 {
                    let frame_time = total_time / rcx.frame_times.len() as u32;
                    let fps = rcx.frame_times.len() as f64 / total_time.as_secs_f64();
                    println!("{} fps ({:.2?})", fps as u32, frame_time);
                    rcx.frame_times.clear();
                }
                rcx.frame_times.push_back(frame_start);

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                let swapchain_format = window_renderer.swapchain_format();

                // Begin rendering by acquiring the gpu future from the window renderer.
                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                        // Whenever the window resizes we need to recreate everything dependent
                        // on the window size. In this example that
                        // includes the swapchain, the framebuffers
                        // and the dynamic state viewport.
                        new_rcx.attachment_image_views = swapchain_images
                            .iter()
                            .map(|image| ImageView::new_default(image.image().clone()).unwrap())
                            .collect();

                        let mip_levels = max_mip_levels([window_size.width, window_size.height, 1]);
                        let intermediate_image = Image::new(
                            self.renderer.allocators.memory.clone(),
                            ImageCreateInfo {
                                image_type: ImageType::Dim2d,
                                format: swapchain_format,
                                extent: [window_size.width, window_size.height, 1],
                                mip_levels,
                                usage: ImageUsage::COLOR_ATTACHMENT
                                    | ImageUsage::SAMPLED
                                    | ImageUsage::TRANSFER_SRC
                                    | ImageUsage::TRANSFER_DST,
                                ..Default::default()
                            },
                            AllocationCreateInfo::default(),
                        )
                        .expect("Failed to create intermediate image");

                        rcx.intermediate_image_view = ImageView::new_default(intermediate_image)
                            .expect("Failed to create intermediate image view");

                        // Update framebuffer descriptor set with new intermediate image
                        rcx.framebuffer_set = {
                            let layout = new_rcx
                                .pipeline_layout
                                .set_layouts()
                                .get(5)
                                .unwrap()
                                .clone();
                            let write_set = WriteDescriptorSet::image_view_sampler(
                                0,
                                rcx.intermediate_image_view.clone(),
                                self.samplers.mirror_sampler_mipmap.clone(),
                            );
                            DescriptorSet::new(
                                self.renderer.allocators.descriptor_set.clone(),
                                layout,
                                [write_set],
                                [],
                            )
                            .unwrap()
                        };

                        let depth_image = Image::new(
                            self.renderer.allocators.memory.clone(),
                            ImageCreateInfo {
                                image_type: ImageType::Dim2d,
                                format: Format::D32_SFLOAT,
                                extent: [window_size.width, window_size.height, 1],
                                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                                ..Default::default()
                            },
                            AllocationCreateInfo::default(),
                        )
                        .expect("Failed to create depth image");

                        new_rcx.depth_image_view = ImageView::new_default(depth_image)
                            .expect("Failed to create depth image view");
                        rcx.viewport.extent = window_size.into();
                    })
                    .unwrap();

                if self.input_state.forward {
                    self.camera.move_forward(delta_t);
                }
                if self.input_state.backward {
                    self.camera.move_backward(delta_t);
                }
                if self.input_state.left {
                    self.camera.move_left(delta_t);
                }
                if self.input_state.right {
                    self.camera.move_right(delta_t);
                }
                if self.input_state.mouse_dy != 0.0 || self.input_state.mouse_dx != 0.0 {
                    self.camera.rotate(
                        self.input_state.mouse_dx as f32,
                        self.input_state.mouse_dy as f32,
                    );
                }
                self.input_state.mouse_dx = 0.0;
                self.input_state.mouse_dy = 0.0;

                // In order to draw, we have to record a *command buffer*. The command buffer
                // object holds the list of commands that are going to be executed.
                //
                // Recording a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.renderer.allocators.command_buffer.clone(),
                    self.renderer.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let mut opaque_objects = Vec::<usize>::new();
                let mut translucent_objects = Vec::<usize>::new();
                let mut transmissive_objects = Vec::<usize>::new();

                for (mat_specs, obj_pipelines) in &self.renderer.pipeline_manager.pipeline_map {
                    match mat_specs.render_type() {
                        RenderType::Opaque => opaque_objects.extend(obj_pipelines.values()),
                        RenderType::Translucent => {
                            translucent_objects.extend(obj_pipelines.values());
                        }
                        RenderType::Transmissive => {
                            transmissive_objects.extend(obj_pipelines.values());
                        }
                    }
                }

                let mut translucent_sorted = Vec::new();
                for pcx_idx in translucent_objects.iter().copied() {
                    let pcx = &self.renderer.pipeline_manager.pipelines[pcx_idx];
                    for prim_idx in pcx.prim_indices.iter().copied() {
                        let prim = &self.prim_infos[prim_idx];
                        for obj_idx in prim.object_ids.iter().copied() {
                            let object = &self.scene.objects[obj_idx];
                            if !object.enabled {
                                continue;
                            }
                            let transform = object.transform;
                            let dist = self
                                .camera
                                .position
                                .to_vec()
                                .distance2(transform.w.truncate());
                            translucent_sorted.push((dist, obj_idx, prim_idx, pcx_idx));
                        }
                    }
                }
                translucent_sorted.sort_by(|(a, ..), (b, ..)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });

                let cam_set = {
                    let aspect_ratio = window_size.width as f32 / window_size.height as f32;
                    let proj = self.camera.projection_matrix(aspect_ratio);
                    let view = self.camera.view_matrix();
                    let position = self.camera.position;
                    let view_proj = proj * view;

                    let cam_uniform = fs::Camera {
                        u_ViewMatrix: view.into(),
                        u_ProjectionMatrix: proj.into(),
                        u_ViewProjectionMatrix: view_proj.into(),
                        u_Camera: position.into(),
                    };

                    let subbuffer = self
                        .renderer
                        .allocators
                        .uniform_buffer
                        .allocate_sized()
                        .unwrap();
                    *subbuffer.write().unwrap() = cam_uniform;
                    let write_set = WriteDescriptorSet::buffer(0, subbuffer);

                    DescriptorSet::new(
                        self.renderer.allocators.descriptor_set.clone(),
                        new_rcx
                            .pipeline_layout
                            .set_layouts()
                            .get(1)
                            .unwrap()
                            .clone(),
                        [write_set],
                        [],
                    )
                    .unwrap()
                };

                builder
                    .set_viewport(0, std::iter::once(rcx.viewport.clone()).collect())
                    .unwrap();

                builder
                    .begin_rendering(RenderingInfo {
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some(ClearValue::Float([1.0, 0.0, 1.0, 1.0])),
                            ..RenderingAttachmentInfo::image_view(
                                new_rcx.attachment_image_views
                                    [window_renderer.image_index() as usize]
                                    .clone(),
                            )
                        })],
                        depth_attachment: Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some(1.0f32.into()),
                            ..RenderingAttachmentInfo::image_view(new_rcx.depth_image_view.clone())
                        }),
                        ..Default::default()
                    })
                    .unwrap();

                {
                    builder
                        .bind_pipeline_graphics(rcx.cubemap_pipeline.clone())
                        .unwrap();

                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.cubemap_pipeline.layout().clone(),
                            0,
                            rcx.const_set.clone(),
                        )
                        .unwrap();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.cubemap_pipeline.layout().clone(),
                            1,
                            cam_set.clone(),
                        )
                        .unwrap();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.cubemap_pipeline.layout().clone(),
                            2,
                            rcx.skybox_set.clone(),
                        )
                        .unwrap();

                    builder
                        .bind_index_buffer(rcx.cubemap_index_buffer.clone())
                        .unwrap()
                        .bind_vertex_buffers(0, rcx.cubemap_vertex_buffer.clone())
                        .unwrap();

                    unsafe { builder.draw_indexed(rcx.cubemap_index_count, 1, 0, 0, 0) }.unwrap();
                }

                builder
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.index_buffer.clone())
                    .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        new_rcx.pipeline_layout.clone(),
                        0,
                        rcx.const_set.clone(),
                    )
                    .unwrap();
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        new_rcx.pipeline_layout.clone(),
                        1,
                        cam_set,
                    )
                    .unwrap();
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        new_rcx.pipeline_layout.clone(),
                        2,
                        rcx.skybox_set.clone(),
                    )
                    .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        new_rcx.pipeline_layout.clone(),
                        5,
                        rcx.framebuffer_set.clone(),
                    )
                    .unwrap();

                // Render opaque geometry
                if !opaque_objects.is_empty() {
                    for pcx_idx in opaque_objects {
                        self.renderer.pipeline_manager.pipelines[pcx_idx].render(
                            &mut builder,
                            &new_rcx.pipeline_layout,
                            &self.prim_infos,
                            &self.scene.objects,
                        );
                    }
                }

                if !translucent_sorted.is_empty() {
                    // Render translucent geometry
                    let mut curr_pcx = None;
                    for (_, obj_idx, prim_idx, pcx_idx) in translucent_sorted {
                        let prim = &self.prim_infos[prim_idx];
                        let pcx = &self.renderer.pipeline_manager.pipelines[pcx_idx];
                        if curr_pcx != Some(pcx_idx) {
                            builder
                                .bind_pipeline_graphics(pcx.pipeline.clone())
                                .unwrap();
                            curr_pcx.replace(pcx_idx);
                        }

                        let (mat_set, tex_set) = pcx.material_sets[&prim.mat_idx].clone();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                new_rcx.pipeline_layout.clone(),
                                3,
                                mat_set,
                            )
                            .unwrap();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                new_rcx.pipeline_layout.clone(),
                                4,
                                tex_set,
                            )
                            .unwrap();

                        let object = &self.scene.objects[obj_idx];
                        let transform = object.transform;
                        let normal = transform
                            .transpose()
                            .invert()
                            .unwrap_or_else(Matrix4::identity);
                        let model: [[f32; 4]; 4] = transform.into();
                        #[allow(clippy::useless_conversion)]
                        let data = fs::Object {
                            u_ModelMatrix: model.into(),
                            u_NormalMatrix: normal.into(),
                        };
                        builder
                            .push_constants(new_rcx.pipeline_layout.clone(), 0, data)
                            .unwrap();
                        unsafe {
                            // We add a draw command.
                            builder.draw_indexed(
                                prim.index_count,
                                1,
                                prim.index_offset,
                                prim.vertex_offset,
                                0,
                            )
                        }
                        .unwrap();
                    }
                }

                builder
                    // We leave the render pass.
                    .end_rendering()
                    .unwrap();

                if !transmissive_objects.is_empty() {
                    // Render transmissive geometry
                    let src_image = new_rcx.attachment_image_views
                        [window_renderer.image_index() as usize]
                        .image();
                    let dst_image = rcx.intermediate_image_view.image();
                    builder
                        .blit_image(BlitImageInfo::images(src_image.clone(), dst_image.clone()))
                        .unwrap();
                    let dimensions = src_image.extent();
                    let mip_levels = max_mip_levels(dimensions);
                    for mip_level in 1..mip_levels {
                        let regions = [ImageBlit {
                            src_subresource: ImageSubresourceLayers {
                                aspects: dst_image.format().aspects(),
                                mip_level: mip_level - 1,
                                array_layers: 0..dst_image.array_layers(),
                            },
                            dst_subresource: ImageSubresourceLayers {
                                aspects: dst_image.format().aspects(),
                                mip_level,
                                array_layers: 0..dst_image.array_layers(),
                            },
                            src_offsets: [
                                [0, 0, 0],
                                mip_level_extent(dimensions, mip_level - 1).unwrap(),
                            ],
                            dst_offsets: [
                                [0, 0, 0],
                                mip_level_extent(dimensions, mip_level).unwrap(),
                            ],
                            ..Default::default()
                        }];
                        builder
                            .blit_image(BlitImageInfo {
                                src_image_layout: ImageLayout::General,
                                dst_image_layout: ImageLayout::General,
                                regions: regions.into(),
                                filter: Filter::Linear,
                                ..BlitImageInfo::images(dst_image.clone(), dst_image.clone())
                            })
                            .unwrap();
                    }
                    builder
                        .begin_rendering(RenderingInfo {
                            color_attachments: vec![Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Load, // Load previous contents
                                store_op: AttachmentStoreOp::Store,
                                ..RenderingAttachmentInfo::image_view(
                                    new_rcx.attachment_image_views
                                        [window_renderer.image_index() as usize]
                                        .clone(),
                                )
                            })],
                            depth_attachment: Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Load, // Keep depth buffer
                                store_op: AttachmentStoreOp::Store,
                                ..RenderingAttachmentInfo::image_view(
                                    new_rcx.depth_image_view.clone(),
                                )
                            }),
                            ..Default::default()
                        })
                        .unwrap();

                    for pcx_idx in transmissive_objects {
                        self.renderer.pipeline_manager.pipelines[pcx_idx].render(
                            &mut builder,
                            &new_rcx.pipeline_layout,
                            &self.prim_infos,
                            &self.scene.objects,
                        );
                    }
                    builder
                        // We leave the render pass.
                        .end_rendering()
                        .unwrap();
                }

                let ui = self.imgui_ctx.frame();

                let viewport_dockspace_id = unsafe {
                    igDockSpaceOverViewport(
                        igGetMainViewport(),
                        ImGuiDockNodeFlags_PassthruCentralNode as i32,
                        null(),
                    )
                };

                unsafe {
                    use imgui::sys::{
                        igDockBuilderDockWindow, igDockBuilderRemoveNodeChildNodes,
                        igDockBuilderSplitNode, ImGuiDir_Left, ImGuiDir_Up,
                    };
                    static mut INIT: bool = true;
                    if INIT {
                        INIT = false;

                        igDockBuilderRemoveNodeChildNodes(viewport_dockspace_id);

                        let mut left = 0;
                        let mut rest = 0;
                        igDockBuilderSplitNode(
                            viewport_dockspace_id,
                            ImGuiDir_Left,
                            0.2,
                            &mut left,
                            &mut rest,
                        );

                        let mut top_left = 0;
                        let mut bottom_left = 0;
                        igDockBuilderSplitNode(
                            left,
                            ImGuiDir_Up,
                            0.75,
                            &mut top_left,
                            &mut bottom_left,
                        );

                        igDockBuilderDockWindow(c_str!("Objects").as_ptr(), top_left);
                        igDockBuilderDockWindow(c_str!("Materials").as_ptr(), bottom_left);
                        igDockBuilderDockWindow(c_str!("Transform").as_ptr(), bottom_left);
                    }
                }

                ui.window("Objects").build(|| {
                    fn object_tree_builder(
                        idx: usize,
                        ui: &imgui::Ui,
                        scene: &mut Scene,
                        selected: &mut Option<usize>,
                    ) {
                        let object = &scene.objects[idx];
                        let name = object.name.as_deref().unwrap_or("Unnamed");

                        let builder = ui
                            .tree_node_config(format!("{name}##object{idx}"))
                            .opened(true, Condition::Once)
                            .framed(true)
                            .allow_item_overlap(true) // Make the checkbox clickable when overlaid
                            .open_on_arrow(true)
                            .leaf(scene.children[idx].is_empty());

                        let token = {
                            let _color_token = (*selected == Some(idx)).then(|| {
                                ui.push_style_color(
                                    StyleColor::Header,
                                    ui.style_color(StyleColor::HeaderActive),
                                )
                            });
                            builder.push()
                        };

                        if ui.is_item_clicked() {
                            selected.replace(idx);
                        }

                        if selected.is_some_and(|s| s == idx)
                            && ui.is_item_clicked_with_button(imgui::MouseButton::Right)
                        {
                            selected.take();
                        }

                        if let Some(_token) = ui
                            .drag_drop_source_config("OBJECT_TREE_DRAG")
                            .flags(DragDropFlags::empty())
                            .begin_payload(idx)
                        {
                            object_tree_builder(idx, ui, scene, selected);
                        }

                        if let Some(target) = ui.drag_drop_target() {
                            if let Some(payload) = target.accept_payload::<usize, &str>(
                                "OBJECT_TREE_DRAG",
                                DragDropFlags::empty(),
                            ) {
                                scene.change_parent(payload.unwrap().data, Some(idx));
                            }
                        }

                        let checkbox_size = ui.frame_height();
                        let pos = ui.content_region_max()[0] - checkbox_size;
                        ui.same_line_with_pos(pos);
                        ui.checkbox(format!("##checkbox{idx}"), &mut scene.objects[idx].enabled);

                        if let Some(_token) = token {
                            let mut children: Vec<_> =
                                scene.children[idx].clone().into_iter().collect();
                            children.sort_unstable();
                            for child in children.iter().copied() {
                                object_tree_builder(child, ui, scene, selected);
                            }
                        }
                    }

                    let roots: Vec<_> = self
                        .scene
                        .objects
                        .iter()
                        .filter(|o| o.parent.is_none())
                        .map(|r| r.index)
                        .collect();

                    for root in roots {
                        object_tree_builder(
                            root,
                            ui,
                            &mut self.scene,
                            &mut self.gui.selected_object,
                        );
                    }

                    ui.invisible_button("root_drop_region", ui.content_region_avail());
                    if let Some(target) = ui.drag_drop_target() {
                        if let Some(payload) = target.accept_payload::<usize, &str>(
                            "OBJECT_TREE_DRAG",
                            DragDropFlags::empty(),
                        ) {
                            let from = payload.unwrap().data;
                            self.scene.change_parent(from, None);
                        }
                    }
                });

                if let Some(selected) = self.gui.selected_object {
                    let object = &mut self.scene.objects[selected];
                    if let Some(transform) = gui::object_transform(object, ui) {
                        object.local_transform = transform;
                        self.scene.regenerate();
                    }

                    // Note: need to repeat this line to avoid lifetime issues with the call to Scene::regenerate() above.
                    let object = &mut self.scene.objects[selected];
                    if let Some(mesh_idx) = object.mesh_idx {
                        let mesh = &self.mesh_infos[mesh_idx];
                        let mat_ids: Vec<_> = self
                            .prim_infos
                            .iter()
                            .skip(mesh.prims_offset)
                            .take(mesh.prims_count)
                            .map(|prim| prim.mat_idx)
                            .collect();
                        gui::mesh_materials(
                            &mat_ids,
                            &self.materials,
                            &mut self.gui.selected_material,
                            &ui,
                        );
                    }
                }

                if let Some(mat_idx) = self.gui.selected_material {
                    let mat = &mut self.materials[mat_idx];
                    let old_mat_spec: MaterialSpecializationConstants = (&*mat).into();

                    let changed = ui.window("Material").build(|| {
                        let mut name = mat.name.clone().unwrap_or_default();
                        if ui.input_text("Name", &mut name)
                            .build() {
                            if name.is_empty() {
                                mat.name.take();
                            } else {
                                mat.name.replace(name);
                            }
                        }
                        let mut changed = false;
                        if ui.color_edit4("Base Color Factor", &mut mat.pbr_metallic_roughness.base_color_factor) { changed = true; }
                        // TODO: base_color_texture
                        if ui.slider("Metallic Factor", 0.0, 1.0, &mut mat.pbr_metallic_roughness.metallic_factor) { changed = true; }
                        if ui.slider("Roughness Factor", 0.0, 1.0, &mut mat.pbr_metallic_roughness.roughness_factor) { changed = true; }
                        // TODO: metallic_roughness_texture
                        // TODO: normal_texture
                        // TODO: occlusion_texture
                        // TODO: emissive_texture
                        if ui.color_edit3("Emissive Factor", &mut mat.emissive_factor) { changed = true; }

                        {
                            let choices = ["Opaque", "Mask", "Blend"];
                            let mut selected = match mat.alpha_mode {
                                AlphaMode::Opaque => 0,
                                AlphaMode::Mask => 1,
                                AlphaMode::Blend => 2,
                            };
                            if ui.combo_simple_string("Alpha Mode", &mut selected, &choices) {
                                mat.alpha_mode = match selected {
                                    1 => AlphaMode::Mask,
                                    2 => AlphaMode::Blend,
                                    _ => AlphaMode::Opaque,
                                };
                                changed = true;
                            }
                        }

                        if ui.slider("Alpha Cutoff", 0.0, 1.0, &mut mat.alpha_cutoff.0) { changed = true; }
                        if ui.checkbox("Double Sided", &mut mat.double_sided) { changed = true; }
                        if ui.checkbox("Unlit", &mut mat.unlit) { changed = true; }

                        changed |= gui::material_property(
                            &mut mat.pbr_specular_glossiness,
                            "PBR Specular Glossiness",
                            ui,
                            |spec_gloss| {
                                ui.color_edit4("Diffuse Factor", &mut spec_gloss.diffuse_factor) |
                                    // TODO: diffuse_texture
                                    ui.color_edit3("Specular Factor", &mut spec_gloss.specular_factor) |
                                    ui.slider("Glossiness Factor", 0.0, 1.0, &mut spec_gloss.glossiness_factor)
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.anisotropy,
                            "Anisotropy",
                            ui,
                            |anisotropy| {
                                ui.slider("Anisotropy Strength", 0.0, 1.0, &mut anisotropy.strength) |
                                    ui.slider("Anisotropy Rotation", 0.0, 2. * PI, &mut anisotropy.rotation)
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.clearcoat,
                            "Clearcoat",
                            ui,
                            |clearcoat| {
                                ui.slider("Clearcoat Factor", 0.0, 1.0, &mut clearcoat.factor) |
                                    // TODO: texture
                                    ui.slider("Clearcoat Roughness factor", 0.0, 1.0, &mut clearcoat.roughness_factor)
                                // TODO: roughness_texture
                                // TODO: normal_texture
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.diffuse_transmission,
                            "Diffuse Transmission",
                            ui,
                            |diffuse_transmission| {
                                ui.slider("Diffuse Transmission Factor", 0.0, 1.0, &mut diffuse_transmission.factor) |
                                    // TODO: texture
                                    ui.color_edit3("Diffuse Transmission Color Factor", &mut diffuse_transmission.color_factor)
                                // TODO: color_texture
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.dispersion,
                            "Dispersion",
                            ui,
                            |dispersion| {
                                gui::drag_float(&mut dispersion.0, 0.0, f32::infinity(), "Dispersion", "%0.2f")
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.emissive_strength,
                            "Emissive Strength",
                            ui,
                            |emissive_strength| {
                                gui::drag_float(&mut emissive_strength.0, 0.0, f32::infinity(), "Emissive Strength", "%0.2f")
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.ior,
                            "IOR",
                            ui,
                            |ior| {
                                gui::drag_float(&mut ior.0, 1.0, f32::infinity(), "IOR", "%0.2f")
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.iridescence,
                            "Iridescence",
                            ui,
                            |iridescence| {
                                ui.slider("Iridescence Factor", 0.0, 1.0, &mut iridescence.factor) |
                                    // TODO: texture
                                    gui::drag_float(&mut iridescence.thickness_minimum, 0.0, f32::infinity(), "Thickness Minimum", "%0.1f") |
                                    gui::drag_float(&mut iridescence.thickness_maximum, 0.0, f32::infinity(), "Thickness Maximum", "%0.1f")
                                // TODO: thickness texture
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.sheen,
                            "Sheen",
                            ui,
                            |sheen| {
                                ui.color_edit3("Sheen Color Factor", &mut sheen.color_factor) |
                                    // TODO: color texture
                                    ui.slider("Sheen Roughness Factor", 0.0, 1.0, &mut sheen.roughness_factor)
                                // TODO: roughness texture
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.specular,
                            "Specular",
                            ui,
                            |specular| {
                                ui.slider("Specular Factor", 0.0, 1.0, &mut specular.factor) |
                                    // TODO: texture
                                    ui.color_edit3("Specular Color Factor", &mut specular.color_factor)
                                // TODO: color texture
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.transmission,
                            "Transmission",
                            ui,
                            |transmission| {
                                ui.slider("Transmission Factor", 0.0, 1.0, &mut transmission.factor)
                            }
                        );

                        changed |= gui::material_property(
                            &mut mat.volume,
                            "Volume",
                            ui,
                            |volume| {
                                gui::drag_float(&mut volume.thickness_factor, 0.0, f32::infinity(), "Thickness Factor", "%0.2f") |
                                    // TODO: thickness_texture
                                    gui::drag_float(&mut volume.attenuation_distance, 0.0, f32::infinity(), "Attenuation Distance", "%0.2f") |
                                    ui.color_edit3("Attenuation Color", &mut volume.attenuation_color)
                            }
                        );

                        changed
                    });

                    let pipeline_layout = new_rcx.pipeline_layout.clone();

                    if let Some(true) = changed {
                        self.renderer.remove_material_pipeline(
                            old_mat_spec,
                            mat_idx,
                            &self.mat_prims[mat_idx],
                        );

                        let mat_spec: MaterialSpecializationConstants = (&*mat).into();
                        let (material_set, texture_set) = build_material_texture_sets(
                            mat,
                            &self.renderer.allocators.uniform_buffer,
                            &self.renderer.allocators.descriptor_set,
                            &pipeline_layout,
                            &self.textures,
                            &self.null_texture,
                            &self.samplers.wrap_sampler_mipmap,
                        );

                        let vs = vs::load(self.renderer.context.device().clone()).unwrap();
                        let fs = fs::load(self.renderer.context.device().clone()).unwrap();
                        for prim_idx in self.mat_prims[mat_idx].iter().copied() {
                            let prim = &self.prim_infos[prim_idx];
                            self.renderer.add_pipeline(
                                mat_spec,
                                prim.spec_const,
                                pipeline_layout.clone(),
                                vs.clone(),
                                fs.clone(),
                                material_set.clone(),
                                texture_set.clone(),
                                mat_idx,
                                prim_idx,
                            );
                        }
                    }
                }

                let window_renderer = self.renderer.windows.get_primary_renderer_mut().unwrap();
                rcx.imgui_platform
                    .prepare_render(ui, window_renderer.window());

                let draw_data = self.imgui_ctx.render();

                rcx.imgui_renderer
                    .draw_commands(
                        &mut builder,
                        window_renderer.swapchain_image_view(),
                        draw_data,
                    )
                    .unwrap();

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(
                        self.renderer.context.graphics_queue().clone(),
                        command_buffer,
                    )
                    .unwrap()
                    .boxed();

                window_renderer.present(future, true);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.renderer.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();
        rcx.imgui_platform
            .prepare_frame(self.imgui_ctx.io_mut(), window_renderer.window())
            .unwrap();
        window_renderer.window().request_redraw();
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {
        let now = Instant::now();
        self.imgui_ctx
            .io_mut()
            .update_delta_time(now - self.last_frame);
        self.last_frame = now;
    }
}

#[derive(Debug)]
struct PrimitiveDrawInfo {
    pub index_offset: u32,
    pub vertex_offset: i32,
    pub index_count: u32,
    pub mat_idx: usize,
    pub spec_const: ObjectSpecializationConstants,
    pub object_ids: HashSet<usize>,
}

#[derive(Debug)]
struct MeshDrawInfo {
    pub prims_offset: usize,
    pub prims_count: usize,
}

fn create_buffer<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
    allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    usage: BufferUsage,
    data: I,
) -> Subbuffer<[T]>
where
    I::IntoIter: ExactSizeIterator,
{
    let staging_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )
    .unwrap();
    let device_local_buffer = Buffer::new_slice::<T>(
        allocator,
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        staging_buffer.len(),
    )
    .unwrap();
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            staging_buffer,
            device_local_buffer.clone(),
        ))
        .unwrap();
    let command_buffer = builder.build().unwrap();

    sync::now(queue.device().clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    device_local_buffer
}
