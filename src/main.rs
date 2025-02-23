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
use crate::gltf::{load_gltf, CombinedVertex, Gltf, Object, TextureFormat};
use crate::material::Material;
use crate::shader::*;
use cgmath::{Matrix, Matrix4, Rad, SquareMatrix};
use image::{ColorType, DynamicImage, ImageBuffer, ImageReader};
use ktx2::SupercompressionScheme;
use rayon::iter::Either;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::f32::consts::FRAC_PI_4;
use std::time::{Duration, Instant};
use std::{error::Error, sync::Arc};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{
    BufferImageCopy, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::half::f16;
use vulkano::image::sampler::SamplerAddressMode::{ClampToEdge, Repeat};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers, ImageType,
};
use vulkano::padded::Padded;
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::shader::{DescriptorBindingRequirements, ShaderModule, ShaderStages};
use vulkano::swapchain::PresentMode;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::{DeviceFeatures, Queue},
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
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton};
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
mod material;
mod shader;

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

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    vertex_buffer: Subbuffer<[CombinedVertex]>,
    index_buffer: Subbuffer<[u32]>,
    draw_infos: Vec<Vec<PrimitiveDrawInfo>>,
    materials: Vec<Material>,
    objects: Vec<Object>,
    textures: Vec<Arc<ImageView>>,
    null_texture: Arc<ImageView>,
    skyboxes: Skybox,
    sampler: Arc<Sampler>,
    wrap_sampler: Arc<Sampler>,
    camera: FirstPersonCamera,
    input_state: InputState,
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
    attachment_image_views: Vec<Arc<ImageView>>,
    depth_image_view: Arc<ImageView>,
    pipelines: HashMap<
        MaterialSpecializationConstants,
        HashMap<ObjectSpecializationConstants, PipelineContext>,
    >,
    viewport: Viewport,
    frame_times: VecDeque<Instant>,
}

#[allow(clippy::too_many_arguments)]
fn upload_image<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pixels: I,
    extent: [u32; 3],
    mip_levels: u32,
    array_layers: u32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
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
                    format.block_size() * (array_layers * mip_width * mip_height) as DeviceSize;
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
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
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
                Some(scheme) => panic!("Unsupported compression scheme: {:?}", scheme),
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
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        queue,
        Format::R16G16B16A16_SFLOAT,
    )?)
}

fn load_png(
    path: &str,
    image_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
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
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
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
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue,
                Format::R16G16B16A16_UNORM,
            )?)
        }
        _ => panic!("Unsupported color type {:?}", color),
    }
}

impl App {
    fn new() -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                ..Default::default()
            },
            ..Default::default()
        });
        let windows = VulkanoWindows::default();

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            context.device().clone(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let Gltf {
            meshes,
            textures,
            texture_maps,
            materials,
            objects,
        } = load_gltf("models/DamagedHelmet.glb").expect("Couldn't load gltf model");

        let timer = Instant::now();

        let (vert_count, index_count) = meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .map(|prim| (prim.vertices.len(), prim.indices.len()))
            .reduce(|(v1, i1), (v2, i2)| (v1 + v2, i1 + i2))
            .unwrap();

        let (vertex_buffer, index_buffer, mut draw_infos) = {
            let mut combined_verts = Vec::with_capacity(vert_count / 3);
            let mut combined_indices = Vec::with_capacity(index_count);
            let mut draw_infos = Vec::new();

            for mesh in meshes {
                let mut prim_infos = Vec::new();
                for prim in mesh.primitives {
                    let info = PrimitiveDrawInfo {
                        index_offset: combined_indices.len() as u32,
                        vertex_offset: combined_verts.len() as i32,
                        index_count: prim.indices.len() as u32,
                        mat_idx: prim.mat_idx,
                        object_constants: prim.spec_constants,
                        object_ids: Vec::new(),
                    };
                    prim_infos.push(info);

                    combined_verts.extend(prim.vertices);
                    combined_indices.extend(prim.indices);
                }
                draw_infos.push(prim_infos);
            }

            let vertex_buffer = create_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                BufferUsage::VERTEX_BUFFER,
                combined_verts,
            );

            let index_buffer = create_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                BufferUsage::INDEX_BUFFER,
                combined_indices,
            );

            println!("Combined vertex data in {:?}", timer.elapsed());

            (vertex_buffer, index_buffer, draw_infos)
        };

        for (obj_idx, object) in objects.iter().enumerate() {
            for prim in &mut draw_infos[object.mesh_idx] {
                prim.object_ids.push(obj_idx);
            }
        }

        let mut image_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let textures = {
            let timer = Instant::now();

            let textures = texture_maps
                .into_iter()
                .map(|texture_map| {
                    let texture = &textures[texture_map.index];
                    let is_srgb = texture_map
                        .usage
                        .as_ref()
                        .map(Either::is_left)
                        .unwrap_or(false);

                    let (pixels, format) = match texture.format {
                        TextureFormat::R8G8B8A8 => (
                            texture.pixels.clone(),
                            match is_srgb {
                                true => Format::R8G8B8A8_SRGB,
                                false => Format::R8G8B8A8_UNORM,
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
                                match is_srgb {
                                    true => Format::R8G8B8A8_SRGB,
                                    false => Format::R8G8B8A8_UNORM,
                                },
                            )
                        }
                        TextureFormat::R8 => (
                            texture.pixels.clone(),
                            match is_srgb {
                                true => Format::R8_SRGB,
                                false => Format::R8_UNORM,
                            },
                        ),
                        TextureFormat::R16G16B16A16 => (
                            texture.pixels.clone(),
                            match is_srgb {
                                true => Format::R16G16B16A16_SFLOAT,
                                false => Format::R16G16B16A16_UNORM,
                            },
                        ),
                        _ => panic!("unsupported texture format: {:?}", texture.format),
                    };

                    let extent: [u32; 3] = [texture.width, texture.height, 1];

                    upload_image(
                        &mut image_builder,
                        pixels,
                        extent,
                        1,
                        1,
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        context.graphics_queue().clone(),
                        format,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

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
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                Format::R8G8B8A8_UNORM,
            )
            .unwrap()
        };

        let skyboxes = {
            let env = "helipad";

            let lambertian = load_ktx2(
                format!("textures/{env}/lambertian/diffuse.ktx2").as_str(),
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();
            let ggx = load_ktx2(
                format!("textures/{env}/ggx/specular.ktx2").as_str(),
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();

            let charlie = load_ktx2(
                format!("textures/{env}/charlie/sheen.ktx2").as_str(),
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();

            // let lut_charlie = load_png(
            //     "textures/lut_charlie.png",
            //     &mut image_builder,
            //     memory_allocator.clone(),
            //     command_buffer_allocator.clone(),
            //     context.graphics_queue().clone(),
            // )
            //    .unwrap();

            // let lut_ggx = load_png(
            //     "textures/lut_ggx.png",
            //     &mut image_builder,
            //     memory_allocator.clone(),
            //     command_buffer_allocator.clone(),
            //     context.graphics_queue().clone(),
            // )
            // .unwrap();

            let lut_charlie = upload_image(
                &mut image_builder,
                include_bytes!("../textures/lut_charlie.bin")
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits)
                    })
                    .collect::<Vec<_>>(),
                [1024, 1024, 1],
                1,
                1,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap();

            let lut_ggx = upload_image(
                &mut image_builder,
                include_bytes!("../textures/lut_ggx.bin")
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits)
                    })
                    .collect::<Vec<_>>(),
                [1024, 1024, 1],
                1,
                1,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap();

            let lut_sheen_e = load_png(
                "textures/lut_sheen_E.png",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
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
            .execute(context.graphics_queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let sampler = Sampler::new(
            context.device().clone(),
            SamplerCreateInfo {
                address_mode: [ClampToEdge, ClampToEdge, Repeat],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .expect("Couldn't create sampler");

        let wrap_sampler = Sampler::new(
            context.device().clone(),
            SamplerCreateInfo {
                address_mode: [Repeat; 3],
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .expect("Couldn't create sampler");

        let camera = FirstPersonCamera::new();

        App {
            context,
            windows,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            vertex_buffer,
            index_buffer,
            draw_infos,
            materials,
            objects,
            textures,
            null_texture,
            skyboxes,
            sampler,
            wrap_sampler,
            camera,
            input_state: Default::default(),
            rcx: None,
        }
    }

    fn set_cursor_confinement(&mut self, confined: bool) {
        let window = self.windows.get_primary_window().unwrap();
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
        InputState {
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

fn build_pipeline(
    device: Arc<Device>,
    swapchain_format: Format,
    vert: Arc<ShaderModule>,
    frag: Arc<ShaderModule>,
    specialization_constants: SpecializationConstants,
    double_sided: bool,
) -> Arc<GraphicsPipeline> {
    // First, we load the shaders that the pipeline will use: the vertex shader and the
    // fragment shader.
    //
    // A Vulkan shader can in theory contain multiple entry points, so we have to specify
    // which one.

    let constants: Vec<_> = specialization_constants.into();

    let vs = vert
        .specialize(constants.clone().into_iter().collect())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = frag
        .specialize(constants.into_iter().collect())
        .unwrap()
        .entry_point("main")
        .unwrap();

    // Automatically generate a vertex input state from the vertex shader's input
    // interface, that takes a single vertex buffer containing `Vertex` structs.
    let vertex_input_state = CombinedVertex::per_vertex().definition(&vs).unwrap();

    // Make a list of the shader stages that the pipeline will have.
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // We must now create a **pipeline layout** object, which describes the locations and
    // types of descriptor sets and push constants used by the shaders in the pipeline.
    //
    // Multiple pipelines can share a common layout object, which is more efficient. The
    // shaders in a pipeline must use a subset of the resources described in its pipeline
    // layout, but the pipeline layout is allowed to contain resources that are not present
    // in the shaders; they can be used by shaders in other pipelines that share the same
    // layout. Thus, it is a good idea to design shaders so that many pipelines have common
    // resource locations, which allows them to share pipeline layouts.
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
        // set = 3
        vec![
            DescriptorType::UniformBuffer, // binding = 0 uniform Material
            DescriptorType::UniformBuffer, // binding = 1 uniform MatSamplers
        ],
        // set = 4
        vec![DescriptorType::CombinedImageSampler; 22], // Texture Samplers
    ];

    let push_constant_ranges = vec![PushConstantRange {
        stages: ShaderStages::all_graphics(),
        offset: 0,
        size: size_of::<shader::fs::Object>() as u32,
    }];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: bindings
                .into_iter()
                .map(|set| {
                    DescriptorSetLayout::new(
                        device.clone(),
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
            push_constant_ranges,
            ..Default::default()
        },
    )
    .unwrap();

    // We describe the formats of attachment images where the colors, depth and/or stencil
    // information will be written. The pipeline will only be usable with this particular
    // configuration of the attachment images.
    let subpass = PipelineRenderingCreateInfo {
        // We specify a single color attachment that will be rendered to. When we begin
        // rendering, we will specify a swapchain image to be used as this attachment, so
        // here we set its format to be the same format as the swapchain.
        color_attachment_formats: vec![Some(swapchain_format)],
        depth_attachment_format: Some(Format::D32_SFLOAT),
        ..Default::default()
    };

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
                cull_mode: if double_sided {
                    CullMode::None
                } else {
                    CullMode::Back
                },
                ..Default::default()
            }),
            // How multiple fragment shader samples are converted to a single pixel value.
            // The default value does not perform any multisampling.
            multisample_state: Some(MultisampleState::default()),
            // How pixel values are combined with the values already present in the
            // framebuffer. The default value overwrites the old value with the new one,
            // without any blending.
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.color_attachment_formats.len() as u32,
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            // Dynamic states allows us to specify parts of the pipeline settings when
            // recording the command buffer, before we perform drawing. Here, we specify
            // that the viewport should be dynamic.
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

struct PipelineContext {
    pipeline: Arc<GraphicsPipeline>,
    material_sets: HashMap<usize, (Arc<DescriptorSet>, Arc<DescriptorSet>)>,
    const_set: Arc<DescriptorSet>,
    skybox_set: Arc<DescriptorSet>,
    draw_infos: Vec<PrimitiveDrawInfo>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                present_mode: PresentMode::Mailbox,
                width: 1524.,
                height: 1500.,
                scale_factor_override: Some(1.0),
                ..Default::default()
            },
            |_| {},
        );
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let window_size = window_renderer.window().inner_size();

        // Create image views from the current swapchain images.
        let attachment_image_views = window_renderer.swapchain_image_views().to_vec();

        let depth_image = Image::new(
            self.memory_allocator.clone(),
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

        let depth_image_view =
            ImageView::new_default(depth_image.clone()).expect("Failed to create depth image view");

        let vertex_shader = vs::load(self.context.device().clone()).unwrap();
        let fragment_shader = fs::load(self.context.device().clone()).unwrap();

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        // let pipeline = build_pipeline(
        //     self.context.device().clone(),
        //     window_renderer.swapchain_format(),
        //     vertex_shader.clone(),
        //     fragment_shader.clone(),
        // );

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
            for object in &self.objects {
                for info in &self.draw_infos[object.mesh_idx] {
                    let mat_idx = info.mat_idx;
                    mat_prim_map[mat_idx].push(info.clone());
                }
            }
            mat_prim_map
        };
        /*
        for object in self.objects {
            for info in self.draw_infos[object.mesh_idx] {
                let mat_idx = info.mat_idx;
                let material_constants = mat_constants[mat_idx];
                pipelines.entry(material_constants).or_default().entry(info.object_constants).or_insert_with(|| PipelineContext {
                    pipeline: build_pipeline(
                        self.context.device().clone(),
                        window_renderer.swapchain_format(),
                        vertex_shader.clone(),
                        fragment_shader.clone(),
                        SpecializationConstants {
                            object_constants: info.object_constants,
                            material_constants,
                        }
                    ),
                });

            }
        }
         */

        let const_set = {
            #[allow(clippy::useless_conversion)]
            let uniform = fs::Constants {
                u_MipCount: min(
                    self.skyboxes.ggx.image().mip_levels() as i32,
                    self.skyboxes.charlie.image().mip_levels() as i32,
                )
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
            let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform;

            WriteDescriptorSet::buffer(0, subbuffer)
        };

        let skybox_set = {
            [
                (&self.skyboxes.lambertian, &self.wrap_sampler),
                (&self.skyboxes.ggx, &self.wrap_sampler),
                (&self.skyboxes.charlie, &self.wrap_sampler),
                (&self.skyboxes.lut_ggx, &self.sampler),
                (&self.skyboxes.lut_charlie, &self.sampler),
                (&self.skyboxes.lut_sheen_e, &self.sampler),
            ]
            .into_iter()
            .enumerate()
            .map(|(idx, (texture, sampler))| {
                WriteDescriptorSet::image_view_sampler(idx as u32, texture.clone(), sampler.clone())
            })
        };

        let pipelines = {
            let mut pipelines = HashMap::<
                MaterialSpecializationConstants,
                HashMap<ObjectSpecializationConstants, PipelineContext>,
            >::new();

            self.materials
                .iter()
                .enumerate()
                .for_each(|(mat_idx, mat)| {
                    let material_set = {
                        let mat_uniform: fs::Material = mat.into();
                        let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                        *subbuffer.write().unwrap() = mat_uniform;
                        WriteDescriptorSet::buffer(0, subbuffer)
                    };

                    let mat_sampler_set = {
                        let mat_samplers: fs::MatSamplers = mat.into();
                        let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
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
                        // (set = 4, binding = 15) u_TransmissionFramebufferSampler
                        None, // TODO
                        // (set = 4, binding = 16) u_ThicknessSampler
                        mat.volume
                            .as_ref()
                            .and_then(|v| v.thickness_texture.as_ref()),
                        // (set = 4, binding = 17) u_IridescenceSampler
                        mat.iridescence.as_ref().and_then(|i| i.texture.as_ref()),
                        // (set = 4, binding = 18) u_IridescenceThicknessSampler
                        mat.iridescence
                            .as_ref()
                            .and_then(|i| i.thickness_texture.as_ref()),
                        // (set = 4, binding = 19) u_DiffuseTransmissionSampler
                        mat.diffuse_transmission
                            .as_ref()
                            .and_then(|d| d.texture.as_ref()),
                        // (set = 4, binding = 20) u_DiffuseTransmissionColorSampler
                        mat.diffuse_transmission
                            .as_ref()
                            .and_then(|d| d.color_texture.as_ref()),
                        // (set = 4, binding = 21) u_AnisotropySampler
                        mat.anisotropy.as_ref().and_then(|a| a.texture.as_ref()),
                    ]
                    .map(|opt_t| opt_t.map(|t| &self.textures[t.texture.index]));

                    for prim in &mat_prim_map[mat_idx] {
                        let mat_const = mat.into();
                        let pcx = pipelines
                            .entry(mat_const)
                            .or_default()
                            .entry(prim.object_constants)
                            .or_insert_with(|| {
                                let spec_constants = SpecializationConstants {
                                    object_constants: prim.object_constants,
                                    material_constants: mat_const,
                                };
                                let pipeline = build_pipeline(
                                    self.context.device().clone(),
                                    window_renderer.swapchain_format(),
                                    vertex_shader.clone(),
                                    fragment_shader.clone(),
                                    spec_constants,
                                    mat.double_sided,
                                );

                                let material_sets = HashMap::new();
                                let const_set = DescriptorSet::new(
                                    self.descriptor_set_allocator.clone(),
                                    #[allow(clippy::get_first)]
                                    pipeline.layout().set_layouts().get(0).unwrap().clone(),
                                    [const_set.clone()],
                                    [],
                                )
                                .unwrap();
                                let skybox_set = DescriptorSet::new(
                                    self.descriptor_set_allocator.clone(),
                                    pipeline.layout().set_layouts().get(2).unwrap().clone(),
                                    skybox_set.clone(),
                                    [],
                                )
                                .unwrap();

                                let draw_infos = Vec::new();

                                PipelineContext {
                                    pipeline,
                                    material_sets,
                                    const_set,
                                    skybox_set,
                                    draw_infos,
                                }
                            });

                        pcx.draw_infos.push(prim.clone());

                        pcx.material_sets.entry(mat_idx).or_insert_with(|| {
                            let material_set = DescriptorSet::new(
                                self.descriptor_set_allocator.clone(),
                                pcx.pipeline.layout().set_layouts().get(3).unwrap().clone(),
                                [material_set.clone(), mat_sampler_set.clone()].into_iter(),
                                [],
                            )
                            .unwrap();

                            let texture_set = DescriptorSet::new(
                                self.descriptor_set_allocator.clone(),
                                pcx.pipeline.layout().set_layouts().get(4).unwrap().clone(),
                                textures
                                    .into_iter()
                                    .map(|t| t.unwrap_or(&self.null_texture))
                                    .enumerate()
                                    .map(|(idx, texture)| {
                                        WriteDescriptorSet::image_view_sampler(
                                            idx as u32,
                                            texture.clone(),
                                            self.wrap_sampler.clone(),
                                        )
                                    }),
                                [],
                            )
                            .unwrap();

                            (material_set, texture_set)
                        });
                    }
                });

            pipelines
        };

        self.rcx = Some(RenderContext {
            attachment_image_views,
            depth_image_view,
            pipelines,
            viewport,
            frame_times,
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
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();

        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => window_renderer.resize(),
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                ..
            } => {
                self.set_cursor_confinement(true);
            }
            WindowEvent::KeyboardInput { event, .. } => {
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
                    // dbg!(self.camera.position);
                    // dbg!(self.camera.get_view_matrix());
                    rcx.frame_times.clear();
                }
                rcx.frame_times.push_back(frame_start);

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Begin rendering by acquiring the gpu future from the window renderer.
                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                        // Whenever the window resizes we need to recreate everything dependent
                        // on the window size. In this example that
                        // includes the swapchain, the framebuffers
                        // and the dynamic state viewport.
                        rcx.attachment_image_views = swapchain_images
                            .iter()
                            .map(|image| ImageView::new_default(image.image().clone()).unwrap())
                            .collect();
                        let depth_image = Image::new(
                            self.memory_allocator.clone(),
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

                        rcx.depth_image_view = ImageView::new_default(depth_image.clone())
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
                    self.command_buffer_allocator.clone(),
                    self.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*. We specify which
                    // attachments we are going to use for rendering here, which needs to match
                    // what was previously specified when creating the pipeline.
                    .begin_rendering(RenderingInfo {
                        // As before, we specify one color attachment, but now we specify the image
                        // view to use as well as how it should be used.
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            // `Clear` means that we ask the GPU to clear the content of this
                            // attachment at the start of rendering.
                            load_op: AttachmentLoadOp::Clear,
                            // `Store` means that we ask the GPU to store the rendered output in
                            // the attachment image. We could also ask it to discard the result.
                            store_op: AttachmentStoreOp::Store,
                            // The value to clear the attachment with. Here we clear it with a blue
                            // color.
                            //
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // with clear values, any others should use `None` as the clear value.
                            clear_value: Some([0.2, 0.2, 0.2, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                rcx.attachment_image_views[window_renderer.image_index() as usize]
                                    .clone(),
                            )
                        })],
                        depth_attachment: Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some(1.0f32.into()),
                            ..RenderingAttachmentInfo::image_view(rcx.depth_image_view.clone())
                        }),
                        ..Default::default()
                    })
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap();

                builder
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.index_buffer.clone())
                    .unwrap();

                let cam_set = {
                    let proj = {
                        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
                        let near = 0.005;
                        let far = 10000.0;

                        let proj = cgmath::perspective(Rad(FRAC_PI_4), aspect_ratio, near, far);
                        // Vulkan clip space has inverted Y and half Z, compared with OpenGL.
                        // A corrective transformation is needed to make an OpenGL perspective matrix
                        // work properly. See here for more info:
                        // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
                        let correction = Matrix4::<f32>::new(
                            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
                            0.5, 1.0,
                        );

                        correction * proj
                    };
                    let view = self.camera.get_view_matrix();
                    let position = self.camera.position;
                    let view_proj = proj * view;

                    let cam_uniform = fs::Camera {
                        u_ViewMatrix: view.into(),
                        u_ProjectionMatrix: proj.into(),
                        u_ViewProjectionMatrix: view_proj.into(),
                        u_Camera: position.into(),
                    };

                    let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = cam_uniform;
                    WriteDescriptorSet::buffer(0, subbuffer)
                };

                for obj_pipelines in rcx.pipelines.values() {
                    for pcx in obj_pipelines.values() {
                        builder
                            .bind_pipeline_graphics(pcx.pipeline.clone())
                            .unwrap();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                pcx.pipeline.layout().clone(),
                                0,
                                pcx.const_set.clone(),
                            )
                            .unwrap();
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                pcx.pipeline.layout().clone(),
                                2,
                                pcx.skybox_set.clone(),
                            )
                            .unwrap();

                        {
                            let set = DescriptorSet::new(
                                self.descriptor_set_allocator.clone(),
                                pcx.pipeline.layout().set_layouts().get(1).unwrap().clone(),
                                [cam_set.clone()],
                                [],
                            )
                            .unwrap();

                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    pcx.pipeline.layout().clone(),
                                    1,
                                    set,
                                )
                                .unwrap();
                        }

                        for prim in &pcx.draw_infos {
                            let (mat_set, tex_set) = pcx.material_sets[&prim.mat_idx].clone();
                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    pcx.pipeline.layout().clone(),
                                    3,
                                    mat_set,
                                )
                                .unwrap();

                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    pcx.pipeline.layout().clone(),
                                    4,
                                    tex_set,
                                )
                                .unwrap();

                            for obj_idx in &prim.object_ids {
                                let object = &self.objects[*obj_idx];
                                let model = object.transform;
                                let normal = model.transpose().invert().unwrap();
                                let data = fs::Object {
                                    u_ModelMatrix: model.into(),
                                    u_NormalMatrix: normal.into(),
                                };
                                builder
                                    .push_constants(pcx.pipeline.layout().clone(), 0, data)
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
                    }
                }

                builder
                    // We leave the render pass.
                    .end_rendering()
                    .unwrap();

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(self.context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                window_renderer.present(future, false);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

#[derive(Clone, Debug)]
struct PrimitiveDrawInfo {
    pub index_offset: u32,
    pub vertex_offset: i32,
    pub index_count: u32,
    pub mat_idx: usize,
    pub object_constants: ObjectSpecializationConstants,
    pub object_ids: Vec<usize>,
}

fn create_buffer<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
    allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
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
            staging_buffer.clone(),
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

// The next step is to create the shaders.
//
// The raw shader creation API provided by the vulkano library is unsafe for various
// reasons, so The `shader!` macro provides a way to generate a Rust module from GLSL
// source - in the example below, the source is provided as a string input directly to the
// shader, but a path to a source file can be provided as well. Note that the user must
// specify the type of shader (e.g. "vertex", "fragment", etc.) using the `ty` option of
// the macro.
//
// The items generated by the `shader!` macro include a `load` function which loads the
// shader using an input logical device. The module also includes type definitions for
// layout structures defined in the shader source, for example uniforms and push constants.
//
// A more detailed overview of what the `shader!` macro generates can be found in the
// vulkano-shaders crate docs. You can view them at https://docs.rs/vulkano-shaders/
