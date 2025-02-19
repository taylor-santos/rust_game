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
use cgmath::{Matrix, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use image::{DynamicImage, ImageBuffer, ImageReader};
use ktx2::SupercompressionScheme;
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
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers, ImageType,
};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::padded::Padded;
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::shader::{DescriptorBindingRequirements, ShaderStages, SpecializationConstant};
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

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    event_loop.run_app(&mut app)
}

struct Skybox {
    pub lambertian: Arc<ImageView>,
    pub ggx: Arc<ImageView>,
    pub lut_ggx: Arc<ImageView>,
    pub charlie: Arc<ImageView>,
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
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    frame_times: VecDeque<Instant>,
    material_sets: Vec<(Arc<DescriptorSet>, Arc<DescriptorSet>)>,
    const_set: Arc<DescriptorSet>,
    skybox_set: Arc<DescriptorSet>,
}

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

    let rgba_image = img.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    let pixels = rgba_image.into_raw();

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

        let (vertex_buffer, index_buffer, draw_infos) = {
            let mut combined_verts = Vec::with_capacity(vert_count / 3);
            let mut combined_indices = Vec::with_capacity(index_count);
            let mut draw_infos = Vec::new();

            for mesh in meshes {
                let mut prim_infos = Vec::new();
                for prim in mesh.primitives {
                    prim_infos.push(PrimitiveDrawInfo {
                        index_offset: combined_indices.len() as u32,
                        vertex_offset: combined_verts.len() as i32,
                        index_count: prim.indices.len() as u32,
                        mat_idx: prim.mat_idx,
                    });

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

        let mut image_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let textures = {
            let timer = Instant::now();

            let textures = textures
                .into_iter()
                .map(|texture| {
                    let pixels = match texture.format {
                        TextureFormat::R8G8B8A8 => texture.pixels,
                        TextureFormat::R8G8B8 => DynamicImage::ImageRgb8(
                            ImageBuffer::from_raw(texture.width, texture.height, texture.pixels)
                                .unwrap(),
                        )
                        .to_rgba8()
                        .into_raw(),
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
                        Format::R8G8B8A8_UNORM,
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
            let pixel = vec![255u8; 4]; // RGBA black
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
            let lambertian = load_ktx2(
                "textures/field/lambertian/diffuse.ktx2",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();
            let ggx = load_ktx2(
                "textures/field/ggx/specular.ktx2",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();
            let lut_ggx = load_png(
                "textures/lut_ggx.png",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();
            let charlie = load_ktx2(
                "textures/field/charlie/sheen.ktx2",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
            )
            .unwrap();
            let lut_charlie = load_png(
                "textures/lut_charlie.png",
                &mut image_builder,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
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
                lut_ggx,
                charlie,
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
            SamplerCreateInfo::simple_repeat_linear(),
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                present_mode: PresentMode::Immediate,
                width: 1804.,
                height: 1885.,
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

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.

            let specialization_constants = [
                true.into(),  // HAS_NORMAL_VEC3
                true.into(),  // HAS_TANGENT_VEC4
                true.into(),  // MATERIAL_METALLICROUGHNESS
                false.into(), // MATERIAL_SPECULARGLOSSINESS
                false.into(), // MATERIAL_CLEARCOAT
                false.into(), // MATERIAL_SHEEN
                false.into(), // MATERIAL_SPECULAR
                false.into(), // MATERIAL_TRANSMISSION
                false.into(), // MATERIAL_VOLUME
                false.into(), // MATERIAL_IRIDESCENCE
                false.into(), // MATERIAL_DIFFUSE_TRANSMISSION
                false.into(), // MATERIAL_ANISOTROPY
                false.into(), // MATERIAL_IOR
                false.into(), // MATERIAL_DISPERSION
                false.into(), // MATERIAL_EMISSIVE_STRENGTH
                false.into(), // MATERIAL_UNLIT
            ];

            let vs = vs::load(self.context.device().clone())
                .unwrap()
                .specialize(
                    specialization_constants
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (i as u32, v))
                        .collect(),
                )
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.context.device().clone())
                .unwrap()
                .specialize(
                    specialization_constants
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (i as u32, v))
                        .collect(),
                )
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
                size: size_of::<fs::Object>() as u32,
            }];

            let layout = PipelineLayout::new(
                self.context.device().clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: bindings
                        .into_iter()
                        .map(|set| {
                            DescriptorSetLayout::new(
                                self.context.device().clone(),
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
                color_attachment_formats: vec![Some(window_renderer.swapchain_format())],
                depth_attachment_format: Some(Format::D32_SFLOAT),
                ..Default::default()
            };

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                self.context.device().clone(),
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
        };

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

        let material_sets: Vec<_> = {
            let identity3 = [
                Padded([1f32, 0., 0.]),
                Padded([0., 1., 0.]),
                Padded([0., 0., 1.]),
            ];
            self.materials
                .iter()
                .map(|mat| {
                    let material_set = {
                        let mat_uniform = fs::Material {
                            u_MetallicFactor: mat.pbr_metallic_roughness.metallic_factor.into(),
                            u_RoughnessFactor: mat.pbr_metallic_roughness.roughness_factor.into(),
                            u_BaseColorFactor: mat.pbr_metallic_roughness.base_color_factor.into(),
                            u_SpecularFactor: Default::default(),
                            u_DiffuseFactor: Default::default(),
                            u_GlossinessFactor: 0.0,
                            u_SheenRoughnessFactor: mat
                                .sheen_roughness_factor
                                .unwrap_or_default()
                                .into(),
                            u_SheenColorFactor: mat.sheen_color_factor.unwrap_or_default().into(),
                            u_ClearcoatFactor: 0.0,
                            u_ClearcoatRoughnessFactor: Default::default(),
                            u_KHR_materials_specular_specularColorFactor: Default::default(),
                            u_KHR_materials_specular_specularFactor: 0.0,
                            u_TransmissionFactor: 1.0,
                            u_ThicknessFactor: 0.0.into(),
                            u_AttenuationColor: Default::default(),
                            u_AttenuationDistance: 0.0,
                            u_IridescenceFactor: 0.0,
                            u_IridescenceIor: 0.0,
                            u_IridescenceThicknessMinimum: 0.0,
                            u_IridescenceThicknessMaximum: 0.0,
                            u_DiffuseTransmissionFactor: Default::default(),
                            u_DiffuseTransmissionColorFactor: Default::default(),
                            u_EmissiveStrength: 0.0,
                            u_Ior: 1.0.into(),
                            u_Anisotropy: Default::default(),
                            u_Dispersion: 0.0,
                            u_AlphaCutoff: mat.alpha_cutoff.unwrap_or(0.5).into(),
                            //u_vertNormalUVTransform: mat.normal_texture.as_ref().and_then(|t| t.info.transform).map(Into::into).unwrap_or(identity3).into(),
                        };
                        let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                        *subbuffer.write().unwrap() = mat_uniform;
                        WriteDescriptorSet::buffer(0, subbuffer)
                    };

                    let mat_sampler_set = {
                        let mat_samplers = fs::MatSamplers {
                            u_NormalScale: mat
                                .normal_texture
                                .as_ref()
                                .map(|t| t.scale)
                                .unwrap_or(1.0),
                            u_NormalUVSet: mat
                                .normal_texture
                                .as_ref()
                                .map(|t| t.info.texture.tex_coord as i32)
                                .unwrap_or(0)
                                .into(),
                            u_NormalUVTransform: mat
                                .normal_texture
                                .as_ref()
                                .and_then(|t| t.info.transform)
                                .map(Into::into)
                                .unwrap_or(identity3)
                                .into(),
                            u_EmissiveFactor: mat.emissive_factor.into(),
                            u_EmissiveUVSet: mat
                                .emissive_texture
                                .as_ref()
                                .map(|t| t.texture.tex_coord as i32)
                                .unwrap_or(0)
                                .into(),
                            u_EmissiveUVTransform: mat
                                .emissive_texture
                                .as_ref()
                                .and_then(|t| t.transform)
                                .map(Into::into)
                                .unwrap_or(identity3)
                                .into(),
                            u_OcclusionUVSet: mat
                                .occlusion_texture
                                .as_ref()
                                .map(|t| t.info.texture.tex_coord as i32)
                                .unwrap_or(0)
                                .into(),
                            u_OcclusionStrength: mat
                                .occlusion_texture
                                .as_ref()
                                .map(|t| t.scale)
                                .unwrap_or(0.0)
                                .into(),
                            u_OcclusionUVTransform: mat
                                .occlusion_texture
                                .as_ref()
                                .and_then(|t| t.info.transform)
                                .map(Into::into)
                                .unwrap_or(identity3)
                                .into(),
                            u_BaseColorUVSet: mat
                                .base_color_texture
                                .as_ref()
                                .map(|t| t.texture.tex_coord as i32)
                                .unwrap_or(0)
                                .into(),
                            u_BaseColorUVTransform: mat
                                .base_color_texture
                                .as_ref()
                                .and_then(|t| t.transform)
                                .map(Into::into)
                                .unwrap_or(identity3)
                                .into(),
                            u_MetallicRoughnessUVSet: mat
                                .pbr_metallic_roughness
                                .metallic_roughness_texture
                                .as_ref()
                                .map(|t| t.texture.tex_coord as i32)
                                .unwrap_or(0)
                                .into(),
                            u_MetallicRoughnessUVTransform: mat
                                .pbr_metallic_roughness
                                .metallic_roughness_texture
                                .as_ref()
                                .and_then(|t| t.transform)
                                .map(Into::into)
                                .unwrap_or(identity3)
                                .into(),
                            u_SheenRoughnessUVSet: 0.into(), // TODO
                            u_SheenRoughnessUVTransform: identity3, // TODO
                            u_SheenColorUVSet: 0.into(),     // TODO
                            u_SheenColorUVTransform: identity3, // TODO
                            u_DiffuseUVSet: 0.into(),        // TODO
                            u_DiffuseUVTransform: identity3, // TODO
                            u_SpecularGlossinessUVSet: 0.into(), // TODO
                            u_SpecularGlossinessUVTransform: identity3, // TODO
                            u_ClearcoatUVSet: 0.into(),      // TODO
                            u_ClearcoatUVTransform: identity3, // TODO
                            u_ClearcoatRoughnessUVSet: 0.into(), // TODO
                            u_ClearcoatRoughnessUVTransform: identity3, // TODO
                            u_ClearcoatNormalUVSet: 0.into(), // TODO
                            u_ClearcoatNormalUVTransform: identity3, // TODO
                            u_ClearcoatNormalScale: 1.0.into(), // TODO
                            u_SpecularUVSet: 0.into(),       // TODO
                            u_SpecularUVTransform: identity3, // TODO
                            u_SpecularColorUVSet: 0.into(),  // TODO
                            u_SpecularColorUVTransform: identity3, // TODO
                            u_TransmissionUVSet: 0.into(),   // TODO
                            u_TransmissionUVTransform: identity3, // TODO
                            u_TransmissionFramebufferSize: self
                                .windows
                                .get_primary_window()
                                .unwrap()
                                .inner_size()
                                .into(), // TODO
                            u_ThicknessUVSet: 0.into(),      // TODO
                            u_ThicknessUVTransform: identity3, // TODO
                            u_IridescenceUVSet: 0.into(),    // TODO
                            u_IridescenceUVTransform: identity3, // TODO
                            u_IridescenceThicknessUVSet: 0.into(), // TODO
                            u_IridescenceThicknessUVTransform: identity3, // TODO
                            u_DiffuseTransmissionUVSet: 0.into(), // TODO
                            u_DiffuseTransmissionUVTransform: identity3, // TODO
                            u_DiffuseTransmissionColorUVSet: 0.into(), // TODO
                            u_DiffuseTransmissionColorUVTransform: identity3, // TODO
                            u_AnisotropyUVSet: 0.into(),     // TODO
                            u_AnisotropyUVTransform: identity3, // TODO
                        };
                        let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                        *subbuffer.write().unwrap() = mat_samplers;
                        WriteDescriptorSet::buffer(1, subbuffer)
                    };

                    let textures = [
                        // (set = 4, binding = 0) u_NormalSampler
                        mat.normal_texture.as_ref().map(|t| &t.info.texture),
                        // (set = 4, binding = 1) u_EmissiveSampler
                        mat.emissive_texture.as_ref().map(|t| &t.texture),
                        // (set = 4, binding = 2) u_OcclusionSampler
                        mat.occlusion_texture.as_ref().map(|t| &t.info.texture),
                        // (set = 4, binding = 3) u_BaseColorSampler
                        mat.base_color_texture.as_ref().map(|t| &t.texture),
                        // (set = 4, binding = 4) u_MetallicRoughnessSampler
                        mat.pbr_metallic_roughness
                            .metallic_roughness_texture
                            .as_ref()
                            .map(|t| &t.texture),
                        // (set = 4, binding = 5) u_DiffuseSampler
                        None, // TODO
                        // (set = 4, binding = 6) u_SpecularGlossinessSampler
                        None, // TODO
                        // (set = 4, binding = 7) u_ClearcoatSampler
                        None, // TODO
                        // (set = 4, binding = 8) u_ClearcoatRoughnessSampler
                        None, // TODO
                        // (set = 4, binding = 9) u_ClearcoatNormalSampler
                        None, // TODO
                        // (set = 4, binding = 10) u_SheenColorSampler
                        None, // TODO
                        // (set = 4, binding = 11) u_SheenRoughnessSampler
                        None, // TODO
                        // (set = 4, binding = 12) u_SpecularSampler
                        None, // TODO
                        // (set = 4, binding = 13) u_SpecularColorSampler
                        None, // TODO
                        // (set = 4, binding = 14) u_TransmissionSampler
                        None, // TODO
                        // (set = 4, binding = 15) u_TransmissionFramebufferSampler
                        None, // TODO
                        // (set = 4, binding = 16) u_ThicknessSampler
                        None, // TODO
                        // (set = 4, binding = 17) u_IridescenceSampler
                        None, // TODO
                        // (set = 4, binding = 18) u_IridescenceThicknessSampler
                        None, // TODO
                        // (set = 4, binding = 19) u_DiffuseTransmissionSampler
                        None, // TODO
                        // (set = 4, binding = 20) u_DiffuseTransmissionColorSampler
                        None, // TODO
                        // (set = 4, binding = 21) u_AnisotropySampler
                        None, // TODO
                    ]
                    .map(|t| t.map(|t| &self.textures[t.index]));

                    let material_set = DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        pipeline.layout().set_layouts().get(3).unwrap().clone(),
                        [material_set, mat_sampler_set].into_iter(),
                        [],
                    )
                    .unwrap();

                    let texture_set = DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        pipeline.layout().set_layouts().get(4).unwrap().clone(),
                        textures
                            .into_iter()
                            .enumerate()
                            .map(|(idx, texture)| match texture {
                                Some(texture) => WriteDescriptorSet::image_view_sampler(
                                    idx as u32,
                                    texture.clone(),
                                    self.sampler.clone(),
                                ),
                                None => WriteDescriptorSet::image_view_sampler(
                                    idx as u32,
                                    self.null_texture.clone(),
                                    self.sampler.clone(),
                                ),
                            }),
                        [],
                    )
                    .unwrap();

                    (material_set, texture_set)
                })
                .collect()
        };

        let const_set = {
            let const_set = {
                let uniform = fs::Constants {
                    u_MipCount: 5.into(), // TODO
                    u_EnvRotation: [
                        Padded([0f32, 0., 1.]),
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

            DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                pipeline.layout().set_layouts().get(0).unwrap().clone(),
                [const_set],
                [],
            )
            .unwrap()
        };

        let skybox_set = {
            let skybox_set = {
                [
                    &self.skyboxes.lambertian,
                    &self.skyboxes.ggx,
                    &self.skyboxes.lut_ggx,
                    &self.skyboxes.charlie,
                    &self.skyboxes.lut_charlie,
                    &self.skyboxes.lut_sheen_e,
                ]
                .into_iter()
                .enumerate()
                .map(|(idx, texture)| {
                    WriteDescriptorSet::image_view_sampler(
                        idx as u32,
                        texture.clone(),
                        self.sampler.clone(),
                    )
                })
            };

            DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                pipeline.layout().set_layouts().get(2).unwrap().clone(),
                skybox_set,
                [],
            )
            .unwrap()
        };

        self.rcx = Some(RenderContext {
            attachment_image_views,
            depth_image_view,
            pipeline,
            viewport,
            frame_times,
            material_sets,
            const_set,
            skybox_set,
        });
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
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
                            clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
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
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap();

                builder
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.index_buffer.clone())
                    .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.pipeline.layout().clone(),
                        0,
                        rcx.const_set.clone(),
                    )
                    .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.pipeline.layout().clone(),
                        2,
                        rcx.skybox_set.clone(),
                    )
                    .unwrap();

                {
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

                    let view_proj = proj * view;

                    let cam_uniform = fs::Camera {
                        u_ViewMatrix: view.into(),
                        u_ProjectionMatrix: proj.into(),
                        u_ViewProjectionMatrix: view_proj.into(),
                        u_Camera: self.camera.position.into(),
                    };

                    /*
                    let matrix = Matrix4::new(
                        0.6345784664154053,
                        0.,
                        0.7728584408760071,
                        0.,
                        0.11100656539201736,
                        0.9896312952041626,
                        -0.09114524722099304,
                        0.,
                        -0.7648448944091797,
                        0.14363117516040802,
                        0.6279987096786499,
                        0.,
                        -2.3856077194213867,
                        0.4475397765636444,
                        1.769579529762268,
                        1.
                    );

                    let proj = cgmath::perspective(
                        Rad(0.7853981633974483),
                        window_size.width as f32 / window_size.height as f32,
                        0.0037298484617011808,
                        37.29848461701181,
                    );
                    println!("{:?}",proj);

                    let correction = Matrix4::<f32>::new(
                        1.0, 0.0, 0.0, 0.0,
                        0.0, -1.0, 0.0, 0.0,
                        0.0, 0.0, 0.5, 0.0,
                        0.0, 0.0, 0.5, 1.0,
                    );

                    let view_proj = correction * proj * matrix.invert().unwrap();


                    let cam_uniform = fs::Camera {
                        u_Camera: [matrix.w.x, matrix.w.y, matrix.w.z],
                        u_ViewProjectionMatrix: view_proj.into(),
                    };

                     */

                    let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = cam_uniform;
                    let write_set = WriteDescriptorSet::buffer(0, subbuffer);

                    let set = DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        rcx.pipeline.layout().set_layouts().get(1).unwrap().clone(),
                        [write_set],
                        [],
                    )
                    .unwrap();

                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.pipeline.layout().clone(),
                            1,
                            set,
                        )
                        .unwrap();
                }

                for object in &self.objects {
                    {
                        let model = object.transform;
                        let normal = model.transpose().invert().unwrap();
                        let uniform = fs::Object {
                            u_ModelMatrix: model.into(),
                            u_NormalMatrix: normal.into(),
                        };
                        builder
                            .push_constants(rcx.pipeline.layout().clone(), 0, uniform)
                            .unwrap();
                    }

                    for prim in &self.draw_infos[object.mesh_idx] {
                        let (mat_set, tex_set) = rcx.material_sets[prim.mat_idx].clone();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                rcx.pipeline.layout().clone(),
                                3,
                                mat_set,
                            )
                            .unwrap();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                rcx.pipeline.layout().clone(),
                                4,
                                tex_set,
                            )
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

#[derive(Clone, Copy, Debug)]
struct PrimitiveDrawInfo {
    index_offset: u32,
    vertex_offset: i32,
    index_count: u32,
    mat_idx: usize,
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

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
        include: ["shaders"],
        define: [
            ("HAS_POSITION_VEC3", "1"),
            ("HAS_TEXCOORD_0_VEC2", "1"),
            // Keep synced with fragment shader
            // ("HAS_TEXCOORD_0_VEC2", "1"),
            // ("HAS_TEXCOORD_1_VEC2", "1"),
            // ("HAS_VERT_NORMAL_UV_TRANSFORM", "1"),
            // ("HAS_COLOR_0_VEC3", "1"),
            // ("HAS_COLOR_0_VEC4", "1"),

            // Unique to vertex shader
            // ("USE_INSTANCING", "1"),
            // ("USE_MORPHING", "1"),
            // ("USE_SKINNING", "1"),
        ],
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.glsl",
        include: ["shaders"],
        define: [
            ("HAS_NORMAL_MAP", "1"),
            ("HAS_OCCLUSION_MAP", "1"),
            ("HAS_EMISSIVE_MAP", "1"),
            ("HAS_BASE_COLOR_MAP", "1"),
            ("HAS_METALLIC_ROUGHNESS_MAP", "1"),
            ("ALPHAMODE", "ALPHAMODE_OPAQUE"),
            ("HAS_POSITION_VEC3", "1"),
            ("HAS_TEXCOORD_0_VEC2", "1"),
            ("USE_IBL", "1"),
            ("TONEMAP_KHR_PBR_NEUTRAL", "1"),
            ("DEBUG", "DEBUG_NONE"),
            // sheen
            // ("HAS_NORMAL_MAP", "1"),
            // ("HAS_OCCLUSION_UV_TRANSFORM", "1"),
            // ("HAS_OCCLUSION_MAP", "1"),
            // ("HAS_BASECOLOR_UV_TRANSFORM", "1"),
            // ("HAS_BASE_COLOR_MAP", "1"),
            // ("ALPHAMODE", "ALPHAMODE_OPAQUE"),
            // ("HAS_POSITION_VEC3", "1"),
            // ("HAS_TEXCOORD_0_VEC2", "1"),
            // ("HAS_TEXCOORD_1_VEC2", "1"),
            // //("HAS_VERT_NORMAL_UV_TRANSFORM", "1"),
            // ("USE_IBL", "1"),
            // ("TONEMAP_KHR_PBR_NEUTRAL", "1"),
            // ("DEBUG", "DEBUG_NORMAL_SHADING"),

        //    // Keep synced with vertex shader
        //    ("HAS_TEXCOORD_0_VEC2", "1"),
        //    ("HAS_TEXCOORD_1_VEC2", "1"),
        //    // ("HAS_COLOR_0_VEC3", "1"),
        //    // ("HAS_COLOR_0_VEC4", "1"),

        //    // Unique to fragment shader
        //    ("DEBUG", "DEBUG_NONE"),
        //    ("ALPHAMODE", "ALPHAMODE_OPAQUE"),
        //    ("HAS_NORMAL_MAP", "1"),
        //    ("HAS_BASE_COLOR_MAP", "1"),
        //    ("HAS_EMISSIVE_MAP", "1"),
        //    ("USE_IBL", "1"),
        //    ("HAS_OCCLUSION_MAP", "1"),
        //    ("HAS_METALLIC_ROUGHNESS_MAP", "1"),
        //    ("TONEMAP_KHR_PBR_NEUTRAL", "1"),
            // ("HAS_ANISOTROPY_MAP", "1"),
            // ("HAS_ANISOTROPY_UV_TRANSFORM", "1"),
            // ("HAS_BASECOLOR_UV_TRANSFORM", "1"),
            // ("HAS_CLEARCOAT_MAP", "1"),
            // ("HAS_CLEARCOAT_NORMAL_MAP", "1"),
            // ("HAS_CLEARCOAT_ROUGHNESS_MAP", "1"),
            // ("HAS_CLEARCOAT_UV_TRANSFORM", "1"),
            // ("HAS_CLEARCOATNORMAL_UV_TRANSFORM", "1"),
            // ("HAS_CLEARCOATROUGHNESS_UV_TRANSFORM", "1"),
            // ("HAS_DIFFUSE_MAP", "1"),
            // ("HAS_DIFFUSE_TRANSMISSION_COLOR_MAP", "1"),
            // ("HAS_DIFFUSE_TRANSMISSION_MAP", "1"),
            // ("HAS_DIFFUSE_UV_TRANSFORM", "1"),
            // ("HAS_DIFFUSETRANSMISSION_UV_TRANSFORM", "1"),
            // ("HAS_DIFFUSETRANSMISSIONCOLOR_UV_TRANSFORM", "1"),
            // ("HAS_EMISSIVE_UV_TRANSFORM", "1"),
            // ("HAS_IRIDESCENCE_MAP", "1"),
            // ("HAS_IRIDESCENCE_THICKNESS_MAP", "1"),
            // ("HAS_IRIDESCENCE_UV_TRANSFORM", "1"),
            // ("HAS_IRIDESCENCETHICKNESS_UV_TRANSFORM", "1"),
            // ("HAS_METALLICROUGHNESS_UV_TRANSFORM", "1"),
            // ("HAS_NORMAL_UV_TRANSFORM", "1"),
            // ("HAS_OCCLUSION_UV_TRANSFORM", "1"),
            // ("HAS_SHEEN_COLOR_MAP", "1"),
            // ("HAS_SHEEN_ROUGHNESS_MAP", "1"),
            // ("HAS_SHEENCOLOR_UV_TRANSFORM", "1"),
            // ("HAS_SHEENROUGHNESS_UV_TRANSFORM", "1"),
            // ("HAS_SPECULAR_COLOR_MAP", "1"),
            // ("HAS_SPECULAR_GLOSSINESS_MAP", "1"),
            // ("HAS_SPECULAR_MAP", "1"),
            // ("HAS_SPECULAR_UV_TRANSFORM", "1"),
            // ("HAS_SPECULARCOLOR_UV_TRANSFORM", "1"),
            // ("HAS_SPECULARGLOSSINESS_UV_TRANSFORM", "1"),
            // ("HAS_THICKNESS_MAP", "1"),
            // ("HAS_THICKNESS_UV_TRANSFORM", "1"),
            // ("HAS_TRANSMISSION_MAP", "1"),
            // ("HAS_TRANSMISSION_UV_TRANSFORM", "1"),
            // ("LINEAR_OUTPUT", "1"),
            // ("NOT_TRIANGLE", "1"),
            // ("TONEMAP_ACES_HILL", "1"),
            // ("TONEMAP_ACES_HILL_EXPOSURE_BOOST", "1"),
            // ("TONEMAP_ACES_NARKOWICZ", "1"),
            // ("USE_PUNCTUAL", "1"),
        ],
    }
}
