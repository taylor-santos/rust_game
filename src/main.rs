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
use cgmath::{Matrix4, Rad};
use image::{DynamicImage, ImageBuffer};
use std::collections::VecDeque;
use std::f32::consts::FRAC_PI_4;
use std::time::{Duration, Instant};
use std::{error::Error, sync::Arc};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{
    CopyBufferInfo, CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::{ImageCreateInfo, ImageType};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
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
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    sync::{self, GpuFuture},
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
    material_sets: Vec<Arc<DescriptorSet>>,
    object_sets: Vec<Arc<DescriptorSet>>,
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

        let textures = {
            let timer = Instant::now();
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                context.graphics_queue().queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("Couldn't create Vulkan command buffer");

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
                    let upload_buffer = create_buffer(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        context.graphics_queue().clone(),
                        BufferUsage::TRANSFER_SRC,
                        pixels,
                    );

                    let image = Image::new(
                        memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R8G8B8A8_UNORM,
                            extent,
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                    )
                    .unwrap();

                    builder
                        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                            upload_buffer,
                            image.clone(),
                        ))
                        .unwrap();

                    ImageView::new_default(image).unwrap()
                })
                .collect::<Vec<_>>();

            builder
                .build()
                .unwrap()
                .execute(context.graphics_queue().clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            println!(
                "Uploaded {} textures in {:?}",
                textures.len(),
                timer.elapsed()
            );

            textures
        };

        let null_texture = {
            let pixel = vec![0u8, 0u8, 0u8, 255u8]; // RGBA black
            let extent: [u32; 3] = [1, 1, 1]; // 1x1 texture
            let upload_buffer = create_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                BufferUsage::TRANSFER_SRC,
                pixel,
            );

            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                context.graphics_queue().queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();

            let command_buffer = builder.build().unwrap();

            command_buffer
                .execute(context.graphics_queue().clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            ImageView::new_default(image).unwrap()
        };

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
            let vs = vs::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.context.device().clone())
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
            let layout = PipelineLayout::new(
                self.context.device().clone(),
                // Since we only have one pipeline in this example, and thus one pipeline layout,
                // we automatically generate the creation info for it from the resources used in
                // the shaders. In a real application, you would specify this information manually
                // so that you can re-use one layout in multiple pipelines.
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.context.device().clone())
                    .unwrap(),
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

        let material_sets = {
            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            self.materials
                .iter()
                .map(|mat| {
                    let descriptor_set = {
                        let mat_uniform = fs::Material {
                            baseColorFactor: mat.pbr_metallic_roughness.base_color_factor.into(),
                            metallicFactor: mat.pbr_metallic_roughness.metallic_factor.into(),
                            roughnessFactor: mat.pbr_metallic_roughness.roughness_factor.into(),
                            emissiveFactor: mat.emissive_factor.into(),
                            emissiveStrength: mat.emissive_strength.unwrap_or(1.0).into(),
                        };
                        let subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                        *subbuffer.write().unwrap() = mat_uniform;
                        WriteDescriptorSet::buffer(0, subbuffer)
                    };

                    let textures = [
                        mat.base_color_texture
                            .as_ref()
                            .map(|t| &self.textures[t.index]),
                        mat.normal_texture
                            .as_ref()
                            .map(|t| &self.textures[t.texture.index]),
                        mat.pbr_metallic_roughness
                            .metallic_roughness_texture
                            .as_ref()
                            .map(|t| &self.textures[t.index]),
                        mat.emissive_texture
                            .as_ref()
                            .map(|t| &self.textures[t.index]),
                    ];

                    DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        layout.clone(),
                        std::iter::once(descriptor_set).chain(
                            textures
                                .into_iter()
                                .enumerate()
                                .map(|(idx, texture)| match texture {
                                    Some(texture) => WriteDescriptorSet::image_view_sampler(
                                        1 + idx as u32,
                                        texture.clone(),
                                        self.sampler.clone(),
                                    ),
                                    None => WriteDescriptorSet::image_view_sampler(
                                        1 + idx as u32,
                                        self.null_texture.clone(),
                                        self.sampler.clone(),
                                    ),
                                }),
                        ),
                        [],
                    )
                    .unwrap()
                })
                .collect()
        };

        let object_sets = {
            let layout = pipeline.layout().set_layouts().get(1).unwrap();
            self.objects
                .iter()
                .map(|object| {
                    let descriptor_set = {
                        let obj_uniform = vs::Object {
                            model: object.transform.into(),
                        };
                        let obj_subbuffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                        *obj_subbuffer.write().unwrap() = obj_uniform;
                        WriteDescriptorSet::buffer(0, obj_subbuffer)
                    };

                    DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        layout.clone(),
                        [descriptor_set],
                        [],
                    )
                    .unwrap()
                })
                .collect()
        };

        self.rcx = Some(RenderContext {
            attachment_image_views,
            depth_image_view,
            pipeline,
            viewport,
            frame_times,
            material_sets,
            object_sets,
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

                {
                    let light_direction = [0.5, 1.0, 0.0].into();
                    let light_color = [1.0, 1.0, 1.0].into();
                    let light_ambient = [0.4, 0.4, 0.4].into();

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

                    let push_constants = fs::PushConstants {
                        light_direction,
                        light_color,
                        light_ambient,
                        cam_viewproj: (proj * view).into(),
                        cam_position: self.camera.position.into(),
                    };

                    builder
                        .push_constants(rcx.pipeline.layout().clone(), 0, push_constants)
                        .unwrap();
                };

                builder
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.index_buffer.clone())
                    .unwrap();

                for (object_idx, object) in self.objects.iter().enumerate() {
                    let object_set = rcx.object_sets[object_idx].clone();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.pipeline.layout().clone(),
                            1,
                            object_set,
                        )
                        .unwrap();
                    for prim in &self.draw_infos[object.mesh_idx] {
                        let mat_set = rcx.material_sets[prim.mat_idx].clone();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                rcx.pipeline.layout().clone(),
                                0,
                                mat_set,
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
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
    }
}
