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
use crate::gltf::{load_gltf, Gltf, Object, TextureFormat};
use crate::material::Material;
use cgmath::{Matrix4, Rad};
use image::{DynamicImage, ImageBuffer};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::FRAC_PI_4;
use std::time::Instant;
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
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
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
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::CursorGrabMode;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod camera;
mod gltf;
mod material;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    depth_image_view: Arc<ImageView>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    frame_times: VecDeque<Instant>,
    material_sets: Vec<Arc<DescriptorSet>>,
    object_sets: Vec<Arc<DescriptorSet>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("Couldn't load Vulkan library");

        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw
        // to a window.
        let required_extensions = Surface::required_extensions(event_loop).expect("No extensions");

        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
            .expect("Couldn't create Vulkan instance");

        // Choose device extensions that we're going to use. In order to present images to a
        // surface, we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // We then choose which physical device to use. First, we enumerate all the available
        // physical devices, then apply filters to narrow them down to those that can support our
        // needs.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Couldn't enumerate physical devices")
            .filter(|p| {
                // For this example, we require at least Vulkan 1.3, or a device that has the
                // `khr_dynamic_rendering` extension available.
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| {
                // Some devices may not support the extensions or features that your application,
                // or report properties and limits that are not sufficient for your application.
                // These should be filtered out here.
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will
                // execute our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example a
                // draw queue and a compute queue), similar to CPU threads. This is something you
                // have to have to manage manually in Vulkan. Queues of the same type belong to the
                // same queue family.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue
                // to handle data transfers in parallel with graphics operations. You may also need
                // a separate queue for compute operations, if your application uses those.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing
                        // to a window surface, as we do in this example, we also need to check
                        // that queues in this queue family are capable of presenting images to the
                        // surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    // The code here searches for the first queue family that is suitable. If none
                    // is found, `None` is returned to `filter_map`, which disqualifies this
                    // physical device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the
            // application. However, not every device is equal, some are preferred over others.
            // Now, we assign each physical device a score, and pick the device with the lowest
            // ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a
            // "default" or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // If the selected device doesn't have Vulkan 1.3 available, then we need to enable the
        // `khr_dynamic_rendering` extension manually. This extension became a core part of Vulkan
        // in version 1.3 and later, so it's always available then and it does not need to be
        // enabled. We can be sure that this extension will be available on the selected physical
        // device, because we filtered out unsuitable devices in the device selection code above.
        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // An iterator of created queues is returned by the function alongside the device.
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device,
            DeviceCreateInfo {
                // The list of queues that we are going to use. Here we only use one queue, from
                // the previously chosen queue family.
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                // A list of optional features and extensions that our program needs to work
                // correctly. Some parts of the Vulkan specs are optional and must be enabled
                // manually at device creation. In this example the only things we are going to
                // need are the `khr_swapchain` extension that allows us to draw to a window, and
                // `khr_dynamic_rendering` if we don't have Vulkan 1.3 available.
                enabled_extensions: device_extensions,

                // In order to render with Vulkan 1.3's dynamic rendering, we need to enable it
                // here. Otherwise, we are only allowed to render with a render pass object, as in
                // the standard triangle example. The feature is required to be supported by the
                // device if it supports Vulkan 1.3 and higher, or if the `khr_dynamic_rendering`
                // extension is available, so we don't need to check for support.
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
            .expect("Couldn't create Vulkan device and queues");

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
        // only use one queue in this example, so we just retrieve the first and only element of
        // the iterator.
        let queue = queues.next().expect("Couldn't get Vulkan queue");

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
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
            .par_iter()
            .flat_map(|mesh| &mesh.primitives)
            .map(|prim| (prim.positions.len(), prim.indices.len()))
            .reduce(|| (0, 0), |(v1, i1), (v2, i2)| (v1 + v2, i1 + i2));

        let (vertex_buffer, index_buffer, draw_infos) = {
            let combined_verts = std::sync::Mutex::new(Vec::with_capacity(vert_count / 3));
            let combined_indices = std::sync::Mutex::new(Vec::with_capacity(index_count));
            let draw_infos = std::sync::Mutex::new(Vec::new());

            meshes.par_iter().for_each(|mesh| {
                let mut prim_infos = Vec::new();

                for prim in &mesh.primitives {
                    let vertex_batch: Vec<CombinedVertex> = prim
                        .positions
                        .par_chunks_exact(3)
                        .zip(prim.normals.par_chunks_exact(3))
                        .zip(prim.tangents.par_chunks_exact(3))
                        .zip(prim.texcoords.par_chunks_exact(2))
                        .map(|(((position, normal), tangent), texcoord)| CombinedVertex {
                            position: position.try_into().unwrap(),
                            normal: normal.try_into().unwrap(),
                            tangent: tangent.try_into().unwrap(),
                            texcoord: texcoord.try_into().unwrap(),
                        })
                        .collect();

                    let mut combined_verts_lock = combined_verts.lock().unwrap();
                    let mut combined_indices_lock = combined_indices.lock().unwrap();

                    prim_infos.push(PrimitiveDrawInfo {
                        index_offset: combined_indices_lock.len() as u32,
                        vertex_offset: combined_verts_lock.len() as i32,
                        index_count: prim.indices.len() as u32,
                        mat_idx: prim.mat_idx,
                    });

                    combined_indices_lock.extend(&prim.indices);
                    combined_verts_lock.extend(vertex_batch);
                }

                let mut draw_infos_lock = draw_infos.lock().unwrap();
                draw_infos_lock.push(prim_infos);
            });

            let combined_verts = combined_verts.into_inner().unwrap();
            let combined_indices = combined_indices.into_inner().unwrap();
            let draw_infos = draw_infos.into_inner().unwrap();

            let vertex_buffer = create_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
                BufferUsage::VERTEX_BUFFER,
                combined_verts,
            );

            let index_buffer = create_buffer(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
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
                queue.queue_family_index(),
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
                        queue.clone(),
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
                .execute(queue.clone())
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
                queue.clone(),
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
                queue.queue_family_index(),
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
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            ImageView::new_default(image).unwrap()
        };

        let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear())
            .expect("Couldn't create sampler");

        let camera = FirstPersonCamera::new();

        App {
            instance,
            device,
            queue,
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
        let window = self.rcx.as_mut().unwrap().window.clone();
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
        // The objective of this example is to draw a triangle on a window. To do so, we first need
        // to create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
        //
        // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
        // object from it, which represents the drawable surface of a window. For that we must wrap
        // the `winit::window::Window` in an `Arc`.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    // Some drivers report an `min_image_count` of 1, but fullscreen mode requires
                    // at least 2. Therefore we must ensure the count is at least 2, otherwise the
                    // program would crash when entering fullscreen mode on those drivers.
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,

                    // The size of the window, only used to initially setup the swapchain.
                    //
                    // NOTE:
                    // On some drivers the swapchain extent is specified by
                    // `surface_capabilities.current_extent` and the swapchain size must use this
                    // extent. This extent is always the same as the window size.
                    //
                    // However, other drivers don't specify a value, i.e.
                    // `surface_capabilities.current_extent` is `None`. These drivers will allow
                    // anything, but the only sensible value is the window size.
                    //
                    // Both of these cases need the swapchain to use the window size, so we just
                    // use that.
                    image_extent: window_size.into(),

                    image_usage: ImageUsage::COLOR_ATTACHMENT,

                    // The alpha mode indicates how the alpha value of the final image will behave.
                    // For example, you can choose whether the window will be opaque or
                    // transparent.
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            )
                .unwrap()
        };

        // When creating the swapchain, we only created plain images. To use them as an attachment
        // for rendering, we must wrap then in an image view.
        //
        // Since we need to draw to multiple images, we are going to create a different image view
        // for each image.
        let attachment_image_views = window_size_dependent_setup(&images);

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
            let vs = vs::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.device.clone())
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
                self.device.clone(),
                // Since we only have one pipeline in this example, and thus one pipeline layout,
                // we automatically generate the creation info for it from the resources used in
                // the shaders. In a real application, you would specify this information manually
                // so that you can re-use one layout in multiple pipelines.
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
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
                color_attachment_formats: vec![Some(swapchain.image_format())],
                depth_attachment_format: Some(Format::D32_SFLOAT),
                ..Default::default()
            };

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                self.device.clone(),
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

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's) or, on Android, when the application went to the background and goes back
        // to the foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        // In the loop below we are going to submit commands to the GPU. Submitting a command
        // produces an object that implements the `GpuFuture` trait, which holds the resources for
        // as long as they are in use by the GPU.
        //
        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to
        // avoid that, we store the submission of the previous frame here.
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

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
            window,
            swapchain,
            attachment_image_views,
            depth_image_view,
            pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
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
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => rcx.recreate_swapchain = true,
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

                let window_size = rcx.window.inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            present_mode: PresentMode::Mailbox,
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;

                    // Now that we have new swapchain images, we must create new image views from
                    // them as well.
                    rcx.attachment_image_views = window_size_dependent_setup(&new_images);

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

                    rcx.recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

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
                    self.queue.queue_family_index(),
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
                                // We specify image view corresponding to the currently acquired
                                // swapchain image, to use for this attachment.
                                rcx.attachment_image_views[image_index as usize].clone(),
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
                        let aspect_ratio = rcx.swapchain.image_extent()[0] as f32
                            / rcx.swapchain.image_extent()[1] as f32;
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

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct CombinedVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    tangent: [f32; 3],
    #[format(R32G32_SFLOAT)]
    texcoord: [f32; 2],
}

#[derive(Clone, Copy)]
struct PrimitiveDrawInfo {
    index_offset: u32,
    vertex_offset: i32,
    index_count: u32,
    mat_idx: usize,
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(images: &[Arc<Image>]) -> Vec<Arc<ImageView>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}

fn create_buffer<T: BufferContents + Send + Sync, I: IntoIterator<Item=T>>(
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
