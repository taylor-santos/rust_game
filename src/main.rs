// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

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
use crate::gltf::{load_gltf, TextureFormat};
use cgmath::{Deg, Matrix4, Rad};
use image::{DynamicImage, ImageBuffer};
use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{
    CopyBufferToImageInfo, PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::{ImageCreateInfo, ImageType};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::PresentMode;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
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
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::event::{DeviceEvent, ElementState, MouseButton, VirtualKeyCode};
use winit::window::{CursorGrabMode, Window};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod camera;
mod gltf;
mod material;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Tangent {
    #[format(R32G32B32_SFLOAT)]
    tangent: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Texcoord {
    #[format(R32G32_SFLOAT)]
    texcoord: [f32; 2],
}

fn create_buffer<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
    allocator: Arc<StandardMemoryAllocator>,
    usage: BufferUsage,
    data: I,
) -> Subbuffer<[T]>
where
    I::IntoIter: ExactSizeIterator,
{
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data,
    )
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (meshes, textures, materials) = load_gltf("models/DamagedHelmet.glb")?;

    let event_loop = EventLoop::new();

    let library = VulkanLibrary::new()?;

    // The first step of any Vulkan program is to create an instance.
    //
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need to
    // enable manually. To do so, we ask `Surface` for the list of extensions required to draw to
    // a window.
    let required_extensions = Surface::required_extensions(&event_loop);

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
    )?;

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
    //
    // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
    // object from it, which represents the drawable surface of a window. For that we must wrap the
    // `winit::window::Window` in an `Arc`.
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let surface = Surface::from_window(instance.clone(), window.clone())?;

    // Choose device extensions that we're going to use. In order to present images to a surface,
    // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
    let mut device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // We then choose which physical device to use. First, we enumerate all the available physical
    // devices, then apply filters to narrow them down to those that can support our needs.
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()?
        .filter(|p| {
            // For this example, we require at least Vulkan 1.3, or a device that has the
            // `khr_dynamic_rendering` extension available.
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| {
            // Some devices may not support the extensions or features that your application, or
            // report properties and limits that are not sufficient for your application. These
            // should be filtered out here.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will execute
            // our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a draw
            // queue and a compute queue), similar to CPU threads. This is something you have to
            // have to manage manually in Vulkan. Queues of the same type belong to the same queue
            // family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-world application, you may want to use a separate dedicated transfer queue to
            // handle data transfers in parallel with graphics operations. You may also need a
            // separate queue for compute operations, if your application uses those.
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // We select a queue family that supports graphics operations. When drawing to
                    // a window surface, as we do in this example, we also need to check that
                    // queues in this queue family are capable of presenting images to the surface.
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|i| (p, i as u32))
        })
        // All the physical devices that pass the filters above are suitable for the application.
        // However, not every device is equal, some are preferred over others. Now, we assign each
        // physical device a score, and pick the device with the lowest ("best") score.
        //
        // In this example, we simply select the best-scoring device to use in the application.
        // In a real-world setting, you may want to use the best-scoring device only as a "default"
        // or "recommended" device, and let the user choose the device themself.
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
    // in version 1.3 and later, so it's always available then and it does not need to be enabled.
    // We can be sure that this extension will be available on the selected physical device,
    // because we filtered out unsuitable devices in the device selection code above.
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
            // The list of queues that we are going to use. Here we only use one queue, from the
            // previously chosen queue family.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            // A list of optional features and extensions that our program needs to work correctly.
            // Some parts of the Vulkan specs are optional and must be enabled manually at device
            // creation. In this example the only things we are going to need are the
            // `khr_swapchain` extension that allows us to draw to a window, and
            // `khr_dynamic_rendering` if we don't have Vulkan 1.3 available.
            enabled_extensions: device_extensions,

            // In order to render with Vulkan 1.3's dynamic rendering, we need to enable it here.
            // Otherwise, we are only allowed to render with a render pass object, as in the
            // standard triangle example. The feature is required to be supported by the device if
            // it supports Vulkan 1.3 and higher, or if the `khr_dynamic_rendering` extension is
            // available, so we don't need to check for support.
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::empty()
            },

            ..Default::default()
        },
    )?;

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. We only
    // use one queue in this example, so we just retrieve the first and only element of the
    // iterator.
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
    // swapchain allocates the color buffers that will contain the image that will ultimately be
    // visible on the screen. These images are returned alongside the swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only pass
        // values that are allowed by the capabilities.
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())?;

        // Choosing the internal format that the images will have.
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())?[0]
            .0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                // Some drivers report an `min_image_count` of 1, but fullscreen mode requires at
                // least 2. Therefore we must ensure the count is at least 2, otherwise the program
                // would crash when entering fullscreen mode on those drivers.
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
                image_extent: window.inner_size().into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,

                // The present mode determines how the swapchain behaves when multiple images are
                // waiting in the queue to be presented.
                //
                // `PresentMode::Immediate` (vsync off) displays the latest image immediately,
                // without waiting for the next vertical blanking period. This may cause tearing.
                //
                // `PresentMode::Fifo` (vsync on) appends the latest image to the end of the queue,
                // and the front of the queue is removed during each vertical blanking period to be
                // presented. No tearing will be visible.
                present_mode: PresentMode::Immediate,

                // The alpha mode indicates how the alpha value of the final image will behave. For
                // example, you can choose whether the window will be opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )?
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let uniform_buffer = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    );

    // First, we load the shaders that the pipeline will use:
    // the vertex shader and the fragment shader.
    //
    // A Vulkan shader can in theory contain multiple entry points, so we have to specify which
    // one.
    let vs = vs::load(device.clone())?.entry_point("main").unwrap();
    let fs = fs::load(device.clone())?.entry_point("main").unwrap();

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.
    let (mut pipeline, mut framebuffers, mut depth_buffer) = window_size_dependent_setup(
        device.clone(),
        vs.clone(),
        fs.clone(),
        swapchain.image_format(),
        &images,
        memory_allocator.clone(),
    );

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain. Here,
    // we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    // Before we can start creating and recording command buffers, we need a way of allocating
    // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command pools
    // underneath and provides a safe interface for them.
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    struct MeshInfo {
        pub vertex_buffer: Subbuffer<[f32]>,
        pub normal_buffer: Subbuffer<[f32]>,
        pub tangent_buffer: Subbuffer<[f32]>,
        pub texcoord_buffer: Subbuffer<[f32]>,
        pub index_buffer: Subbuffer<[u32]>,
        pub index_count: u32,
        pub mat_idx: usize,
    }

    let mesh_infos = meshes
        .into_iter()
        .map(|mesh| {
            let index_count = mesh.indices.len() as u32;
            let vertex_buffer = create_buffer(
                memory_allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                mesh.positions,
            );
            let normal_buffer = create_buffer(
                memory_allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                mesh.normals,
            );
            let tangent_buffer = create_buffer(
                memory_allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                mesh.tangents,
            );
            let texcoord_buffer = create_buffer(
                memory_allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                mesh.texcoords,
            );
            let index_buffer = create_buffer(
                memory_allocator.clone(),
                BufferUsage::INDEX_BUFFER,
                mesh.indices,
            );
            MeshInfo {
                vertex_buffer,
                normal_buffer,
                tangent_buffer,
                texcoord_buffer,
                index_buffer,
                index_count,
                mat_idx: mesh.mat_idx,
            }
        })
        .collect::<Vec<_>>();

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear())?;

    let images = {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let image_buffer_time = Instant::now();
        let images = textures
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
                    BufferUsage::TRANSFER_SRC,
                    pixels.into_iter(),
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

        let upload_command_buffer = builder.build()?;

        upload_command_buffer
            .execute(queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        dbg!(image_buffer_time.elapsed());

        images
    };

    let null_texture = {
        let pixel = vec![0u8, 0u8, 0u8, 255u8]; // RGBA black
        let extent: [u32; 3] = [1, 1, 1]; // 1x1 texture
        let upload_buffer = create_buffer(
            memory_allocator.clone(),
            BufferUsage::TRANSFER_SRC,
            pixel.into_iter(),
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
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        ))?;

        let command_buffer = builder.build()?;

        command_buffer
            .execute(queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        ImageView::new_default(image)?
    };

    // Initialization is finally finished!

    let mut camera = FirstPersonCamera::new();

    set_cursor_confinement(window.as_ref(), false);

    let mut mouse_attached = false;
    let mut forward = false;
    let mut right = false;
    let mut backward = false;
    let mut left = false;
    let mut last_instant = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => recreate_swapchain = true,
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    ..
                } => {
                    set_cursor_confinement(window.as_ref(), true);
                    mouse_attached = true;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    let pressed = input.state == ElementState::Pressed;

                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => forward = pressed,
                        Some(VirtualKeyCode::A) => left = pressed,
                        Some(VirtualKeyCode::S) => backward = pressed,
                        Some(VirtualKeyCode::D) => right = pressed,
                        Some(VirtualKeyCode::Q) => *control_flow = ControlFlow::Exit,
                        Some(VirtualKeyCode::Escape) => {
                            set_cursor_confinement(window.as_ref(), false);
                            mouse_attached = false;
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (x, y) },
                ..
            } => {
                if mouse_attached {
                    camera.rotate(x as f32, y as f32);
                }
            }
            Event::MainEventsCleared => {
                let now = Instant::now();
                let delta_time = now.duration_since(last_instant);
                last_instant = now;

                let delta_t = delta_time.as_secs_f32();

                if forward {
                    camera.move_forward(delta_t);
                }
                if left {
                    camera.move_left(delta_t);
                }
                if right {
                    camera.move_right(delta_t);
                }
                if backward {
                    camera.move_backward(delta_t);
                }
            }
            Event::RedrawEventsCleared => {
                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if recreate_swapchain {
                    dbg!(recreate_swapchain);
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers, new_depth_buffer) =
                        window_size_dependent_setup(
                            device.clone(),
                            vs.clone(),
                            fs.clone(),
                            swapchain.image_format(),
                            &new_images,
                            memory_allocator.clone(),
                        );
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    depth_buffer = new_depth_buffer;
                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // In order to draw, we have to build a *command buffer*. The command buffer object
                // holds the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
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
                                framebuffers[image_index as usize].clone(),
                            )
                        })],
                        depth_attachment: Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some(1.0f32.into()), // clear depth to 1.0
                            ..RenderingAttachmentInfo::image_view(depth_buffer.clone())
                        }),
                        ..Default::default()
                    })
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap();

                {
                    let camera_subbuffer = {
                        let proj = {
                            let aspect_ratio = swapchain.image_extent()[0] as f32
                                / swapchain.image_extent()[1] as f32;
                            let near = 0.005;
                            let far = 10000.0;

                            let proj = cgmath::perspective(Rad(FRAC_PI_4), aspect_ratio, near, far);
                            // Vulkan clip space has inverted Y and half Z, compared with OpenGL.
                            // A corrective transformation is needed to make an OpenGL perspective matrix
                            // work properly. See here for more info:
                            // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
                            let correction = Matrix4::<f32>::new(
                                1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
                                0.0, 0.5, 1.0,
                            );

                            correction * proj
                        };
                        let view = camera.get_view_matrix();

                        let camera_uniform = vs::Camera {
                            viewproj: (proj * view).into(),
                            position: camera.position.into(),
                        };
                        let camera_subbuffer = uniform_buffer.allocate_sized().unwrap();
                        *camera_subbuffer.write().unwrap() = camera_uniform;
                        camera_subbuffer
                    };

                    let light_subbuffer = {
                        let light = fs::Light {
                            direction: [0.5, 1.0, 0.0].into(),
                            color: [1.0, 1.0, 1.0].into(),
                            ambient: [0.4, 0.4, 0.4].into(),
                        };
                        let light_subbuffer = uniform_buffer.allocate_sized().unwrap();
                        *light_subbuffer.write().unwrap() = light;
                        light_subbuffer
                    };

                    let write_sets = [
                        WriteDescriptorSet::buffer(0, camera_subbuffer),
                        WriteDescriptorSet::buffer(1, light_subbuffer),
                    ];

                    let layout0 = pipeline.layout().set_layouts().get(0).unwrap();
                    let persistent_sets = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        layout0.clone(),
                        write_sets,
                        [],
                    )
                    .unwrap();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipeline.layout().clone(),
                            0,
                            persistent_sets,
                        )
                        .unwrap()
                };

                for mesh in &mesh_infos {
                    let mat = &materials[mesh.mat_idx];

                    let layout1 = pipeline.layout().set_layouts().get(1).unwrap();
                    let object_set = {
                        // let model = Matrix4::from_scale(0.01);
                        let model =
                            Matrix4::from_angle_y(Deg(180.0)) * Matrix4::from_angle_x(Deg(90.0));
                        let uniform_obj = fs::Object {
                            model: model.into(),
                            baseColorFactor: mat.pbr_metallic_roughness.base_color_factor.into(),
                            metallicFactor: mat.pbr_metallic_roughness.metallic_factor.into(),
                            roughnessFactor: mat.pbr_metallic_roughness.roughness_factor.into(),
                            emissiveFactor: mat.emissive_factor.into(),
                            emissiveStrength: mat.emissive_strength.unwrap_or(1.0).into(),
                        };
                        let obj_subbuffer = uniform_buffer.allocate_sized().unwrap();
                        *obj_subbuffer.write().unwrap() = uniform_obj;
                        WriteDescriptorSet::buffer(0, obj_subbuffer)
                    };

                    let images = [
                        mat.base_color_texture.as_ref().map(|t| &images[t.index]),
                        mat.normal_texture
                            .as_ref()
                            .map(|t| &images[t.texture.index]),
                        mat.pbr_metallic_roughness
                            .metallic_roughness_texture
                            .as_ref()
                            .map(|t| &images[t.index]),
                        mat.emissive_texture.as_ref().map(|t| &images[t.index]),
                    ];
                    let persistent_set = PersistentDescriptorSet::new(
                        &descriptor_set_allocator,
                        layout1.clone(),
                        std::iter::once(object_set).chain(images.into_iter().enumerate().map(
                            |(idx, image)| match image {
                                Some(image) => WriteDescriptorSet::image_view_sampler(
                                    1 + idx as u32,
                                    image.clone(),
                                    sampler.clone(),
                                ),
                                None => WriteDescriptorSet::image_view_sampler(
                                    1 + idx as u32,
                                    null_texture.clone(),
                                    sampler.clone(),
                                ),
                            },
                        )),
                        [],
                    )
                    .unwrap();

                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipeline.layout().clone(),
                            1,
                            persistent_set.clone(),
                        )
                        .unwrap()
                        .bind_vertex_buffers(
                            0,
                            (
                                mesh.vertex_buffer.clone(),
                                mesh.normal_buffer.clone(),
                                mesh.tangent_buffer.clone(),
                                mesh.texcoord_buffer.clone(),
                            ),
                        )
                        .unwrap()
                        .bind_index_buffer(mesh.index_buffer.clone())
                        .unwrap()
                        .draw_indexed(mesh.index_count, 1, 0, 0, 0)
                        .unwrap();
                }

                builder
                    // We leave the render pass.
                    .end_rendering()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
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
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: EntryPoint,
    fs: EntryPoint,
    color_format: Format,
    images: &[Arc<Image>],
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> (Arc<GraphicsPipeline>, Vec<Arc<ImageView>>, Arc<ImageView>) {
    let attachment_image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>();

    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT, // Include ` | ImageUsage::TRANSIENT_ATTACHMENT` if the depth buffer will not be sampled
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    // Automatically generate a vertex input state from the vertex shader's input interface,
    // that takes a single vertex buffer containing `Vertex` structs.
    let vertex_input_state = [
        Position::per_vertex(),
        Normal::per_vertex(),
        Tangent::per_vertex(),
        Texcoord::per_vertex(),
    ]
    .definition(&vs.info().input_interface)
    .unwrap();

    // Make a list of the shader stages that the pipeline will have.
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // We must now create a **pipeline layout** object, which describes the locations and types of
    // descriptor sets and push constants used by the shaders in the pipeline.
    //
    // Multiple pipelines can share a common layout object, which is more efficient.
    // The shaders in a pipeline must use a subset of the resources described in its pipeline
    // layout, but the pipeline layout is allowed to contain resources that are not present in the
    // shaders; they can be used by shaders in other pipelines that share the same layout.
    // Thus, it is a good idea to design shaders so that many pipelines have common resource
    // locations, which allows them to share pipeline layouts.
    let layout = PipelineLayout::new(
        device.clone(),
        // Since we only have one pipeline in this example, and thus one pipeline layout,
        // we automatically generate the creation info for it from the resources used in the
        // shaders. In a real application, you would specify this information manually so that you
        // can re-use one layout in multiple pipelines.
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    // We describe the formats of attachment images where the colors, depth and/or stencil
    // information will be written. The pipeline will only be usable with this particular
    // configuration of the attachment images.
    let subpass = PipelineRenderingCreateInfo {
        // We specify a single color attachment that will be rendered to. When we begin
        // rendering, we will specify a swapchain image to be used as this attachment, so here
        // we set its format to be the same format as the swapchain.
        color_attachment_formats: vec![Some(color_format)],
        depth_attachment_format: Some(Format::D32_SFLOAT),
        ..Default::default()
    };

    let extent = images[0].extent();
    let viewport_state = ViewportState {
        viewports: [Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        }]
        .into_iter()
        .collect(),
        ..Default::default()
    };

    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            // How vertex data is read from the vertex buffers into the vertex shader.
            vertex_input_state: Some(vertex_input_state),
            // How vertices are arranged into primitive shapes.
            // The default primitive shape is a triangle.
            input_assembly_state: Some(InputAssemblyState::default()),
            // How primitives are transformed and clipped to fit the framebuffer.
            viewport_state: Some(viewport_state),
            // How polygons are culled and converted into a raster of pixels.
            // The default value does not perform any culling.
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            // How multiple fragment shader samples are converted to a single pixel value.
            // The default value does not perform any multisampling.
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            // How pixel values are combined with the values already present in the framebuffer.
            // The default value overwrites the old value with the new one, without any blending.
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.color_attachment_formats.len() as u32,
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap();

    (pipeline, attachment_image_views, depth_buffer)
}

fn set_cursor_confinement(window: &Window, state: bool) {
    if state {
        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
            .unwrap();
    } else {
        window.set_cursor_grab(CursorGrabMode::None).unwrap();
    }
    // No cursor if mouse is confined
    window.set_cursor_visible(!state);
}

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
