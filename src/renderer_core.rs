use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::time::Duration;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::{Device, DeviceExtensions, DeviceFeatures, DeviceOwned, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{max_mip_levels, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::{PresentMode, Surface, SwapchainCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;

pub struct RendererCore {
    context: VulkanoContext,
    windows: VulkanoWindows,
    allocators: Allocators,
    swapchain_manager: Option<SwapchainManager>,
}

pub struct SwapchainManager {
    image_views: Vec<SwapchainImages>,
}

pub struct SwapchainImages {
    color: Arc<ImageView>,
    depth: Arc<ImageView>,
    opaque: Arc<ImageView>,
    tonemap: Arc<ImageView>,
}

pub struct RendererFrame<'a> {
    future: Option<Box<dyn GpuFuture>>,
    command_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    swapchain_images: &'a SwapchainImages,
    queue: Arc<Queue>,
    renderer: &'a mut VulkanoWindowRenderer,
    image_index: usize,
}

pub struct RendererPass<'a> {
    command_builder: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

struct Allocators {
    memory: Arc<StandardMemoryAllocator>,
    descriptor_set: Arc<StandardDescriptorSetAllocator>,
    command_buffer: Arc<StandardCommandBufferAllocator>,
    uniform_buffer: SubbufferAllocator,
}

impl RendererCore {
    pub fn new() -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: DeviceFeatures {
                dynamic_rendering: true,
                ..Default::default()
            },
            device_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..Default::default()
            },
            ..Default::default()
        });

        let windows = VulkanoWindows::default();

        let allocators = Allocators::new(context.device().clone());

        Self {
            context,
            windows,
            allocators,
            swapchain_manager: None,
        }
    }

    pub fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_descriptor: WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo),
    ) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        let window_id = self.windows.create_window(
            event_loop,
            &self.context,
            &window_descriptor,
            swapchain_create_info_modify,
        );

        let renderer = self.windows.get_renderer(window_id).unwrap();
        let color_views = renderer.swapchain_image_views();

        self.swapchain_manager.replace(SwapchainManager::new(
            color_views.to_vec(),
            &self.allocators.memory,
        ));
    }

    pub fn resize(&mut self) {
        self.windows
            .get_primary_renderer_mut()
            .map(VulkanoWindowRenderer::resize);
    }

    pub fn redraw(&mut self) {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        renderer.window().request_redraw();
    }

    pub fn begin_frame(&mut self) -> RendererFrame {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        let future = renderer
            .acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                self.swapchain_manager.replace(SwapchainManager::new(
                    swapchain_images.to_vec(),
                    &self.allocators.memory,
                ));
            })
            .unwrap();

        let image_index = renderer.image_index() as usize;
        dbg!(image_index);

        let mut command_builder = AutoCommandBufferBuilder::primary(
            self.allocators.command_buffer.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let window_size = renderer.window().inner_size();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        command_builder
            .set_viewport(0, std::iter::once(viewport).collect())
            .unwrap();

        let swapchain_images = &self.swapchain_manager.as_ref().unwrap().image_views[image_index];

        let queue = self.context.graphics_queue().clone();

        RendererFrame {
            future: Some(future),
            command_builder: Some(command_builder),
            swapchain_images,
            queue,
            renderer,
            image_index,
        }
    }

    pub fn device(&self) -> Arc<Device> {
        self.context.device().clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.context.graphics_queue().clone()
    }

    pub fn surface(&self) -> Arc<Surface> {
        self.windows.get_primary_renderer().unwrap().surface()
    }

    pub fn swapchain_format(&self) -> Format {
        self.windows
            .get_primary_renderer()
            .unwrap()
            .swapchain_format()
    }
}

impl RendererFrame<'_> {
    pub fn opaque_pass(&mut self) -> RendererPass {
        self.command_builder
            .as_mut()
            .unwrap()
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(ClearValue::Float([1.0, 0.0, 1.0, 1.0])),
                    ..RenderingAttachmentInfo::image_view(self.swapchain_images.color.clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(1.0f32.into()),
                    ..RenderingAttachmentInfo::image_view(self.swapchain_images.depth.clone())
                }),
                ..Default::default()
            })
            .unwrap();

        RendererPass {
            command_builder: self.command_builder.as_mut().unwrap(),
        }
    }
}

impl Drop for RendererFrame<'_> {
    fn drop(&mut self) {
        let command_buffer = self.command_builder.take().unwrap().build().unwrap();

        let future = self
            .future
            .take()
            .unwrap()
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence()
            .boxed();

        self.renderer.present(future, false);
    }
}

impl RendererPass<'_> {
    pub fn bind_pipeline(&mut self, pipeline: Arc<GraphicsPipeline>) {
        self.command_builder
            .bind_pipeline_graphics(pipeline)
            .unwrap();
    }
}

impl Drop for RendererPass<'_> {
    fn drop(&mut self) {
        self.command_builder.end_rendering().unwrap();
    }
}

impl SwapchainManager {
    pub fn new(
        color_views: Vec<Arc<ImageView>>,
        memory_alloc: &Arc<StandardMemoryAllocator>,
    ) -> Self {
        let image_views = color_views
            .into_iter()
            .map(|color| {
                let extent = color.image().extent();

                let depth = {
                    let image = Image::new(
                        memory_alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::D32_SFLOAT,
                            extent,
                            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                    )
                    .unwrap();
                    ImageView::new_default(image).unwrap()
                };

                let opaque = {
                    let mip_levels = max_mip_levels(extent);
                    let image = Image::new(
                        memory_alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R16G16B16A16_SFLOAT,
                            extent,
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
                    ImageView::new_default(image).unwrap()
                };

                let tonemap = {
                    let image = Image::new(
                        memory_alloc.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R16G16B16A16_SFLOAT,
                            extent,
                            usage: ImageUsage::COLOR_ATTACHMENT
                                | ImageUsage::SAMPLED
                                | ImageUsage::TRANSFER_SRC
                                | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                    )
                    .unwrap();
                    ImageView::new_default(image).unwrap()
                };

                SwapchainImages {
                    color,
                    depth,
                    opaque,
                    tonemap,
                }
            })
            .collect();

        Self { image_views }
    }
}

impl Allocators {
    fn new(device: Arc<Device>) -> Self {
        let memory = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let command_buffer = Arc::new(StandardCommandBufferAllocator::new(
            device,
            Default::default(),
        ));

        let uniform_buffer = SubbufferAllocator::new(
            memory.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            memory,
            descriptor_set,
            command_buffer,
            uniform_buffer,
        }
    }
}
