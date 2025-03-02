use crate::shader::{fs, vs};
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::{Device, DeviceExtensions, DeviceFeatures};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::PipelineLayout;
use vulkano::shader::{DescriptorBindingRequirements, ShaderModule, ShaderStages};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::VulkanoWindows;
use winit::window::WindowId;

pub struct Renderer {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub allocators: Allocators,
    pub rcx: Option<RendererContext>,
}

pub struct Allocators {
    pub memory: Arc<StandardMemoryAllocator>,
    pub descriptor_set: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer: Arc<StandardCommandBufferAllocator>,
    pub uniform_buffer: SubbufferAllocator,
}

pub struct RendererContext {
    pub attachment_image_views: Vec<Arc<ImageView>>,
    pub depth_image_view: Arc<ImageView>,
    pub pipeline_layout: Arc<PipelineLayout>,
    pub vertex_shader: Arc<ShaderModule>,
    pub fragment_shader: Arc<ShaderModule>,
}

impl Renderer {
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
        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        let allocators = Allocators::new(context.device());

        let rcx = None;

        Self {
            context,
            windows,
            allocators,
            rcx,
        }
    }

    fn build_pipeline_layout(
        &self,
        bindings: &[&[DescriptorType]],
        push_constant_size: u32,
    ) -> Arc<PipelineLayout> {
        let set_layouts = bindings
            .iter()
            .map(|set| {
                let bindings = set
                    .iter()
                    .enumerate()
                    .map(|(idx, &binding)| {
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
                    .collect();

                DescriptorSetLayout::new(
                    self.context.device().clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let push_constant_ranges = vec![PushConstantRange {
            stages: ShaderStages::all_graphics(),
            offset: 0,
            size: push_constant_size,
        }];

        PipelineLayout::new(
            self.context.device().clone(),
            PipelineLayoutCreateInfo {
                set_layouts,
                push_constant_ranges,
                ..Default::default()
            },
        )
        .unwrap()
    }
}

impl Allocators {
    fn new(device: &Arc<Device>) -> Self {
        let memory = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
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

impl RendererContext {
    pub fn new(renderer: &mut Renderer, window_id: WindowId) -> Self {
        let window_renderer = renderer.windows.get_renderer(window_id).unwrap();
        let window_size = window_renderer.window().inner_size();

        let attachment_image_views = window_renderer.swapchain_image_views().to_vec();

        let depth_image = Image::new(
            renderer.allocators.memory.clone(),
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
            ImageView::new_default(depth_image).expect("Failed to create depth image view");

        let pipeline_layout = renderer.build_pipeline_layout(
            &[
                // set = 0
                &[
                    DescriptorType::UniformBuffer, // binding = 0 uniform Constants
                ] as &[_],
                // set = 1
                &[
                    DescriptorType::UniformBuffer, // binding = 0 uniform Camera
                ],
                // set = 2
                &[DescriptorType::CombinedImageSampler; 6], // IBL Samplers
                // set = 3
                &[
                    DescriptorType::UniformBuffer, // binding = 0 uniform Material
                    DescriptorType::UniformBuffer, // binding = 1 uniform MatSamplers
                ],
                // set = 4
                &[DescriptorType::CombinedImageSampler; 21], // Texture Samplers
                // set = 5
                &[DescriptorType::CombinedImageSampler], // Framebuffer Sampler
            ],
            size_of::<fs::Object>() as u32,
        );

        let vertex_shader = vs::load(renderer.context.device().clone()).unwrap();

        let fragment_shader = fs::load(renderer.context.device().clone()).unwrap();

        Self {
            attachment_image_views,
            depth_image_view,
            pipeline_layout,
            vertex_shader,
            fragment_shader,
        }
    }
}
