use crate::gltf::{CombinedVertex, Object};
use crate::shader::{fs, RenderType, SpecializationConstants};
use crate::PrimitiveDrawInfo;
use cgmath::{Matrix, Matrix4, SquareMatrix};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{Device, DeviceExtensions, DeviceFeatures};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
    ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{DescriptorBindingRequirements, EntryPoint, ShaderModule, ShaderStages};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::VulkanoWindows;
use winit::window::WindowId;

pub struct Renderer {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub allocators: Allocators,
    pub pipelines: HashMap<SpecializationConstants, PipelineContext>,
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
}

pub struct PipelineContext {
    pub pipeline: Arc<GraphicsPipeline>,
    pub material_sets: HashMap<usize, (Arc<DescriptorSet>, Arc<DescriptorSet>)>,
}

impl PipelineContext {
    pub fn render(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline_layout: &Arc<PipelineLayout>,
        primitives: &[PrimitiveDrawInfo],
        objects: &[Object],
    ) {
        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        for (&prim_idx, (mat_set, tex_set)) in &self.material_sets {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline_layout.clone(),
                    3,
                    mat_set.clone(),
                )
                .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline_layout.clone(),
                    4,
                    tex_set.clone(),
                )
                .unwrap();

            let prim = &primitives[prim_idx];
            for object in prim.object_ids.iter().filter_map(|idx| {
                let object = &objects[*idx];
                object.enabled.then_some(object)
            }) {
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
                    .push_constants(pipeline_layout.clone(), 0, data)
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

        let pipelines = HashMap::new();

        let rcx = None;

        Self {
            context,
            windows,
            allocators,
            pipelines,
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

    pub fn add_pipeline(
        &mut self,
        spec: SpecializationConstants,
        pipeline_layout: Arc<PipelineLayout>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> &mut PipelineContext {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let swapchain_format = window_renderer.swapchain_format();

        self.pipelines.entry(spec).or_insert_with(|| {
            let constants: Vec<_> = spec.into();
            let vs = vs
                .specialize(constants.clone().into_iter().collect())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs
                .specialize(constants.clone().into_iter().collect())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let (color_blend_state, depth_stencil_state, cull_mode) =
                if spec.material_constants.render_type() == RenderType::Translucent {
                    let color_blend_state = ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: true,
                        },
                    );
                    let depth_stencil_state = DepthStencilState {
                        depth: Some(DepthState::default()),
                        ..Default::default()
                    };

                    (color_blend_state, depth_stencil_state, CullMode::None)
                } else {
                    let color_blend_state = ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    );
                    let depth_stencil_state = DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    };

                    (color_blend_state, depth_stencil_state, CullMode::Back)
                };

            let pipeline = build_pipeline::<CombinedVertex>(
                self.context.device().clone(),
                swapchain_format,
                pipeline_layout,
                vs,
                fs,
                color_blend_state,
                depth_stencil_state,
                cull_mode,
            );
            let material_sets = HashMap::new();

            PipelineContext {
                pipeline,
                material_sets,
            }
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_pipeline<V: Vertex>(
    device: Arc<Device>,
    swapchain_format: Format,
    layout: Arc<PipelineLayout>,
    vs: EntryPoint,
    fs: EntryPoint,
    color_blend_state: ColorBlendState,
    depth_stencil_state: DepthStencilState,
    cull_mode: CullMode,
) -> Arc<GraphicsPipeline> {
    // Automatically generate a vertex input state from the vertex shader's input
    // interface, that takes a single vertex buffer containing `Vertex` structs.
    let vertex_input_state = V::per_vertex().definition(&vs).unwrap();

    // Make a list of the shader stages that the pipeline will have.
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

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
        device,
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
                cull_mode,
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

        Self {
            attachment_image_views,
            depth_image_view,
            pipeline_layout,
        }
    }
}
