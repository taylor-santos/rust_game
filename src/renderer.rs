use crate::gltf::{CombinedVertex, Object, TextureFormat};
use crate::shader::{fs, RenderType, SpecializationConstants};
use crate::{create_buffer, PrimitiveDrawInfo};
use cgmath::{Matrix, Matrix4, SquareMatrix};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Formatter;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferImageCopy, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{Device, DeviceExtensions, DeviceFeatures, Queue};
use vulkano::format::Format;
use vulkano::half::f16;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageFormatInfo,
    ImageSubresourceLayers, ImageType, ImageUsage,
};
use vulkano::instance::InstanceCreateInfo;
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
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{DescriptorBindingRequirements, EntryPoint, ShaderModule, ShaderStages};
use vulkano::DeviceSize;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::VulkanoWindows;
use winit::window::WindowId;

pub struct Renderer {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub allocators: Allocators,
    pub pipelines: HashMap<SpecializationConstants, PipelineContext>,
    pub format_conv: FormatGraph,
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

pub enum Pixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F16(Vec<f16>),
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

#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
pub struct TextureType {
    pub format: TextureFormat,
    pub is_srgb: bool,
}

impl std::fmt::Display for TextureType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.format {
            TextureFormat::R8 => write!(f, "R8")?,
            TextureFormat::R8G8 => write!(f, "R8G8")?,
            TextureFormat::R8G8B8 => write!(f, "R8G8B8")?,
            TextureFormat::R8G8B8A8 => write!(f, "R8G8B8A8")?,
            TextureFormat::R16 => write!(f, "R16")?,
            TextureFormat::R16G16 => write!(f, "R16G16")?,
            TextureFormat::R16G16B16 => write!(f, "R16G16B16")?,
            TextureFormat::R16G16B16A16 => write!(f, "R16G16B16A16")?,
            TextureFormat::R32G32B32FLOAT => write!(f, "R32G32B32FLOAT")?,
            TextureFormat::R32G32B32A32FLOAT => write!(f, "R32G32B32A32FLOAT")?,
        }
        if self.is_srgb {
            write!(f, "_SRGB")?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct FormatGraphNode {
    edges: HashMap<TextureType, FormatGraphEdge>,
    out: Option<Format>,
}

#[derive(Clone, Copy)]
struct FormatGraphEdge {
    pub from: TextureType,
    pub to: TextureType,
    pub conv: fn(&[u8]) -> Vec<u8>,
}

pub struct FormatGraph {
    nodes: HashMap<TextureType, FormatGraphNode>,
}

impl Into<TextureType> for Format {
    fn into(self) -> TextureType {
        match self {
            Format::R8_UNORM => TextureType {
                format: TextureFormat::R8,
                is_srgb: false,
            },
            Format::R8_SRGB => TextureType {
                format: TextureFormat::R8,
                is_srgb: true,
            },
            Format::R8G8_UNORM => TextureType {
                format: TextureFormat::R8G8,
                is_srgb: false,
            },
            Format::R8G8_SRGB => TextureType {
                format: TextureFormat::R8G8,
                is_srgb: true,
            },
            Format::R8G8B8_UNORM => TextureType {
                format: TextureFormat::R8G8B8,
                is_srgb: false,
            },
            Format::R8G8B8_SRGB => TextureType {
                format: TextureFormat::R8G8B8,
                is_srgb: true,
            },
            Format::R8G8B8A8_UNORM => TextureType {
                format: TextureFormat::R8G8B8A8,
                is_srgb: false,
            },
            Format::R8G8B8A8_SRGB => TextureType {
                format: TextureFormat::R8G8B8A8,
                is_srgb: true,
            },
            Format::R16_UNORM => TextureType {
                format: TextureFormat::R16,
                is_srgb: false,
            },
            Format::R16G16_UNORM => TextureType {
                format: TextureFormat::R16G16,
                is_srgb: false,
            },
            Format::R16G16B16_UNORM => TextureType {
                format: TextureFormat::R16G16B16,
                is_srgb: false,
            },
            Format::R16G16B16A16_UNORM => TextureType {
                format: TextureFormat::R16G16B16A16,
                is_srgb: false,
            },
            Format::R32G32B32_SFLOAT => TextureType {
                format: TextureFormat::R32G32B32FLOAT,
                is_srgb: false,
            },
            Format::R32G32B32A32_SFLOAT => TextureType {
                format: TextureFormat::R32G32B32A32FLOAT,
                is_srgb: false,
            },
            _ => panic!("Unsupported format: {:?}", self),
        }
    }
}

impl FormatGraph {
    fn new(supported_formats: Vec<Format>) -> FormatGraph {
        dbg!(&supported_formats);
        let mut nodes = HashMap::new();

        for format in supported_formats {
            nodes.insert(
                format.into(),
                FormatGraphNode {
                    edges: HashMap::new(),
                    out: Some(format),
                },
            );
        }

        for is_srgb in [false, true] {
            {
                let from = TextureType {
                    format: TextureFormat::R8G8B8,
                    is_srgb,
                };
                let to = TextureType {
                    format: TextureFormat::R8G8B8A8,
                    is_srgb,
                };
                nodes
                    .entry(from)
                    .or_insert_with(Default::default)
                    .edges
                    .insert(
                        to,
                        FormatGraphEdge {
                            from,
                            to,
                            conv: |pixels| {
                                pixels
                                    .chunks_exact(3)
                                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2], u8::MAX])
                                    .collect()
                            },
                        },
                    );
            }
            {
                let from = TextureType {
                    format: TextureFormat::R8G8,
                    is_srgb,
                };
                let to = TextureType {
                    format: TextureFormat::R8G8B8,
                    is_srgb,
                };
                nodes
                    .entry(from)
                    .or_insert_with(Default::default)
                    .edges
                    .insert(
                        to,
                        FormatGraphEdge {
                            from,
                            to,
                            conv: |pixels| {
                                pixels
                                    .chunks_exact(2)
                                    .flat_map(|chunk| [chunk[0], chunk[1], 0])
                                    .collect()
                            },
                        },
                    );
            }

            {
                let from = TextureType {
                    format: TextureFormat::R8,
                    is_srgb,
                };
                let to = TextureType {
                    format: TextureFormat::R8G8,
                    is_srgb,
                };
                nodes
                    .entry(from)
                    .or_insert_with(Default::default)
                    .edges
                    .insert(
                        to,
                        FormatGraphEdge {
                            from,
                            to,
                            conv: |pixels| {
                                pixels.into_iter().flat_map(|&pixel| [pixel, 0]).collect()
                            },
                        },
                    );
            }
            {
                let from = TextureType {
                    format: TextureFormat::R16G16B16,
                    is_srgb,
                };
                let to = TextureType {
                    format: TextureFormat::R8G8B8,
                    is_srgb,
                };
                nodes
                    .entry(from)
                    .or_insert_with(Default::default)
                    .edges
                    .insert(
                        to,
                        FormatGraphEdge {
                            from,
                            to,
                            conv: |pixels| {
                                pixels
                                    .chunks_exact(2)
                                    .map(|chunk| {
                                        (u16::from_le_bytes([chunk[0], chunk[1]]) / 257) as u8
                                    })
                                    .collect()
                            },
                        },
                    );
            }
        }

        FormatGraph { nodes }
    }

    pub fn convert(&self, from: TextureType, bytes: &[u8]) -> (Format, Vec<u8>) {
        let mut queue = vec![(&from, Vec::<FormatGraphEdge>::new())];
        let mut visited = HashSet::new();

        print!("Converting {}", from);

        while let Some((format, edges)) = queue.pop() {
            if let Some(node) = self.nodes.get(&format) {
                if let Some(out) = node.out {
                    let mut bytes = bytes.to_vec();
                    for edge in edges {
                        print!(" -> {}", edge.to);
                        bytes = (edge.conv)(&bytes);
                    }
                    println!(" -> {:?}", out);
                    return (out, bytes);
                }

                for (next, edge) in &node.edges {
                    if visited.insert(next) {
                        let new_convs = edges
                            .iter()
                            .copied()
                            .chain(std::iter::once(*edge))
                            .collect();
                        queue.push((next, new_convs));
                    }
                }
            }
        }
        panic!("No suitable conversion from {:?} found!", from);
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

        let supported_formats = [
            Format::R8_UNORM,
            Format::R8_SRGB,
            Format::R8G8_UNORM,
            Format::R8G8_SRGB,
            Format::R8G8B8_UNORM,
            Format::R8G8B8_SRGB,
            Format::R8G8B8A8_UNORM,
            Format::R8G8B8A8_SRGB,
            Format::R16_UNORM,
            Format::R16G16_UNORM,
            Format::R16G16B16_UNORM,
            Format::R16G16B16A16_UNORM,
            Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32_SFLOAT,
        ]
        .into_iter()
        .filter(|&format| {
            context
                .device()
                .physical_device()
                .image_format_properties(ImageFormatInfo {
                    format,
                    image_type: ImageType::Dim2d,
                    usage: ImageUsage::SAMPLED,
                    ..Default::default()
                })
                .unwrap()
                .is_some()
        })
        .collect::<Vec<_>>();

        let windows = VulkanoWindows::default();
        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        let allocators = Allocators::new(context.device());

        let pipelines = HashMap::new();

        let rcx = None;

        let format_conv = FormatGraph::new(supported_formats);

        Self {
            context,
            windows,
            allocators,
            pipelines,
            format_conv,
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
        format: Format,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> &mut PipelineContext {
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
            let (color_blend_state, depth_stencil_state) =
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
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    };

                    (color_blend_state, depth_stencil_state)
                } else {
                    let color_blend_state = ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    );
                    let depth_stencil_state = DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    };

                    (color_blend_state, depth_stencil_state)
                };

            let cull_mode = if spec.material_constants.DOUBLE_SIDED {
                CullMode::None
            } else {
                CullMode::Back
            };

            let pipeline = build_pipeline::<CombinedVertex>(
                self.context.device().clone(),
                format,
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

    fn upload_pixels<T: BufferContents + Send + Sync, I: IntoIterator<Item = T>>(
        &self,
        pixels: I,
        extent: [u32; 3],
        mip_levels: u32,
        array_layers: u32,
        format: Format,
        image: Arc<Image>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Box<dyn Error>>
    where
        I::IntoIter: ExactSizeIterator,
    {
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

                    buffer_offset += format.block_size()
                        * DeviceSize::from(array_layers * mip_width * mip_height);
                    // Each successive Mip level is 4x smaller than the last, each dimension must be divided by 2
                    mip_width /= 2;
                    mip_height /= 2;

                    region
                })
                .collect()
        };

        let upload_buffer = create_buffer(
            self.allocators.memory.clone(),
            self.allocators.command_buffer.clone(),
            self.context.graphics_queue(),
            BufferUsage::TRANSFER_SRC,
            pixels,
        );
        builder.copy_buffer_to_image(CopyBufferToImageInfo {
            regions,
            ..CopyBufferToImageInfo::buffer_image(upload_buffer, image)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_image(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pixels: Pixels,
        extent: [u32; 3],
        mip_levels: u32,
        array_layers: u32,
        format: Format,
    ) -> Result<Arc<ImageView>, Box<dyn Error>> {
        let flags = if array_layers == 6 {
            ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            ImageCreateFlags::empty()
        };

        let image = Image::new(
            self.allocators.memory.clone(),
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

        match pixels {
            Pixels::U8(pixels) => self.upload_pixels(
                pixels,
                extent,
                mip_levels,
                array_layers,
                format,
                image.clone(),
                builder,
            )?,
            Pixels::U16(pixels) => self.upload_pixels(
                pixels,
                extent,
                mip_levels,
                array_layers,
                format,
                image.clone(),
                builder,
            )?,
            Pixels::F16(pixels) => self.upload_pixels(
                pixels,
                extent,
                mip_levels,
                array_layers,
                format,
                image.clone(),
                builder,
            )?,
        }

        let view_type = if array_layers == 6 {
            ImageViewType::Cube
        } else {
            ImageViewType::Dim2d
        };

        let create_info = ImageViewCreateInfo {
            view_type,
            ..ImageViewCreateInfo::from_image(&image)
        };

        ImageView::new(image, create_info).map_err(Into::into)
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

#[allow(clippy::too_many_arguments)]
pub fn build_fullscreen_pipeline(
    device: Arc<Device>,
    swapchain_format: Format,
    layout: Arc<PipelineLayout>,
    vs: EntryPoint,
    fs: EntryPoint,
    color_blend_state: ColorBlendState,
    depth_stencil_state: DepthStencilState,
) -> Arc<GraphicsPipeline> {
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
            vertex_input_state: Some(VertexInputState::default()),
            stages: stages.into_iter().collect(),
            // How vertices are arranged into primitive shapes. The default primitive shape
            // is a triangle.
            input_assembly_state: Some(InputAssemblyState::default()),
            // How primitives are transformed and clipped to fit the framebuffer. We use a
            // resizable viewport, set to draw over the entire window.
            viewport_state: Some(ViewportState::default()),
            // How polygons are culled and converted into a raster of pixels. The default
            // value does not perform any culling.
            rasterization_state: Some(RasterizationState::default()),
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
