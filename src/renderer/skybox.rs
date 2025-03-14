use crate::shader::{skybox_fs, skybox_vs};
use crate::{App, RenderContext};
use std::slice;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexType};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::Image;
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::DeviceSize;
use vulkano_taskgraph::command_buffer::RecordingCommandBuffer;
use vulkano_taskgraph::resource::HostAccessType;
use vulkano_taskgraph::{Id, Task, TaskContext, TaskResult};

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct SkyboxVertex {
    #[format(R32G32B32_SFLOAT)]
    pub a_position: [f32; 3],
}

impl From<[f32; 3]> for SkyboxVertex {
    fn from(a_position: [f32; 3]) -> Self {
        Self { a_position }
    }
}

pub struct SkyboxTask {
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    index_buffer_count: u32,
    skybox_sampler_set: Arc<DescriptorSet>,
    pipeline: Arc<GraphicsPipeline>,
}

impl SkyboxTask {
    pub(crate) fn new(
        app: &App,
        pipeline_layout: Arc<PipelineLayout>,
        swapchain_format: Format,
        skybox_image_id: Id<Image>,
        sampler: Arc<Sampler>,
    ) -> Self {
        let (vertex_buffer_id, index_buffer_id, index_buffer_count) = {
            let vertices = [
                SkyboxVertex {
                    a_position: [-1.0, -1.0, -1.0],
                },
                SkyboxVertex {
                    a_position: [1.0, -1.0, -1.0],
                },
                SkyboxVertex {
                    a_position: [1.0, 1.0, -1.0],
                },
                SkyboxVertex {
                    a_position: [-1.0, 1.0, -1.0],
                },
                SkyboxVertex {
                    a_position: [-1.0, -1.0, 1.0],
                },
                SkyboxVertex {
                    a_position: [1.0, -1.0, 1.0],
                },
                SkyboxVertex {
                    a_position: [1.0, 1.0, 1.0],
                },
                SkyboxVertex {
                    a_position: [-1.0, 1.0, 1.0],
                },
            ];

            let indices: Vec<u32> = [
                1, 2, 0, 2, 3, 0, 6, 2, 1, 1, 5, 6, 6, 5, 4, 4, 7, 6, 6, 3, 2, 7, 3, 6, 3, 7, 0, 7,
                4, 0, 5, 1, 0, 4, 5, 0,
            ]
            .into_iter()
            .collect();

            let index_buffer_count = indices.len() as u32;

            let vertex_buffer_id = app
                .resources
                .create_buffer(
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::for_value(vertices.as_slice()).unwrap(),
                )
                .unwrap();

            app.resources
                .flight(app.flight_id)
                .unwrap()
                .wait(None)
                .unwrap();
            unsafe {
                vulkano_taskgraph::execute(
                    app.context.graphics_queue(),
                    &app.resources,
                    app.flight_id,
                    |_cbf, tcx| {
                        tcx.write_buffer::<[SkyboxVertex]>(vertex_buffer_id, ..)?
                            .copy_from_slice(&vertices);
                        Ok(())
                    },
                    [(vertex_buffer_id, HostAccessType::Write)],
                    [],
                    [],
                )
            }
            .unwrap();

            let index_buffer_id = app
                .resources
                .create_buffer(
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    DeviceLayout::for_value(indices.as_slice()).unwrap(),
                )
                .unwrap();

            app.resources
                .flight(app.flight_id)
                .unwrap()
                .wait(None)
                .unwrap();
            unsafe {
                vulkano_taskgraph::execute(
                    app.context.graphics_queue(),
                    &app.resources,
                    app.flight_id,
                    |_cbf, tcx| {
                        tcx.write_buffer::<[u32]>(index_buffer_id, ..)?
                            .copy_from_slice(&indices);
                        Ok(())
                    },
                    [(index_buffer_id, HostAccessType::Write)],
                    [],
                    [],
                )
            }
            .unwrap();

            (vertex_buffer_id, index_buffer_id, index_buffer_count)
        };

        let pipeline = {
            let vs = skybox_vs::load(pipeline_layout.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = skybox_fs::load(pipeline_layout.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = SkyboxVertex::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain_format)],
                depth_attachment_format: None,
                ..Default::default()
            };

            GraphicsPipeline::new(
                pipeline_layout.device().clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        let skybox_sampler_set = {
            let state = app.resources.image(skybox_image_id).unwrap();
            let texture = state.image();

            DescriptorSet::new(
                app.descriptor_set_allocator.clone(),
                pipeline_layout.set_layouts()[0].clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    ImageView::new(
                        texture.clone(),
                        ImageViewCreateInfo {
                            view_type: ImageViewType::Cube,
                            ..ImageViewCreateInfo::from_image(texture)
                        },
                    )
                    .unwrap(),
                    sampler.clone(),
                )],
                [],
            )
            .unwrap()
        };

        Self {
            vertex_buffer_id,
            index_buffer_id,
            index_buffer_count,
            skybox_sampler_set,
            pipeline,
        }
    }
}

impl Task for SkyboxTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let pipeline_layout = self.pipeline.layout();

        cbf.set_viewport(0, slice::from_ref(&rcx.viewport))?;
        cbf.bind_pipeline_graphics(&self.pipeline)?;
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[])?;
        cbf.bind_index_buffer(
            self.index_buffer_id,
            0,
            (self.index_buffer_count as usize * size_of::<u32>()) as DeviceSize,
            IndexType::U32,
        )?;

        cbf.as_raw().bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline_layout,
            0,
            &[self.skybox_sampler_set.as_raw()],
            &[],
        )?;

        let aspect_ratio = rcx.viewport.extent[0] / rcx.viewport.extent[1];
        let proj = rcx.camera.projection_matrix(aspect_ratio);
        let view = rcx.camera.view_matrix();
        let view_proj = proj * view;
        cbf.push_constants(
            pipeline_layout,
            0,
            &skybox_vs::Camera {
                u_ViewProjectionMatrix: view_proj.into(),
            },
        )?;

        cbf.draw_indexed(self.index_buffer_count, 1, 0, 0, 0)?;

        Ok(())
    }
}
