use crate::shader::{tonemap_fs, tonemap_vs};
use crate::{App, RenderContext};
use std::sync::Arc;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageUsage};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano_taskgraph::command_buffer::RecordingCommandBuffer;
use vulkano_taskgraph::{Id, Task, TaskContext, TaskResult};

pub struct TonemapTask {
    pipeline: Arc<GraphicsPipeline>,
    intermediate_image_descriptor_set: Arc<DescriptorSet>,
}

impl TonemapTask {
    pub fn new(
        app: &App,
        pipeline_layout: Arc<PipelineLayout>,
        swapchain_format: Format,
        intermediate_image_id: Id<Image>,
        sampler: Arc<Sampler>,
    ) -> Self {
        let pipeline = {
            let vs = tonemap_vs::load(pipeline_layout.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = tonemap_fs::load(pipeline_layout.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
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
                    vertex_input_state: Some(VertexInputState::default()),
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

        let intermediate_image_descriptor_set = {
            let state = app.resources.image(intermediate_image_id).unwrap();
            let image = state.image();
            let image_view = ImageView::new(
                image.clone(),
                ImageViewCreateInfo {
                    format: image.format(),
                    subresource_range: image.subresource_range(),
                    usage: ImageUsage::SAMPLED,
                    ..Default::default()
                },
            )
            .unwrap();

            DescriptorSet::new(
                app.descriptor_set_allocator.clone(),
                pipeline_layout.set_layouts()[0].clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0, image_view, sampler,
                )],
                [],
            )
            .unwrap()
        };

        Self {
            pipeline,
            intermediate_image_descriptor_set,
        }
    }
}

impl Task for TonemapTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.bind_pipeline_graphics(&self.pipeline)?;
        cbf.as_raw().bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline.layout(),
            0,
            &[self.intermediate_image_descriptor_set.as_raw()],
            &[],
        )?;

        cbf.draw(3, 1, 0, 0)?;

        Ok(())
    }
}
