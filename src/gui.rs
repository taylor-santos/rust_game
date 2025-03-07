use cgmath::num_traits::Float;
use hsv::hsv_to_rgb;
use imgui::{Condition, StyleColor, TableFlags, TreeNodeFlags, WindowFlags};
use std::ffi::CString;

use crate::gltf::Object;
use crate::material::Material;
use crate::transform::Transform;
use imgui::sys;

pub struct Gui {
    pub selected_object: Option<usize>,
    pub selected_material: Option<usize>,
}

// Display a window titled "Transform" to manipulate the transformation properties of the given object.
// Returns Some(transform) if any properties were changed
pub fn object_transform(object: &Object, ui: &imgui::Ui) -> Option<Transform<f32>> {
    ui.window("Transform")
        .build(|| {
            let mut transform = object.local_transform;
            let mut changed = false;
            if ui.collapsing_header("Position", TreeNodeFlags::DEFAULT_OPEN)
                && drag_vec3(
                    &mut [
                        &mut transform.position.x,
                        &mut transform.position.y,
                        &mut transform.position.z,
                    ],
                    ui,
                    "position",
                    "%.2f",
                )
            {
                changed = true;
            }
            if ui.collapsing_header("Rotation", TreeNodeFlags::empty())
                && drag_vec3(
                    &mut [
                        &mut transform.rotation.x.0,
                        &mut transform.rotation.y.0,
                        &mut transform.rotation.z.0,
                    ],
                    ui,
                    "rotation",
                    "%.2f",
                )
            {
                changed = true;
            }
            if ui.collapsing_header("Scale", TreeNodeFlags::empty())
                && drag_vec3(
                    &mut [
                        &mut transform.scale.x,
                        &mut transform.scale.y,
                        &mut transform.scale.z,
                    ],
                    ui,
                    "scale",
                    "%.2f",
                )
            {
                changed = true;
            }
            if ui.collapsing_header("Skew", TreeNodeFlags::empty())
                && drag_vec3(
                    &mut [
                        &mut transform.skew.x,
                        &mut transform.skew.y,
                        &mut transform.skew.z,
                    ],
                    ui,
                    "skew",
                    "%.2f",
                )
            {
                changed = true;
            }

            changed.then_some(transform)
        })
        .flatten()
}

pub fn mesh_materials(
    mat_ids: &[usize],
    materials: &[Material],
    selected: &mut Option<usize>,
    ui: &imgui::Ui,
) {
    ui.window("Materials")
        .flags(WindowFlags::NO_FOCUS_ON_APPEARING)
        .build(|| {
            if let Some(_token) =
                ui.begin_table_with_flags("Materials", 1, TableFlags::SIZING_FIXED_FIT)
            {
                ui.table_setup_column("Material");
                ui.table_headers_row();
                for mat_idx in mat_ids.iter().copied() {
                    let mat = &materials[mat_idx];
                    ui.table_next_row();

                    ui.table_next_column();
                    let color = mat.pbr_metallic_roughness.base_color_factor;
                    color_square(ui, color, "");
                    ui.same_line();
                    let name = mat.name.as_deref().unwrap_or("Unnamed Material");
                    let is_selected = selected.is_some_and(|s| s == mat_idx);
                    if ui
                        .selectable_config(name)
                        .selected(is_selected)
                        .span_all_columns(true)
                        .build()
                    {
                        selected.replace(mat_idx);
                    }
                }
            }
        });
}

pub fn drag_float(
    val: &mut f32,
    min: f32,
    max: f32,
    label: impl Into<Vec<u8>>,
    fmt: impl Into<Vec<u8>>,
) -> bool {
    unsafe {
        sys::igDragFloat(
            CString::new(label).unwrap().as_ptr(),
            val,
            0.01,
            min,
            max,
            CString::new(fmt).unwrap().as_ptr(),
            0,
        )
    }
}

pub fn color_square(ui: &imgui::Ui, color: [f32; 4], label: &str) {
    let _t = ui.push_style_color(StyleColor::Button, color);
    let _t = ui.push_style_color(StyleColor::ButtonHovered, color);
    let _t = ui.push_style_color(StyleColor::ButtonActive, color);
    ui.button_with_size(label, [ui.frame_height(), ui.frame_height()]);
}

pub fn drag_vec3(
    val: &mut [&mut f32; 3],
    ui: &imgui::Ui,
    label: &str,
    fmt: impl Into<Vec<u8>> + Copy,
) -> bool {
    let axes = [
        ("X", hsv_to_rgb(0., 0.6, 0.6)),
        ("Y", hsv_to_rgb(130., 0.6, 0.6)),
        ("Z", hsv_to_rgb(211., 0.6, 0.6)),
    ];
    let mut changed = false;
    for (i, &(axis, color)) in axes.iter().enumerate() {
        let color = [
            f32::from(color.0) / 255.0,
            f32::from(color.1) / 255.0,
            f32::from(color.2) / 255.0,
            1.0,
        ];
        color_square(ui, color, axis);
        ui.same_line();

        if drag_float(
            val[i],
            f32::neg_infinity(),
            f32::infinity(),
            format!("##{label}_{axis}"),
            fmt,
        ) {
            changed = true;
        }
    }

    changed
}

pub fn material_property<T, F>(property: &mut Option<T>, label: &str, ui: &imgui::Ui, f: F) -> bool
where
    T: Default,
    F: Fn(&mut T) -> bool,
{
    if let Some(_token) = ui
        .tree_node_config(label)
        .opened(property.is_some(), Condition::Always)
        .push()
    {
        let mut changed = property.is_none();
        changed |= f(property.get_or_insert_default());
        changed
    } else {
        property.take().is_some()
    }
}
