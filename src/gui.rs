use cgmath::num_traits::Float;
use hsv::hsv_to_rgb;
use imgui::{Condition, StyleColor};
use std::ffi::CString;

use imgui::sys;

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
    val: [&mut f32; 3],
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
            color.0 as f32 / 255.0,
            color.1 as f32 / 255.0,
            color.2 as f32 / 255.0,
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
