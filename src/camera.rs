use std::f32::consts::{FRAC_PI_4, PI};
use std::ops::Neg;

use cgmath::num_traits::clamp;
use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Point3, Rad, Vector3};

#[derive(Debug)]
pub struct FirstPersonCamera {
    pub position: Point3<f32>,
    pub fov_y: Rad<f32>,
    pub near: f32,
    pub far: f32,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub speed: f32,
    pub sensitivity: f32,
}

impl FirstPersonCamera {
    pub fn new() -> FirstPersonCamera {
        FirstPersonCamera::default()
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw = (self.yaw + Rad::from(Deg(delta_x * self.sensitivity))) % Rad::full_turn();
        let min = Rad::turn_div_4().neg();
        let max = Rad::turn_div_4();
        self.pitch = clamp(
            self.pitch - Rad::from(Deg(delta_y * self.sensitivity)),
            min,
            max,
        );
    }

    pub fn move_forward(&mut self, delta_time: f32) {
        let forward = self.forward();
        self.position += forward * self.speed * delta_time;
    }

    pub fn move_backward(&mut self, delta_time: f32) {
        let forward = self.forward();
        self.position -= forward * self.speed * delta_time;
    }

    pub fn move_left(&mut self, delta_time: f32) {
        let right = self.right();
        self.position -= right * self.speed * delta_time;
    }

    pub fn move_right(&mut self, delta_time: f32) {
        let right = self.right();
        self.position += right * self.speed * delta_time;
    }

    fn forward(&self) -> Vector3<f32> {
        let pitch_cos = self.pitch.cos();
        Vector3::new(
            -self.yaw.sin() * pitch_cos,
            self.pitch.sin(),
            self.yaw.cos() * pitch_cos,
        )
    }

    fn up(&self) -> Vector3<f32> {
        let pitch_sin = self.pitch.sin();
        Vector3::new(
            self.yaw.sin() * pitch_sin,
            self.pitch.cos(),
            -self.yaw.cos() * pitch_sin,
        )
    }

    fn right(&self) -> Vector3<f32> {
        Vector3::new(-self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let forward = self.forward();
        let up = self.up();

        Matrix4::look_at_rh(self.position, self.position + forward, up)
    }

    pub fn projection_matrix(&self, aspect_ratio: f32) -> Matrix4<f32> {
        let proj = cgmath::perspective(self.fov_y, aspect_ratio, self.near, self.far);
        // Vulkan clip space has inverted Y and half Z, compared with OpenGL.
        // A corrective transformation is needed to make an OpenGL perspective matrix
        // work properly. See here for more info:
        // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
        #[rustfmt::skip]
        let correction = Matrix4::<f32>::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );

        correction * proj
    }
}

impl Default for FirstPersonCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 4.0),
            fov_y: Rad(FRAC_PI_4),
            near: 0.05,
            far: 1000.0,
            yaw: Rad(PI),
            pitch: Rad(0.0),
            speed: 5.0,
            sensitivity: 0.1,
        }
    }
}
