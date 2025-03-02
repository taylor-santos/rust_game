use cgmath::{
    BaseFloat, Euler, Matrix, Matrix3, Matrix4, Quaternion, Rad, SquareMatrix, Vector3, Zero,
};
use std::ops::Neg;

#[derive(Debug, Copy, Clone)]
pub struct Transform<T> {
    pub position: Vector3<T>,
    pub rotation: Euler<Rad<T>>,
    pub scale: Vector3<T>,
    pub skew: Vector3<T>,
}

impl<T> From<Matrix4<T>> for Transform<T>
where
    T: BaseFloat,
{
    fn from(matrix: Matrix4<T>) -> Transform<T> {
        let position = matrix.w.truncate();
        let rzs = Matrix3::from_cols(
            matrix.x.truncate(),
            matrix.y.truncate(),
            matrix.z.truncate(),
        );
        let mut zs = cholesky(rzs.transpose() * rzs);
        let mut scale = Vector3::new(zs.x.x, zs.y.y, zs.z.z);
        let zst = zs.transpose();
        let shears =
            Matrix3::from_cols(zst.x / scale.x, zst.y / scale.y, zst.z / scale.z).transpose();
        let skew = Vector3::new(shears.y.x, shears.z.x, shears.z.y);
        let mut rot_mat = rzs * zs.invert().expect("Matrix is not invertible");
        if rot_mat.determinant().is_sign_negative() {
            scale.x = scale.x.neg();
            zs.x = zs.x.neg();
            rot_mat = rzs * zs.invert().expect("Matrix is not invertible");
        }

        let rotation = Quaternion::from(rot_mat).into();

        Self {
            position,
            rotation,
            scale,
            skew,
        }
    }
}

impl<T> From<Transform<T>> for Matrix4<T>
where
    T: BaseFloat,
{
    fn from(transform: Transform<T>) -> Matrix4<T> {
        let rot_mat: Matrix3<T> = Quaternion::from(transform.rotation).into();
        let mut skew_mat = Matrix3::identity();
        skew_mat.y.x = transform.skew.x;
        skew_mat.z.x = transform.skew.y;
        skew_mat.z.y = transform.skew.z;
        let scale_mat = Matrix3::from_diagonal(transform.scale);
        let m: Matrix4<T> = (rot_mat * scale_mat * skew_mat).into();
        let t = Matrix4::from_translation(transform.position);
        t * m
    }
}

fn cholesky<T>(a: Matrix3<T>) -> Matrix3<T>
where
    T: BaseFloat,
{
    let mut l = Matrix3::<T>::zero();
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}
