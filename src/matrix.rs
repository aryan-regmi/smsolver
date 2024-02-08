use core::{fmt, slice};
use std::{
    fmt::Write,
    mem,
    ops::{Index, IndexMut},
    ptr::NonNull,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum MatrixError {
    #[error("Matrix allocation failed")]
    AllocationError,
}

type MatrixResult<T> = Result<T, MatrixError>;

struct Matrix<'a, const M: usize, const N: usize> {
    // Index = N*row + (M-1)*col
    data: &'a [f32],
    alloc: NonNull<f32>,
}

impl<'a, const M: usize, const N: usize> Matrix<'a, M, N> {
    fn from_vec(v: Vec<f32>) -> MatrixResult<Self> {
        let (alloc, data) = unsafe {
            let mut v = mem::ManuallyDrop::new(v);
            let data = v.as_mut_ptr();
            if data.is_null() {
                return Err(MatrixError::AllocationError);
            }
            let alloc = NonNull::new(data).ok_or(MatrixError::AllocationError)?;
            (alloc, slice::from_raw_parts(data, M * N))
        };
        Ok(Self { data, alloc })
    }
}

impl<'a, const M: usize, const N: usize> Drop for Matrix<'a, M, N> {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.alloc.as_ptr(), M * N, M * N);
        }
    }
}

// impl<const M: usize, const N: usize> fmt::Debug for Matrix<M, N> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "Matrix<{}, {}> [\n", M, N)?;
//         // TODO: Impl
//
//         // for _col in 0..N {
//         //     for _row in 0..M {}
//         // }
//         write!(f, "]\n")
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_init() -> MatrixResult<()> {
        let mat = Matrix::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0])?;

        // dbg!(mat);

        Ok(())
    }
}
