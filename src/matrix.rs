use std::{
    fmt,
    mem::ManuallyDrop,
    ops::{Index, IndexMut},
    ptr::NonNull,
    usize,
};
use thiserror::Error;

// NOTE: N * row + col

#[derive(Debug, Error)]
enum MatrixError {
    #[error("InvalidIndex: {0}")]
    IndexOutOfBounds(String),
}

type MatrixResult<T> = Result<T, MatrixError>;

#[derive(Debug, Clone, Copy)]
struct Dimension {
    rows: usize,
    cols: usize,
}

trait ViewType {}
struct Row;
impl ViewType for Row {}
struct Col;
impl ViewType for Col {}
struct Mat;
impl ViewType for Mat {}

// TODO: Remove ViewType, imitate Matrix instead MatrixView<M,N,T>
//
/// An immutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixView` will create and return new matricies. If in-place operations
/// are required, then they should be done on the raw `Matrix` itself.
#[derive(Clone, Copy)]
struct MatrixView<T> {
    data: NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<T> Index<(usize, usize)> for MatrixView<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let num_rows = self.dimension.rows;
        let num_cols = self.dimension.cols;
        if index.0 >= num_rows {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The row index must be less than {}",
                    num_rows
                ))
                .to_string()
            );
        } else if index.1 >= num_cols {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The column index must be less than {}",
                    num_cols
                ))
                .to_string()
            );
        }

        unsafe {
            let row = index.0 + self.start_row;
            let col = index.1 + self.start_col;
            self.data
                .as_ptr()
                .add(num_cols * row + col)
                .as_ref()
                .unwrap()
        }
    }
}

/// A `M x N` matrix containing elements of type `T`.
///
/// ## Note
/// All operations on a `Matrix` are done in-place; use `MatrixView` if the original matrix should
/// remain unchanged.
struct Matrix<const M: usize, const N: usize, T = f32> {
    data: NonNull<T>,
}

impl<const M: usize, const N: usize, T: Default + Clone> Matrix<M, N, T> {
    fn new() -> Self {
        let data = {
            let mut elems = ManuallyDrop::new(vec![T::default(); M * N]);
            NonNull::new(elems.as_mut_ptr()).unwrap()
        };

        Self { data }
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    const fn size(&self) -> (usize, usize) {
        (M, N)
    }

    fn row(&self, index: usize) -> MatrixResult<MatrixView<T>> {
        if index >= M {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The row index cannot be greater than or equal to {}",
                M
            )));
        }

        Ok(MatrixView {
            data: self.data.clone(),
            start_row: index,
            start_col: 0,
            dimension: Dimension { rows: 1, cols: N },
        })
    }

    fn col(&self, index: usize) -> MatrixResult<MatrixView<T>> {
        if index >= N {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The column index cannot be greater than or equal to {}",
                N
            )));
        }

        Ok(MatrixView {
            data: self.data.clone(),
            start_row: 0,
            start_col: index,
            dimension: Dimension { rows: M, cols: 1 },
        })
    }

    fn view(
        &self,
        start_row: usize,
        start_col: usize,
        dimension: Dimension,
    ) -> MatrixResult<MatrixView<T>> {
        // FIXME: Write error messages
        if start_row >= M {
            panic!()
        } else if start_col >= N {
            panic!()
        } else if dimension.rows > M {
            panic!()
        } else if dimension.cols > N {
        }

        Ok(MatrixView {
            data: self.data.clone(),
            start_row,
            start_col,
            dimension,
        })
    }
}

impl<const M: usize, const N: usize, T> Drop for Matrix<M, N, T> {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.data.as_mut(), M * N, M * N);
        }
    }
}

impl<const M: usize, const N: usize, T> fmt::Debug for Matrix<M, N, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_args!("Matrix ({} x {}) [\n", M, N))?;
        todo!()
    }
}

impl<const M: usize, const N: usize, T> Index<(usize, usize)> for Matrix<M, N, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= M {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The row index must be less than {}", M))
                    .to_string()
            );
        } else if index.1 >= N {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The column index must be less than {}", N))
                    .to_string()
            );
        }

        unsafe {
            self.data
                .as_ptr()
                .add(N * index.0 + index.1)
                .as_ref()
                .unwrap()
        }
    }
}

impl<const M: usize, const N: usize, T> IndexMut<(usize, usize)> for Matrix<M, N, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 >= M {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The row index must be less than {}", M))
                    .to_string()
            );
        } else if index.1 >= N {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The column index must be less than {}", N))
                    .to_string()
            );
        }

        unsafe {
            self.data
                .as_ptr()
                .add(N * index.0 + index.1)
                .as_mut()
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_init() -> MatrixResult<()> {
        let mat: Matrix<2, 2> = Matrix::new();
        assert_eq!(mat.size(), (2, 2));
        Ok(())
    }
}
