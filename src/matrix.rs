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

    #[error("InvalidShape: {0}")]
    InvalidShape(String),
}

type MatrixResult<T> = Result<T, MatrixError>;

#[derive(Debug, Clone, Copy)]
struct Dimension {
    rows: usize,
    cols: usize,
}

/// An immutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixView` will create and return new matricies. If
/// in-place operations are required, then they should be done on a
/// `MatrixViewMut` instead.
#[derive(Clone, Copy)]
struct MatrixView<'a, T> {
    data: &'a NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<'a, T> Index<(usize, usize)> for MatrixView<'a, T> {
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

/// A mutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixViewMut` are done in-place; use `MatrixView` if
/// the original matrix should remain unchanged.
struct MatrixViewMut<'a, T> {
    data: &'a mut NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<'a, T> Index<(usize, usize)> for MatrixViewMut<'a, T> {
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

impl<'a, T> IndexMut<(usize, usize)> for MatrixViewMut<'a, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
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
                .as_mut()
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

    const fn is_square(&self) -> bool {
        M == N
    }

    fn row<'a>(&'a self, index: usize) -> MatrixResult<MatrixView<'a, T>> {
        if index >= M {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The row index cannot be greater than or equal to {}",
                M
            )));
        }

        Ok(MatrixView {
            data: &self.data,
            start_row: index,
            start_col: 0,
            dimension: Dimension { rows: 1, cols: N },
        })
    }

    fn col<'a>(&'a self, index: usize) -> MatrixResult<MatrixView<'a, T>> {
        if index >= N {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The column index cannot be greater than or equal to {}",
                N
            )));
        }

        Ok(MatrixView {
            data: &self.data,
            start_row: 0,
            start_col: index,
            dimension: Dimension { rows: M, cols: 1 },
        })
    }

    fn view<'a>(
        &'a self,
        start_row: usize,
        start_col: usize,
        dimension: Dimension,
    ) -> MatrixResult<MatrixView<'a, T>> {
        if start_row >= M {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start row must be less than {}",
                M
            )));
        } else if start_col >= N {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start col must be less than {}",
                N
            )));
        } else if dimension.rows > M {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more rows than `self".into(),
            ));
        } else if dimension.cols > N {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more columns than `self".into(),
            ));
        }

        Ok(MatrixView {
            data: &self.data,
            start_row,
            start_col,
            dimension,
        })
    }

    fn view_mut<'a>(
        &'a mut self,
        start_row: usize,
        start_col: usize,
        dimension: Dimension,
    ) -> MatrixResult<MatrixViewMut<'a, T>> {
        if start_row >= M {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start row must be less than {}",
                M
            )));
        } else if start_col >= N {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start col must be less than {}",
                N
            )));
        } else if dimension.rows > M {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more rows than `self".into(),
            ));
        } else if dimension.cols > N {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more columns than `self".into(),
            ));
        }

        Ok(MatrixViewMut {
            data: &mut self.data,
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
