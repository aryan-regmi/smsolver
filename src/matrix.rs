use std::{fmt, marker::PhantomData, mem::ManuallyDrop, ops::Index, ptr::NonNull, usize};
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

#[derive(Clone, Copy)]
struct MatrixView<T, V: ViewType = Row> {
    data: NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
    _marker: PhantomData<V>,
}

impl<T> Index<usize> for MatrixView<T, Row> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.dimension.cols {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The column index must be less than {}",
                    self.dimension.cols
                ))
                .to_string()
            );
        }

        unsafe {
            let col = self.start_col + index;
            self.data
                .as_ptr()
                .add(self.dimension.cols * self.start_row + col)
                .as_ref()
                .unwrap()
        }
    }
}

impl<T> Index<usize> for MatrixView<T, Col> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.dimension.rows {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The row index must be less than {}",
                    self.dimension.rows
                ))
                .to_string()
            );
        }

        unsafe {
            let row = index + self.start_row;
            self.data
                .as_ptr()
                .add(self.dimension.cols * row + self.start_col)
                .as_ref()
                .unwrap()
        }
    }
}

impl<T> Index<(usize, usize)> for MatrixView<T, Mat> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= self.dimension.rows {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The row index must be less than {}",
                    self.dimension.rows
                ))
                .to_string()
            );
        } else if index.1 >= self.dimension.cols {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!(
                    "The column index must be less than {}",
                    self.dimension.cols
                ))
                .to_string()
            );
        }

        unsafe {
            let row = index.0 + self.start_row;
            let col = index.1 + self.start_col;
            self.data
                .as_ptr()
                .add(self.dimension.cols * row + col)
                .as_ref()
                .unwrap()
        }
    }
}

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
    fn row(&self, index: usize) -> MatrixResult<MatrixView<T, Row>> {
        if index >= M {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The row index cannot be greater than or equal to {}",
                M
            )));
        }

        Ok(MatrixView::<T, Row> {
            data: self.data.clone(),
            start_row: index,
            start_col: 0,
            dimension: Dimension { rows: 1, cols: N },
            _marker: PhantomData,
        })
    }

    fn col(&self, index: usize) -> MatrixResult<MatrixView<T, Col>> {
        if index >= N {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The column index cannot be greater than or equal to {}",
                N
            )));
        }

        Ok(MatrixView::<T, Col> {
            data: self.data.clone(),
            start_row: 0,
            start_col: index,
            dimension: Dimension { rows: M, cols: 1 },
            _marker: PhantomData,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_init() {
        let mat: Matrix<2, 2> = Matrix::new();
        dbg!(mat);
    }
}
