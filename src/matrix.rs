use std::{
    fmt,
    mem::ManuallyDrop,
    ops::{Index, IndexMut, Mul},
    ptr::NonNull,
    usize,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("InvalidIndex: {0}")]
    IndexOutOfBounds(String),

    #[error("InvalidShape: {0}")]
    InvalidShape(String),
}

pub type MatrixResult<T> = Result<T, MatrixError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dimension {
    pub num_rows: usize,
    pub num_cols: usize,
}

impl Dimension {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self { num_rows, num_cols }
    }
}

/// An immutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixView` will create and return new matricies. If
/// in-place operations are required, then they should be done on a
/// `MatrixViewMut` instead.
#[derive(Clone, Copy)]
pub struct MatrixView<'a, T> {
    data: &'a NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<'a, T> Index<(usize, usize)> for MatrixView<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let num_rows = self.dimension.num_rows;
        let num_cols = self.dimension.num_cols;
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

impl<'a> Mul<f32> for &MatrixView<'a, f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut mat: Matrix<f32> = Matrix::new(self.dimension.num_rows, self.dimension.num_cols);
        let mut view = mat.view_mut(0, 0, self.dimension).unwrap();
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                view[(i, j)] = self[(i, j)] * rhs;
            }
        }

        mat
    }
}

impl<'a> Mul<&MatrixView<'a, f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: &MatrixView<'a, f32>) -> Self::Output {
        let mut mat: Matrix<f32> = Matrix::new(rhs.dimension.num_rows, rhs.dimension.num_cols);
        let mut view = mat.view_mut(0, 0, rhs.dimension).unwrap();
        for i in 0..rhs.dimension.num_rows {
            for j in 0..rhs.dimension.num_cols {
                view[(i, j)] = rhs[(i, j)] * self;
            }
        }

        mat
    }
}

/// A mutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixViewMut` are done in-place; use `MatrixView` if
/// the original matrix should remain unchanged.
pub struct MatrixViewMut<'a, T> {
    data: &'a mut NonNull<T>,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<'a, T> Index<(usize, usize)> for MatrixViewMut<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let num_rows = self.dimension.num_rows;
        let num_cols = self.dimension.num_cols;
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
        let num_rows = self.dimension.num_rows;
        let num_cols = self.dimension.num_cols;
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
pub struct Matrix<T = f32> {
    data: NonNull<T>,
    size: Dimension,
}

impl Matrix<f32> {
    pub fn linspace<const N: usize>(start: f32, end: f32) -> Self {
        let data = {
            let mut values = Vec::with_capacity(N);
            let step = (end - start + 1.0) / N as f32;
            values.push(start);
            for i in 1..N {
                values.push(start + (i as f32) * step);
            }

            NonNull::new(ManuallyDrop::new(values).as_mut_ptr()).unwrap()
        };

        Self {
            data,
            size: Dimension {
                num_rows: 1,
                num_cols: N,
            },
        }
    }
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        let data = {
            let mut elems = ManuallyDrop::new(vec![T::default(); num_rows * num_cols]);
            NonNull::new(elems.as_mut_ptr()).unwrap()
        };

        Self {
            data,
            size: Dimension { num_rows, num_cols },
        }
    }
}

impl<T> Matrix<T> {
    pub fn from_vec<const M: usize, const N: usize>(v: Vec<T>) -> MatrixResult<Self> {
        if v.len() != M * N {
            return Err(MatrixError::InvalidShape(format!(
                "The length of vector must be {} (`M x N`)",
                M * N
            )));
        }

        let data = {
            let mut elems = ManuallyDrop::new(v);
            NonNull::new(elems.as_mut_ptr()).unwrap()
        };

        Ok(Self {
            data,
            size: Dimension {
                num_rows: M,
                num_cols: N,
            },
        })
    }

    pub const fn size(&self) -> Dimension {
        self.size
    }

    pub const fn is_square(&self) -> bool {
        self.size.num_rows == self.size.num_cols
    }

    pub fn row<'a>(&'a self, index: usize) -> MatrixResult<MatrixView<'a, T>> {
        if index >= self.size.num_rows {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The row index cannot be greater than or equal to {}",
                self.size.num_rows
            )));
        }

        Ok(MatrixView {
            data: &self.data,
            start_row: index,
            start_col: 0,
            dimension: Dimension {
                num_rows: 1,
                num_cols: self.size.num_cols,
            },
        })
    }

    pub fn col<'a>(&'a self, index: usize) -> MatrixResult<MatrixView<'a, T>> {
        if index >= self.size.num_cols {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The column index cannot be greater than or equal to {}",
                self.size.num_cols
            )));
        }

        Ok(MatrixView {
            data: &self.data,
            start_row: 0,
            start_col: index,
            dimension: Dimension {
                num_rows: self.size.num_rows,
                num_cols: 1,
            },
        })
    }

    pub fn view<'a>(
        &'a self,
        start_row: usize,
        start_col: usize,
        dimension: Dimension,
    ) -> MatrixResult<MatrixView<'a, T>> {
        if start_row >= self.size.num_rows {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start row must be less than {}",
                self.size.num_rows
            )));
        } else if start_col >= self.size.num_cols {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start col must be less than {}",
                self.size.num_cols
            )));
        } else if dimension.num_rows > self.size.num_rows {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more rows than `self".into(),
            ));
        } else if dimension.num_cols > self.size.num_cols {
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

    pub fn view_mut<'a>(
        &'a mut self,
        start_row: usize,
        start_col: usize,
        dimension: Dimension,
    ) -> MatrixResult<MatrixViewMut<'a, T>> {
        if start_row >= self.size.num_rows {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start row must be less than {}",
                self.size.num_rows
            )));
        } else if start_col >= self.size.num_cols {
            return Err(MatrixError::IndexOutOfBounds(format!(
                "The start col must be less than {}",
                self.size.num_cols
            )));
        } else if dimension.num_rows > self.size.num_rows {
            return Err(MatrixError::InvalidShape(
                "The view cannot have more rows than `self".into(),
            ));
        } else if dimension.num_cols > self.size.num_cols {
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

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        let m = self.size.num_rows;
        let n = self.size.num_cols;
        unsafe {
            Vec::from_raw_parts(self.data.as_mut(), m * n, m * n);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.size.num_rows;
        let n = self.size.num_cols;
        write!(f, "{}", format_args!("Matrix ({} x {}) [\n", m, n))?;
        for row in 0..m {
            for col in 0..n {
                let val = &self[(row, col)];
                write!(f, "{}", format_args!(" {:?} ", val))?;
            }
            write!(f, "\n")?;
        }
        write!(f, "]\n")
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let m = self.size.num_rows;
        let n = self.size.num_cols;
        if index.0 >= m {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The row index must be less than {}", m))
                    .to_string()
            );
        } else if index.1 >= n {
            panic!(
                "{}",
                MatrixError::IndexOutOfBounds(format!("The column index must be less than {}", n))
                    .to_string()
            );
        }

        unsafe {
            self.data
                .as_ptr()
                .add(n * index.0 + index.1)
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
        let mat: Matrix<f32> = Matrix::new(2, 2);
        dbg!(&mat);
        assert_eq!(mat.size(), Dimension::new(2, 2));

        let mat2 = Matrix::linspace::<5>(1.0, 5.0);
        dbg!(mat2);

        Ok(())
    }
}
