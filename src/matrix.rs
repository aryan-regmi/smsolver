use std::{
    alloc::{self, Layout},
    fmt,
    mem::ManuallyDrop,
    ops::{Div, Index, IndexMut, Mul},
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

    #[error("AllocationFailed: Unable to allocate memory for the matrix")]
    AllocationFailed,
}

pub type MatrixResult<T> = Result<T, MatrixError>;

pub trait MatrixSize {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dimension {
    num_rows: usize,
    num_cols: usize,
}

impl Dimension {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self { num_rows, num_cols }
    }
}

impl MatrixSize for Dimension {
    #[inline]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    fn num_cols(&self) -> usize {
        self.num_cols
    }
}

impl From<(usize, usize)> for Dimension {
    fn from(value: (usize, usize)) -> Self {
        Self {
            num_rows: value.0,
            num_cols: value.1,
        }
    }
}

impl MatrixSize for (usize, usize) {
    #[inline]
    fn num_rows(&self) -> usize {
        self.0
    }

    #[inline]
    fn num_cols(&self) -> usize {
        self.1
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

impl<'a> Mul<f32> for MatrixView<'a, f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut mat =
            Matrix::<f32>::new((self.dimension.num_rows, self.dimension.num_cols)).unwrap();
        let mut view = mat.view_mut(0, 0, self.dimension).unwrap();
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                view[(i, j)] = self[(i, j)] * rhs;
            }
        }

        mat
    }
}

impl<'a> Mul<MatrixView<'a, f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: MatrixView<'a, f32>) -> Self::Output {
        let mut mat = Matrix::<f32>::new((rhs.dimension.num_rows, rhs.dimension.num_cols)).unwrap();
        let mut view = mat.view_mut(0, 0, rhs.dimension).unwrap();
        for i in 0..rhs.dimension.num_rows {
            for j in 0..rhs.dimension.num_cols {
                view[(i, j)] = rhs[(i, j)] * self;
            }
        }

        mat
    }
}

impl<'a> Mul<f32> for &MatrixView<'a, f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut mat =
            Matrix::<f32>::new((self.dimension.num_rows, self.dimension.num_cols)).unwrap();
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
        let mut mat = Matrix::<f32>::new((rhs.dimension.num_rows, rhs.dimension.num_cols)).unwrap();
        let mut view = mat.view_mut(0, 0, rhs.dimension).unwrap();
        for i in 0..rhs.dimension.num_rows {
            for j in 0..rhs.dimension.num_cols {
                view[(i, j)] = rhs[(i, j)] * self;
            }
        }

        mat
    }
}

impl<'a> Div<f32> for MatrixView<'a, f32> {
    type Output = Matrix<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut mat =
            Matrix::<f32>::new((self.dimension.num_rows, self.dimension.num_cols)).unwrap();
        let mut view = mat.view_mut(0, 0, self.dimension).unwrap();
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                view[(i, j)] = self[(i, j)] / rhs;
            }
        }

        mat
    }
}

impl<'a> Div<f32> for &MatrixView<'a, f32> {
    type Output = Matrix<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut mat =
            Matrix::<f32>::new((self.dimension.num_rows, self.dimension.num_cols)).unwrap();
        let mut view = mat.view_mut(0, 0, self.dimension).unwrap();
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                view[(i, j)] = self[(i, j)] / rhs;
            }
        }

        mat
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for MatrixView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.dimension.num_rows;
        let n = self.dimension.num_cols;
        write!(f, "{}", format_args!("MatrixView ({} x {}) [\n", m, n))?;
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

/// A mutable view into a matrix.
///
/// ## Note
/// All operations on a `MatrixViewMut` are done in-place; use `MatrixView` if
/// the original matrix should remain unchanged.
pub struct MatrixViewMut<'a, T> {
    data: &'a mut T,
    start_row: usize,
    start_col: usize,
    dimension: Dimension,
}

impl<'a, T> Clone for MatrixViewMut<'a, T> {
    fn clone(&self) -> Self {
        let data = unsafe { (self.data as *const T as *mut T).as_mut().unwrap() };
        Self {
            data,
            start_row: self.start_row.clone(),
            start_col: self.start_col.clone(),
            dimension: self.dimension.clone(),
        }
    }
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
            (self.data as *const T as *mut T)
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
            (self.data as *mut T)
                .add(num_cols * row + col)
                .as_mut()
                .unwrap()
        }
    }
}

impl<'a> Mul<f32> for MatrixViewMut<'a, f32> {
    type Output = MatrixViewMut<'a, f32>;

    fn mul(mut self, rhs: f32) -> Self::Output {
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                self[(i, j)] *= rhs;
            }
        }

        self
    }
}

impl<'a> Mul<MatrixViewMut<'a, f32>> for f32 {
    type Output = MatrixViewMut<'a, f32>;

    fn mul(self, mut rhs: MatrixViewMut<'a, f32>) -> Self::Output {
        for i in 0..rhs.dimension.num_rows {
            for j in 0..rhs.dimension.num_cols {
                rhs[(i, j)] *= self;
            }
        }

        rhs
    }
}

impl<'a> Mul<f32> for &mut MatrixViewMut<'a, f32> {
    type Output = MatrixViewMut<'a, f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                self[(i, j)] *= rhs;
            }
        }

        self.clone()
    }
}

impl<'a> Mul<&mut MatrixViewMut<'a, f32>> for f32 {
    type Output = MatrixViewMut<'a, f32>;

    fn mul(self, rhs: &mut MatrixViewMut<'a, f32>) -> Self::Output {
        for i in 0..rhs.dimension.num_rows {
            for j in 0..rhs.dimension.num_cols {
                rhs[(i, j)] *= self;
            }
        }

        rhs.clone()
    }
}

impl<'a> Div<f32> for MatrixViewMut<'a, f32> {
    type Output = MatrixViewMut<'a, f32>;

    fn div(mut self, rhs: f32) -> Self::Output {
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                self[(i, j)] /= rhs;
            }
        }

        self
    }
}

impl<'a> Div<f32> for &mut MatrixViewMut<'a, f32> {
    type Output = MatrixViewMut<'a, f32>;

    fn div(self, rhs: f32) -> Self::Output {
        for i in 0..self.dimension.num_rows {
            for j in 0..self.dimension.num_cols {
                self[(i, j)] /= rhs;
            }
        }

        self.clone()
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for MatrixViewMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.dimension.num_rows;
        let n = self.dimension.num_cols;
        write!(f, "{}", format_args!("MatrixView ({} x {}) [\n", m, n))?;
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

impl<T> Matrix<T> {
    pub fn new<S: MatrixSize>(size: S) -> MatrixResult<Self> {
        if (size.num_cols() == 0) || (size.num_cols() == 0) {
            return Err(MatrixError::InvalidShape(
                "The number of rows and columns should be larger than zero".to_string(),
            ));
        }

        let data = unsafe {
            let layout = Layout::from_size_align(
                size.num_rows() * size.num_cols(),
                std::mem::align_of::<T>(),
            )
            .expect(&MatrixError::AllocationFailed.to_string());
            let alloced = alloc::alloc(layout) as *mut T;
            NonNull::new(alloced).ok_or(MatrixError::AllocationFailed)?
        };

        Ok(Self {
            data,
            size: Dimension::new(size.num_rows(), size.num_cols()),
        })
    }

    pub fn from_vec(v: Vec<T>, size: Dimension) -> MatrixResult<Self> {
        let num_rows = size.num_rows;
        let num_cols = size.num_cols;
        if v.len() != num_rows * num_cols {
            return Err(MatrixError::InvalidShape(format!(
                "The length of vector must be {} (`M x N`)",
                num_rows * num_cols
            )));
        }

        let data = {
            let mut elems = ManuallyDrop::new(v);
            NonNull::new(elems.as_mut_ptr()).unwrap()
        };

        Ok(Self {
            data,
            size: Dimension { num_rows, num_cols },
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
            data: unsafe { self.data.as_mut() },
            start_row,
            start_col,
            dimension,
        })
    }

    pub fn view_self<'a>(&'a self) -> MatrixView<'a, T> {
        MatrixView {
            data: &self.data,
            start_row: 0,
            start_col: 0,
            dimension: self.size,
        }
    }

    pub fn view_self_mut<'a>(&'a mut self) -> MatrixViewMut<'a, T> {
        MatrixViewMut {
            data: unsafe { self.data.as_mut() },
            start_row: 0,
            start_col: 0,
            dimension: self.size,
        }
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        let m = self.size.num_rows;
        let n = self.size.num_cols;
        unsafe {
            let layout = Layout::from_size_align(m * n, std::mem::align_of::<T>())
                .expect(&MatrixError::AllocationFailed.to_string());
            alloc::dealloc(self.data.as_ptr() as *mut u8, layout);
        }
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

impl Clone for Matrix<f32> {
    fn clone(&self) -> Self {
        self.view_self() * 1.0
    }
}

impl<'a> Mul<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.view(0, 0, self.size).unwrap() * rhs
    }
}

impl<'a> Mul<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: &Matrix<f32>) -> Self::Output {
        self * rhs.view(0, 0, rhs.size).unwrap()
    }
}

impl<'a> Mul<f32> for Matrix<f32> {
    type Output = Matrix<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.view(0, 0, self.size).unwrap() * rhs
    }
}

impl<'a> Mul<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, rhs: Matrix<f32>) -> Self::Output {
        self * rhs.view(0, 0, rhs.size).unwrap()
    }
}

impl<'a> Div<f32> for &Matrix<f32> {
    type Output = Matrix<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        self.view(0, 0, self.size).unwrap() / rhs
    }
}

impl<'a> Div<f32> for Matrix<f32> {
    type Output = Matrix<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        self.view(0, 0, self.size).unwrap() / rhs
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_init() -> MatrixResult<()> {
        let mat: Matrix<f32> = Matrix::new((2, 2))?;
        assert_eq!(mat.size(), Dimension::new(2, 2));
        // dbg!(&mat);

        let mat2 = Matrix::linspace::<5>(1.0, 5.0);
        assert_eq!(mat2.size(), Dimension::new(1, 5));
        let v = vec![1., 2., 3., 4., 5.];
        for j in 0..5 {
            assert_eq!(mat2[(0, j)], v[j]);
        }
        // dbg!(&mat2);

        let mat3 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3).into())?;
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(mat3[(i, j)], v[3 * i + j]);
            }
        }
        // dbg!(mat3);

        Ok(())
    }

    #[test]
    fn can_mul() -> MatrixResult<()> {
        let mut mat = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2).into())?;
        {
            let new_mat = 3.0 * mat.clone() * 1.0;
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(new_mat[(i, j)], 3.0 * mat[(i, j)]);
                }
            }
        }

        // Inplace
        {
            let mat_view = 1.0 * mat.view_self_mut() * 3.0;

            let v = vec![3., 6., 9., 12.];
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(mat_view[(i, j)], v[i * 2 + j]);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn can_div() -> MatrixResult<()> {
        let mut mat = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2).into())?;
        {
            let new_mat = mat.clone() / 3.0;
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(new_mat[(i, j)], (1.0 / 3.0) * mat[(i, j)]);
                }
            }
        }

        // Inplace
        {
            let mat_view = mat.view_self_mut() / 3.0;

            let v = vec![(1. / 3.), (2. / 3.), 1.0, (4. / 3.)];
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(mat_view[(i, j)], v[i * 2 + j]);
                }
            }
        }

        Ok(())
    }
}
