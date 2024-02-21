use std::{
    alloc::{self, Layout},
    fmt,
    mem::ManuallyDrop,
    ops::{Add, AddAssign, Index, IndexMut},
    ptr::NonNull,
};

/// Possible errors returned by matrix methods.
#[derive(Debug, thiserror::Error)]
pub enum MatrixError {
    #[error("IndexOutOfBounds: The row index was {0}, but it must be less than {1} ")]
    RowIndexOutOfBounds(usize, usize),

    #[error("IndexOutOfBounds: The column index was {0}, but it must be less than {1} ")]
    ColIndexOutOfBounds(usize, usize),

    #[error("InvalidShape: {0}")]
    InvalidShape(String),

    #[error("AllocationFailed: Unable to allocate memory for the matrix")]
    AllocationFailed,
}

/// Result type returned by matrix methods
pub type MatrixResult<T> = Result<T, MatrixError>;

/// Represents the dimensions of a matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dimensions(usize, usize);

impl Dimensions {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self(rows, cols)
    }

    pub const fn rows(&self) -> usize {
        self.0
    }

    pub const fn cols(&self) -> usize {
        self.1
    }
}

impl From<(usize, usize)> for Dimensions {
    fn from(value: (usize, usize)) -> Self {
        Self(value.0, value.1)
    }
}

pub struct Matrix<T> {
    buffer: NonNull<T>,
    size: Dimensions,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let m = self.size.rows();
        let n = self.size.cols();
        if index.0 >= m {
            panic!("{}", MatrixError::RowIndexOutOfBounds(index.0, m));
        }
        if index.1 >= n {
            panic!("{}", MatrixError::ColIndexOutOfBounds(index.1, n));
        }

        unsafe {
            self.buffer
                .as_ptr()
                .add(n * index.0 + index.1)
                .as_ref()
                .expect("Invalid matrix element")
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let m = self.size.rows();
        let n = self.size.cols();
        if index.0 >= m {
            panic!("{}", MatrixError::RowIndexOutOfBounds(index.0, m));
        }
        if index.1 >= n {
            panic!("{}", MatrixError::ColIndexOutOfBounds(index.1, n));
        }

        unsafe {
            self.buffer
                .as_ptr()
                .add(n * index.0 + index.1)
                .as_mut()
                .expect("Invalid matrix element")
        }
    }
}

impl<T> Matrix<T> {
    pub fn from_vec(v: Vec<T>, size: Dimensions) -> MatrixResult<Self> {
        // Input validation
        {
            if (size.rows() == 0) || (size.cols() == 0) {
                return Err(MatrixError::InvalidShape(
                    "The number of rows and columns must be greater than 0".into(),
                ));
            }
            if v.len() != size.rows() * size.cols() {
                return Err(MatrixError::InvalidShape(format!(
                    "The length of the vector must be {}",
                    size.rows() * size.cols()
                )));
            }
        }

        let data = {
            let mut elems = ManuallyDrop::new(v);
            NonNull::new(elems.as_mut_ptr()).expect("The vector was invalid")
        };

        Ok(Self { buffer: data, size })
    }

    pub const fn size(&self) -> Dimensions {
        self.size
    }

    pub const fn is_square(&self) -> bool {
        self.size.rows() == self.size.cols()
    }
}

impl Matrix<f64> {
    pub fn linspace<const N: usize>(start: f64, end: f64) -> Self {
        let data = {
            let mut values = Vec::with_capacity(N);
            let step = (end - start + 1.0) / N as f64;
            values.push(start);
            for i in 1..N {
                values.push(start + (i as f64) * step);
            }

            NonNull::new(ManuallyDrop::new(values).as_mut_ptr()).unwrap()
        };

        Self {
            buffer: data,
            size: (1, N).into(),
        }
    }
}

impl<T: Default + Clone> Matrix<T> {
    /// Creates a new matrix with the specified number of rows and columns.
    ///
    /// The created matrix will **not** have the elements zeroed, and may have random values.
    pub fn new(size: Dimensions) -> MatrixResult<Self> {
        Self::from_vec(vec![T::default(); size.rows() * size.cols()], size)
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        let m = self.size.rows();
        let n = self.size.cols();
        unsafe {
            let layout = Layout::from_size_align(m * n, std::mem::align_of::<T>())
                .expect(&MatrixError::AllocationFailed.to_string());
            alloc::dealloc(self.buffer.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: Clone + Default> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        let m = self.size().rows();
        let n = self.size().cols();
        let mut mat = Matrix::from_vec(vec![T::default(); m * n], self.size())
            .expect("Unable to clone matrix");

        for i in 0..self.size().rows() {
            for j in 0..self.size().cols() {
                mat[(i, j)] = self[(i, j)].clone();
            }
        }

        mat
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.size.rows();
        let n = self.size.cols();
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

impl<T: AddAssign + Clone> Add<T> for Matrix<T> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        for i in 0..self.size().rows() {
            for j in 0..self.size().cols() {
                self[(i, j)] += rhs.clone();
            }
        }
        self
    }
}
impl Add<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn add(self, mut rhs: Matrix<f32>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn add(self, mut rhs: Matrix<f64>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<u8>> for u8 {
    type Output = Matrix<u8>;

    fn add(self, mut rhs: Matrix<u8>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<u16>> for u16 {
    type Output = Matrix<u16>;

    fn add(self, mut rhs: Matrix<u16>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<u32>> for u32 {
    type Output = Matrix<u32>;

    fn add(self, mut rhs: Matrix<u32>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<u64>> for u64 {
    type Output = Matrix<u64>;

    fn add(self, mut rhs: Matrix<u64>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<usize>> for usize {
    type Output = Matrix<usize>;

    fn add(self, mut rhs: Matrix<usize>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<i8>> for i8 {
    type Output = Matrix<i8>;

    fn add(self, mut rhs: Matrix<i8>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<i16>> for i16 {
    type Output = Matrix<i16>;

    fn add(self, mut rhs: Matrix<i16>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    fn add(self, mut rhs: Matrix<i32>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<i64>> for i64 {
    type Output = Matrix<i64>;

    fn add(self, mut rhs: Matrix<i64>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}
impl Add<Matrix<isize>> for isize {
    type Output = Matrix<isize>;

    fn add(self, mut rhs: Matrix<isize>) -> Self::Output {
        for i in 0..rhs.size().rows() {
            for j in 0..rhs.size().cols() {
                rhs[(i, j)] += self;
            }
        }
        rhs
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn can_init() -> MatrixResult<()> {
        let mat: Matrix<f32> = Matrix::new((2, 2).into())?;
        assert_eq!(mat.size(), Dimensions::new(2, 2));
        // dbg!(&mat);

        let mat = Matrix::linspace::<5>(1.0, 5.0);
        assert_eq!(mat.size(), Dimensions::new(1, 5));
        let v = vec![1., 2., 3., 4., 5.];
        for j in 0..5 {
            assert_eq!(mat[(0, j)], v[j]);
        }
        // dbg!(&mat);

        let v = vec![1, 2, 3, 4, 5, 6];
        let mat = Matrix::from_vec(v.clone(), (2, 3).into())?;
        assert_eq!(mat.size(), Dimensions::new(2, 3));
        for i in 0..mat.size().rows() {
            for j in 0..mat.size().cols() {
                assert_eq!(mat[(i, j)], v[mat.size().cols() * i + j]);
            }
        }
        // dbg!(&mat);

        Ok(())
    }

    #[test]
    fn can_add() -> MatrixResult<()> {
        let mat = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2).into())?;
        {
            let new_mat: Matrix<f64> = 1.0 + mat.clone() + 3.0;
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(new_mat[(i, j)], 4.0 + mat[(i, j)]);
                }
            }
            // dbg!(&new_mat);
            // dbg!(&mat);
        }

        Ok(())
    }
}
