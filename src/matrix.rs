use core::{fmt, slice};
use std::{
    mem,
    ops::{Index, IndexMut},
    ptr::NonNull,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum MatrixError {
    #[error("The length of the vector must be {0}, but was {1}")]
    InvalidShape(usize, usize),
    #[error("The index {0} is out of bounds")]
    RowIndexOutOfBounds(usize),
}

type MatrixResult<T> = Result<T, MatrixError>;

#[derive(Clone)]
struct Row<'a, const N: usize>([&'a f32; N]);

impl<'a, const N: usize> fmt::Debug for Row<'a, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for val in self.0 {
            write!(f, " {} ", val)?;
        }
        write!(f, "]")
    }
}

impl<'a, const N: usize> Index<usize> for Row<'a, N> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= N {
            panic!(
                "RowIndexOutOfBounds: The index {:?} is out of bounds; it must 
                be less than {}.",
                index, N,
            );
        }

        self.0[index]
    }
}

impl<'a, const N: usize> IndexMut<usize> for Row<'a, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= N {
            panic!(
                "RowIndexOutOfBounds: The index {:?} is out of bounds; it must 
                be less than {}.",
                index, N,
            );
        }
        let val = self.0[index] as *const f32;
        unsafe { (val as *mut f32).as_mut().unwrap() }
    }
}

#[derive(Clone)]
struct Col<'a, const M: usize>([&'a f32; M]);

impl<'a, const M: usize> fmt::Debug for Col<'a, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[\n")?;
        for (i, val) in self.0.iter().enumerate() {
            if i != self.0.len() {
                write!(f, "\t{}\n ", val)?;
            }
        }
        write!(f, "]")
    }
}

impl<'a, const M: usize> Index<usize> for Col<'a, M> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= M {
            panic!(
                "RowIndexOutOfBounds: The index {:?} is out of bounds; it must 
                be less than {}.",
                index, M,
            );
        }
        self.0[index]
    }
}

impl<'a, const M: usize> IndexMut<usize> for Col<'a, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= M {
            panic!(
                "RowIndexOutOfBounds: The index {:?} is out of bounds; it must 
                be less than {}.",
                index, M,
            );
        }
        let val = self.0[index] as *const f32;
        unsafe { (val as *mut f32).as_mut().unwrap() }
    }
}

// TODO: Impl iterator trait
struct Matrix<const M: usize, const N: usize> {
    data: NonNull<f32>,
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    /// Create a `M x N` matrix from the given vector.
    ///
    /// ## Panics
    /// Panics if the length of the vector is not `M * N` or if `M` or `N` are zero.
    fn from_vec(v: Vec<f32>) -> Self {
        if v.len() != M * N {
            panic!(
                "InvalidShape: The length of the vector must be {}, but was {}",
                M * N,
                v.len()
            );
        } else if (M == 0) || (N == 0) {
            panic!("InvalidShape: Cannot create a matrix with `0` rows or columns");
        }

        let data = {
            let mut v = mem::ManuallyDrop::new(v);
            let data = v.as_mut_ptr();
            NonNull::new(data).unwrap()
        };

        Self { data }
    }

    // TODO: Add `from_slice` and `from_array` functions

    // TODO: Add `from_array_slice` functions (matrix from slice of array i.e `&[[1.0, 2.0],[3.0, 4.0]]`)

    // TODO: Add `default` function (create an empty matrix)

    // TODO: Add `identity`, `ones`, `zeros`, and `linspace` functions for initalizations

    // TODO: Generalize to all numerical types (not just f32)

    /// Converts the matrix to a vector, consuming `self`.
    ///
    /// The vector elements will be in the order `N * row + col`.
    ///
    /// ## Safety
    /// This uses `Vec::from_raw_parts`, and must satisfy all of its safety requirements.
    fn to_vec(self) -> Vec<f32> {
        unsafe { Vec::from_raw_parts(self.data.as_ptr(), M * N, M * N) }
    }

    /// Returns the matrix as an immutable slice of length `M * N`.
    ///
    /// The slice elements will be in the order `N * row + col`.
    ///
    /// ## Safety
    /// This uses `slice::from_raw_parts`, and must satisfy all of its safety requirements.
    fn as_slice<'a>(&self) -> &'a [f32] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), M * N) }
    }

    /// Returns the matrix as a mutable slice of length `M * N`.
    ///
    /// The slice elements will be in the order `N * row + col`.
    ///
    /// ## Safety
    /// This uses `slice::from_raw_parts`, and must satisfy all of its safety requirements.
    fn as_slice_mut<'a>(&mut self) -> &'a mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), M * N) }
    }

    /// Gets the index in the slice `data` of the matrix, for the specified row and column
    /// indicies.
    const fn get_index(&self, row: usize, col: usize) -> usize {
        N * row + col
    }

    /// Returns an immutable reference to the element in the matrix at `[row, col]`.
    ///
    /// ## Panics
    /// * Panics if `row` or `col` are greater than or equal `M` and `N` respectively.
    fn index(&self, row: usize, col: usize) -> &f32 {
        if (row >= M) || (col >= N) {
            panic!(
                "IndexOutOfBounds: The index {:?} is out of bounds. The row and
                column indicies must be less than {} and {} respectively.",
                (row, col),
                M,
                N
            );
        }

        let data = self.data.as_ptr();

        unsafe { data.add(self.get_index(row, col)).as_ref().unwrap() }
    }

    /// Returns a mutable reference to the element in the matrix at `[row, col]`.
    ///
    /// ## Panics
    /// Panics if `row` or `col` are greater than `M` and `N` respectively.
    fn index_mut(&self, row: usize, col: usize) -> &mut f32 {
        if (row >= M) || (col >= N) {
            panic!(
                "IndexOutOfBounds: The index {:?} is out of bounds. The row and
                column indicies must be less than {} and {} respectively.",
                (row, col),
                M,
                N
            );
        }

        let data = self.data.as_ptr();

        unsafe { data.add(self.get_index(row, col)).as_mut().unwrap() }
    }

    /// Returns a `Row<N>` containing references to the elements in the specified row of the matrix.
    ///
    /// ## Panics
    /// Panics if the index is greater than or equal to `M`.
    fn row<'a>(&self, idx: usize) -> Row<'a, N> {
        if idx >= M {
            panic!(
                "RowIndexOutOfBounds: The index {} is out of bounds. It must be
                less than {}",
                idx, M
            );
        }
        let mut row = [&0.0; N];

        let data = self.data.as_ptr();
        for i in 0..N {
            row[i] = unsafe { data.add(self.get_index(idx, i)).as_ref().unwrap() };
        }

        Row(row)
    }

    /// Returns a `Col<M>` containing references to the elements in the specified row of the matrix.
    ///
    /// ## Panics
    /// Panics if the index is greater than or equal to `N`.
    fn col<'a>(&self, idx: usize) -> Col<'a, M> {
        if idx >= N {
            panic!(
                "ColumnIndexOutOfBounds: The index {} is out of bounds. It must
                be less than {}",
                idx, N
            );
        }
        let mut col = [&0.0; M];

        let data = self.data.as_ptr();
        for i in 0..M {
            col[i] = unsafe { data.add(self.get_index(i, idx)).as_ref().unwrap() };
        }

        Col(col)
    }

    // TODO: Add `reshape` function (convert to any combination of M*N, consuming `self`)

    /// Returns a new matrix whose elements are the elements of `self` multiplied by the given
    /// `scalar`.
    fn scale(&self, scalar: f32) -> Self {
        let data = unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), M * N) };

        let mut scaled = Vec::with_capacity(M * N);
        for val in data {
            scaled.push(*val * scalar);
        }

        Matrix::from_vec(scaled)
    }

    /// Multiplies all elements of `self` by the given `scalar` inplace.
    fn scale_inplace(&mut self, scalar: f32) {
        let data = unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), M * N) };

        for val in data {
            *val *= scalar;
        }
    }

    // NOTE: Use more efficient algorithim
    //  - Specialize for square matricies, etc
    //
    /// Returns a new matrix containing the result of the matrix multiplaction of `self` and
    /// `other`.
    fn mul<const P: usize>(&self, other: &Matrix<N, P>) -> Matrix<M, P> {
        let res = Matrix::from_vec(vec![0.0; M * P]);

        for i in 0..M {
            for j in 0..P {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += self.index(i, k) * other.index(k, j);
                }
                *res.index_mut(i, j) = sum;
            }
        }

        res
    }
}

impl<const M: usize, const N: usize> Drop for Matrix<M, N> {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.data.as_ptr(), M * N, M * N);
        }
    }
}

impl<const M: usize, const N: usize> Clone for Matrix<M, N> {
    fn clone(&self) -> Self {
        let mut data = vec![0.0; M * N];
        for row in 0..M {
            for col in 0..N {
                data[self.get_index(row, col)] = *self.index(row, col);
            }
        }

        Self::from_vec(data)
    }
}

impl<const M: usize, const N: usize> fmt::Debug for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write!(f, "Matrix<{}> [\n", format_args!("{}, {}", M, N))?;
        // write!(f, "{}", format_args!("Matrix<{}, {}> [\n", M, N))?;
        f.write_fmt(format_args!("Matrix<{}, {}> [\n", M, N))?;
        for i in 0..M {
            f.write_fmt(format_args!("\t{:?}\n", self.row(i)))?;
            // write!(f, "{}", format_args!("\t{:?}\n", self.row(i)))?;
        }
        write!(f, "{}", format_args!("]"))
        // f.debug_list()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_init() {
        let mat = Matrix::<3, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        dbg!(&mat);
    }

    #[test]
    fn can_index() {
        const M: usize = 2;
        const N: usize = 3;
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = Matrix::<M, N>::from_vec(vec.clone());

        for row in 0..M {
            for col in 0..N {
                assert_eq!(*mat.index(row, col), vec[mat.get_index(row, col)])
            }
        }

        let row0 = mat.row(0);
        assert_eq!(row0[0], vec[0]);

        let col0 = mat.col(0);
        assert_eq!(col0[1], vec[3]);
    }

    #[test]
    fn can_scale() {
        const M: usize = 2;
        const N: usize = 3;
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut mat = Matrix::<M, N>::from_vec(vec.clone());
        assert_eq!(mat.as_slice(), vec);

        let scaled = mat.scale(2.0);
        assert_eq!(
            scaled.as_slice(),
            vec.iter().map(|x| x * 2.0).collect::<Vec<_>>()
        );

        mat.scale_inplace(2.0);
        assert_eq!(mat.as_slice(), scaled.as_slice());
    }

    #[test]
    fn can_mul() {
        let a = Matrix::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::<3, 2>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let c = a.mul(&b);

        // FIXME: Add asserts!
        dbg!(a);
        dbg!(b);
        dbg!(c);

        let a = Matrix::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::<2, 2>::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

        let c = a.mul(&b);
        dbg!(a);
        dbg!(b);
        dbg!(c);
    }
}
