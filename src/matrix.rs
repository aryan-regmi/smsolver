use core::fmt;
use std::{
    fmt::Write,
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
        f.debug_list().entries(self.0.iter()).finish()
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
        write!(f, "[")?;
        for val in self.0 {
            write!(f, " {} ", val)?;
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

struct Matrix<const M: usize, const N: usize> {
    // Index = N*row + (M-1)*col
    data: NonNull<f32>,
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    fn from_vec(v: Vec<f32>) -> Self {
        if v.len() != M * N {
            panic!(
                "InvalidShape: The length of the vector must be {}, but was {}",
                M * N,
                v.len()
            );
        }

        let data = {
            let mut v = mem::ManuallyDrop::new(v);
            let data = v.as_mut_ptr();
            NonNull::new(data).unwrap()
        };

        Self { data }
    }

    const fn get_index(&self, row: usize, col: usize) -> usize {
        N * row + col
    }

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

    fn col(&self, idx: usize) -> Col<M> {
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
}

impl<const M: usize, const N: usize> Drop for Matrix<M, N> {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.data.as_ptr(), M * N, M * N);
        }
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
}
