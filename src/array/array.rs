use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Array<T> {
    pub data: Vec<T>,
    pub shape: (usize, usize),
    pub strides: (usize, usize),
}

impl<T: Clone + Default> Array<T> {
    pub fn from_vec(data: Vec<T>, shape: (usize, usize)) -> Self {
        assert_eq!(data.len(), shape.0 * shape.1);
        Array {
            data,
            shape,
            strides: (shape.1, 1),
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        Array {
            data: vec![T::default(); shape.0 * shape.1],
            shape,
            strides: (shape.1, 1),
        }
    }

    pub fn ones(shape: (usize, usize)) -> Array<f64>
    where
        T: Into<f64>,
    {
        Array {
            data: vec![1.0; shape.0 * shape.1],
            shape,
            strides: (shape.1, 1),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<(usize, usize)> for Array<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i * self.strides.0 + j * self.strides.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Array<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i * self.strides.0 + j * self.strides.1]
    }
}