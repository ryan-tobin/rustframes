use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Array<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T: Clone + Default> Array<T> {
    /// Create an array from a vector with given shape
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape");

        let strides = Self::compute_strides(&shape);
        Array {
            data,
            shape,
            strides,
        }
    }

    /// Create an array filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Array {
            data: vec![T::default(); total_size],
            shape,
            strides,
        }
    }

    /// Create an array filled with ones (for numeric types)
    pub fn ones(shape: Vec<usize>) -> Array<f64>
    where
        T: Into<f64>,
    {
        let total_size: usize = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Array {
            data: vec![1.0; total_size],
            shape,
            strides,
        }
    }

    /// Create array with given shape and fill value
    pub fn full(shape: Vec<usize>, fill_value: T) -> Self {
        let total_size: usize = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Array {
            data: vec![fill_value; total_size],
            shape,
            strides,
        }
    }

    /// Create array from range
    pub fn arange(start: T, stop: T, step: T) -> Array<T>
    where
        T: num_traits::Num + PartialOrd + Copy,
    {
        let mut data = Vec::new();
        let mut current = start;
        while current < stop {
            data.push(current);
            current = current + step;
        }
        let len = data.len();
        Array::from_vec(data, vec![len])
    }

    /// Compute strides for row-major (C-style) ordering
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Convert multi-dimensional index to flat index
    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index dimension mismatch");
        indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        for (idx, dim_size) in indices.iter().zip(&self.shape) {
            if *idx >= *dim_size {
                return None;
            }
        }
        let flat_index = self.ravel_index(indices);
        self.data.get(flat_index)
    }

    /// Get mutable element at mutli-dimensional index
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        for (idx, dim_size) in indices.iter().zip(&self.shape) {
            if *idx >= *dim_size {
                return None;
            }
        }
        let flat_index = self.ravel_index(indices);
        self.data.get_mut(flat_index)
    }

    /// Reshape array to new shape (must have same total size)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Array<T> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "Total size must remain the same");

        Array::from_vec(self.data.clone(), new_shape)
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get array size (total elements)
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Transpose 2D array
    pub fn transpose(&self) -> Array<T> {
        if self.ndim() != 2 {
            panic!("Transpose currently only supports 2D arrays");
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = Vec::with_capacity(self.data.len());

        for j in 0..cols {
            for i in 0..rows {
                let flat_idx = i * self.strides[0] + j * self.strides[i];
                new_data.push(self.data[flat_idx].clone());
            }
        }

        Array::from_vec(new_data, vec![cols, rows])
    }
}

// Implement indexing for backwards compability with 2D arrays
impl<T: Clone + Default> Index<(usize, usize)> for Array<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if self.ndim() != 2 {
            panic!("2D indexing only works for 2D arrays");
        }
        let (i, j) = index;
        &self.data[i * self.strides[0] + j * self.strides[1]]
    }
}

impl<T: Clone + Default> IndexMut<(usize, usize)> for Array<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if self.ndim() != 2 {
            panic!("2D indexing only works for 2D arrays");
        }
        let (i, j) = index;
        &mut self.data[i * self.strides[0] + j * self.strides[1]]
    }
}

// Implement indexing for N-dimensional arrays
impl<T: Clone + Default> Index<&[usize]> for Array<T> {
    type Output = T;
    fn index(&self, indices: &[usize]) -> &Self::Output {
        let flat_index = self.ravel_index(indices);
        &self.data[flat_index]
    }
}

impl<T: Clone + Default> IndexMut<&[usize]> for Array<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let flat_index = self.ravel_index(indices);
        &mut self.data[flat_index]
    }
}
