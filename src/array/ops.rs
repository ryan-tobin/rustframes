use super::Array;
use core::f64;
use std::ops::{Add, Div, Mul, Sub};

impl Array<f64> {
    /// Check if two shapes are broadcastable
    pub fn shapes_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
        let max_dims = shape1.len().max(shape2.len());
        for i in 0..max_dims {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).unwrap_or(&1);
            if *dim1 != *dim2 && *dim1 != 1 && *dim2 != 1 {
                return false;
            }
        }
        true
    }

    /// Compute the resulting shape after broadcasting
    pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        if !Self::shapes_broadcastable(shape1, shape2) {
            return None;
        }

        let max_dims = shape1.len().max(shape2.len());
        let mut result_shape = vec![1; max_dims];

        for i in 0..max_dims {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).unwrap_or(&1);
            result_shape[max_dims - i - 1] = (*dim1).max(*dim2);
        }

        Some(result_shape)
    }

    /// Get element with broadcasting
    pub fn get_broadcasted(&self, indices: &[usize], target_shape: &[usize]) -> &f64 {
        let mut actual_indices = vec![0; self.shape.len()];
        let shape_offset = target_shape.len() - self.shape.len();

        for (i, &target_idx) in indices.iter().enumerate() {
            if i >= shape_offset {
                let self_dim_idx = i - shape_offset;
                if self_dim_idx < self.shape.len() {
                    if self.shape[self_dim_idx] == 1 {
                        actual_indices[self_dim_idx] = 0; // Broadcast dimension
                    } else {
                        actual_indices[self_dim_idx] = target_idx;
                    }
                }
            }
        }

        &self[actual_indices.as_slice()]
    }

    /// Element-wise addition with broadcasting
    pub fn add_broadcast(&self, other: &Array<f64>) -> Option<Array<f64>> {
        let result_shape = Self::broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Array::zeros(result_shape.clone());

        let total_elements: usize = result_shape.iter().product();
        for flat_idx in 0..total_elements {
            let indices = Self::unravel_index(flat_idx, &result_shape);
            let val1 = self.get_broadcasted(&indices, &result_shape);
            let val2 = other.get_broadcasted(&indices, &result_shape);
            result[indices.as_slice()] = val1 + val2;
        }

        Some(result)
    }

    /// Element-wise subtraction with broadcasting
    pub fn sub_broadcast(&self, other: &Array<f64>) -> Option<Array<f64>> {
        let result_shape = Self::broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Array::zeros(result_shape.clone());

        let total_elements: usize = result_shape.iter().product();
        for flat_idx in 0..total_elements {
            let indices = Self::unravel_index(flat_idx, &result_shape);
            let val1 = self.get_broadcasted(&indices, &result_shape);
            let val2 = other.get_broadcasted(&indices, &result_shape);
            result[indices.as_slice()] = val1 - val2;
        }

        Some(result)
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul_broadcast(&self, other: &Array<f64>) -> Option<Array<f64>> {
        let result_shape = Self::broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Array::zeros(result_shape.clone());

        let total_elements: usize = result_shape.iter().product();
        for flat_idx in 0..total_elements {
            let indices = Self::unravel_index(flat_idx, &result_shape);
            let val1 = self.get_broadcasted(&indices, &result_shape);
            let val2 = other.get_broadcasted(&indices, &result_shape);
            result[indices.as_slice()] = val1 * val2;
        }

        Some(result)
    }

    /// Element-wise division with broadcasting
    pub fn div_broadcast(&self, other: &Array<f64>) -> Option<Array<f64>> {
        let result_shape = Self::broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Array::zeros(result_shape.clone());

        let total_elements: usize = result_shape.iter().product();
        for flat_idx in 0..total_elements {
            let indices = Self::unravel_index(flat_idx, &result_shape);
            let val1 = self.get_broadcasted(&indices, &result_shape);
            let val2 = other.get_broadcasted(&indices, &result_shape);
            result[indices.as_slice()] = val1 / val2;
        }

        Some(result)
    }

    /// Convert flat index to multi-dimensional index
    pub fn unravel_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remaining = flat_index;

        for (i, &_dim_size) in shape.iter().enumerate() {
            let stride: usize = shape[i + 1..].iter().product();
            indices[i] = remaining / stride;
            remaining %= stride;
        }

        indices
    }

    /// Scalar operations
    pub fn add_scalar(&self, scalar: f64) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn sub_scalar(&self, scalar: f64) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x - scalar).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn mul_scalar(&self, scalar: f64) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn div_scalar(&self, scalar: f64) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        Array::from_vec(data, self.shape.clone())
    }

    /// Reduction operations
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Array<f64> {
        assert!(axis < self.ndim(), "Axis out of bounds");

        let mut result_shape = self.shape.clone();
        result_shape.remove(axis);
        if result_shape.is_empty() {
            result_shape.push(1);
        }

        let mut result = Array::zeros(result_shape.clone());
        let result_size: usize = result_shape.iter().product();

        for result_idx in 0..result_size {
            let mut sum = 0.0;
            let result_indices = Self::unravel_index(result_idx, &result_shape);

            for i in 0..self.shape[axis] {
                let mut full_indices = Vec::new();
                let mut result_iter = result_indices.iter();

                for (dim_idx, _) in self.shape.iter().enumerate() {
                    if dim_idx == axis {
                        full_indices.push(i);
                    } else {
                        full_indices.push(*result_iter.next().unwrap());
                    }
                }

                sum += self[full_indices.as_slice()];
            }

            result[result_indices.as_slice()] = sum;
        }
        result
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.len() as f64
    }

    pub fn mean_axis(&self, axis: usize) -> Array<f64> {
        let sum_result = self.sum_axis(axis);
        let divisor = self.shape[axis] as f64;
        sum_result.div_scalar(divisor)
    }

    pub fn max(&self) -> f64 {
        self.data
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    pub fn min(&self) -> f64 {
        self.data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
    }

    /// Element-wise mathematical functions
    pub fn exp(&self) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.exp()).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn ln(&self) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.ln()).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn sin(&self) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sin()).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn cos(&self) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.cos()).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn sqrt(&self) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sqrt()).collect();
        Array::from_vec(data, self.shape.clone())
    }

    pub fn pow(&self, exponent: f64) -> Array<f64> {
        let data: Vec<f64> = self.data.iter().map(|&x| x.powf(exponent)).collect();
        Array::from_vec(data, self.shape.clone())
    }
}

// Operator implementations using broadcasting
impl Add for &Array<f64> {
    type Output = Array<f64>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add_broadcast(rhs).expect("Shapes not broadcastable")
    }
}

impl Sub for &Array<f64> {
    type Output = Array<f64>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_broadcast(rhs).expect("Shapes not broadcastable")
    }
}

impl Mul for &Array<f64> {
    type Output = Array<f64>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_broadcast(rhs).expect("Shapes not broadcastable")
    }
}

impl Div for &Array<f64> {
    type Output = Array<f64>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div_broadcast(rhs).expect("Shapes not broadcastable")
    }
}

// Scalar operations using trait implementations
impl Add<f64> for &Array<f64> {
    type Output = Array<f64>;
    fn add(self, scalar: f64) -> Self::Output {
        self.add_scalar(scalar)
    }
}

impl Sub<f64> for &Array<f64> {
    type Output = Array<f64>;
    fn sub(self, scalar: f64) -> Self::Output {
        self.sub_scalar(scalar)
    }
}

impl Mul<f64> for &Array<f64> {
    type Output = Array<f64>;
    fn mul(self, scalar: f64) -> Self::Output {
        self.mul_scalar(scalar)
    }
}

impl Div<f64> for &Array<f64> {
    type Output = Array<f64>;
    fn div(self, scalar: f64) -> Self::Output {
        self.div_scalar(scalar)
    }
}
