//! # RustFrames
//!
//! A blazing-fast, memory-safe alternative to NumPy + Pandas, written in Rust.
//! Provides ndarray-like arrays with SIMD acceleration and dataframes with
//! parallel query execution.

use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

/// A simple n-dimensional array backed by a Vec
#[derive(Debug, Clone)]
pub struct Array<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Array<T> {
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().copied().product();
        assert_eq!(data.len(), expected_len, "Data length must match shape");
        Array { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Array<f64> {
    /// SIMD-accelerated elementwise addition
    /// Elementwise addition (scalar loop; replaces previous SIMD helpers)
    pub fn add_simd(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(self.shape, other.shape, "Arrays must have the same shape");

        let mut result_data = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;

        // process in chunks of 4 for a modest unrolling effect
        for i in 0..chunks {
            let start = i * 4;
            result_data.push(self.data[start] + other.data[start]);
            result_data.push(self.data[start + 1] + other.data[start + 1]);
            result_data.push(self.data[start + 2] + other.data[start + 2]);
            result_data.push(self.data[start + 3] + other.data[start + 3]);
        }

        // remainder
        for i in chunks * 4..self.data.len() {
            result_data.push(self.data[i] + other.data[i]);
        }

        Array::from_vec(result_data, self.shape.clone())
    }
    /// SIMD-accelerated dot product for 1D arrays
    /// Dot product for 1D arrays (scalar loop; replaces previous SIMD helpers)
    pub fn dot_1d_simd(&self, other: &Array<f64>) -> f64 {
        assert_eq!(self.shape.len(), 1, "Arrays must be 1D");
        assert_eq!(
            self.shape[0], other.shape[0],
            "Arrays must have same length"
        );

        let mut sum = 0.0;
        let chunks = self.data.len() / 4;

        // process in chunks of 4 for a modest unrolling effect
        for i in 0..chunks {
            let start = i * 4;
            sum += self.data[start] * other.data[start];
            sum += self.data[start + 1] * other.data[start + 1];
            sum += self.data[start + 2] * other.data[start + 2];
            sum += self.data[start + 3] * other.data[start + 3];
        }

        // remainder
        for i in chunks * 4..self.data.len() {
            sum += self.data[i] * other.data[i];
        }

        sum
    }

    /// Parallel matrix multiplication for 2D arrays
    pub fn matmul_parallel(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match");

        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for (j, item) in row.iter_mut().enumerate().take(n) {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += self.data[i * k + kk] * other.data[kk * n + j];
                }
                *item = sum;
            }
        });

        Array::from_vec(result, vec![m, n])
    }

    /// Cache-efficient blocked matrix multiplication
    pub fn matmul_blocked(&self, other: &Array<f64>, block_size: usize) -> Array<f64> {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match");

        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        for i0 in (0..m).step_by(block_size) {
            for j0 in (0..n).step_by(block_size) {
                for k0 in (0..k).step_by(block_size) {
                    let i_max = (i0 + block_size).min(m);
                    let j_max = (j0 + block_size).min(n);
                    let k_max = (k0 + block_size).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = 0.0;
                            for kk in k0..k_max {
                                sum += self.data[i * k + kk] * other.data[kk * n + j];
                            }
                            result[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        Array::from_vec(result, vec![m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_simd() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);
        let c = a.add_simd(&b);
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_dot_1d_simd() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);
        let result = a.dot_1d_simd(&b);
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_matmul_parallel() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul_parallel(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_blocked() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul_blocked(&b, 2);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
