use super::Array;

impl Array<f64> {
    /// Matrix multiplication (dot product)
    pub fn dot(&self, other: &Array<f64>) -> Array<f64> {
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d(other),
            (1, 1) => self.dot_1d(other),
            (2, 1) => self.matvec(other),
            (1, 2) => self.vecmat(other),
            _ => panic!("Unsupported dimensions for dot product"),
        }
    }

    /// 1D vector dot product
    fn dot_1d(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(self.shape[0], other.shape[0], "Vector lengths must match");
        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>();
        Array::from_vec(vec![result], vec![1])
    }

    /// 2D matrix multiplication
    fn matmul_2d(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(
            self.shape[1], other.shape[0],
            "Matrix dimensions incompatible for multiplication"
        );

        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        let mut result = Array::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self[(i, l)] * other[(l, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    /// Matrix-vector multiplication
    fn matvec(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(
            self.shape[1], other.shape[0],
            "Matrix-vector dimensions incompatible"
        );

        let m = self.shape[0];
        let mut result = Array::zeros(vec![m]);

        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..self.shape[1] {
                sum += self[(i, j)] * other.data[j];
            }
            result.data[i] = sum;
        }
        result
    }

    /// Vector-matrix multiplication
    fn vecmat(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(
            self.shape[0], other.shape[0],
            "Vector-matrix dimensions incompatible"
        );

        let n = other.shape[1];
        let mut result = Array::zeros(vec![n]);

        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..other.shape[0] {
                sum += self[&[{ i }][..]] * other[&[{ i }, { j }][..]];
            }
            result[&[{ j }][..]] = sum;
        }
        result
    }

    /// Matrix determinant (2x2 and 3x3 only)
    pub fn det(&self) -> f64 {
        assert_eq!(self.ndim(), 2, "Determinant requires 2D matrix");
        assert_eq!(self.shape[0], self.shape[1], "Matrix must be square");

        match self.shape[0] {
            1 => self.data[0],
            2 => self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)],
            3 => {
                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(0, 2)];
                let d = self[(1, 0)];
                let e = self[(1, 1)];
                let f = self[(1, 2)];
                let g = self[(2, 0)];
                let h = self[(2, 1)];
                let i = self[(2, 2)];

                a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            }
            _ => panic!("Determinant only implemented for 1x1, 2x2, and 3x3 matrices"),
        }
    }

    /// Matrix trace (sum of diagonal elements)
    pub fn trace(&self) -> f64 {
        assert_eq!(self.ndim(), 2, "Trace requires 2D matrix");
        let min_dim = self.shape[0].min(self.shape[1]);

        (0..min_dim).map(|i| self[(i, i)]).sum()
    }

    /// Matrix inverse (2x2 only for now)
    pub fn inv(&self) -> Option<Array<f64>> {
        assert_eq!(self.ndim(), 2, "Inverse requires 2D matrix");
        assert_eq!(self.shape[0], self.shape[1], "Matrix must be square");

        match self.shape[0] {
            1 => {
                let elem = self[(0, 0)];
                if elem.abs() < 1e-10 {
                    None
                } else {
                    Some(Array::from_vec(vec![1.0 / elem], vec![1, 1]))
                }
            }
            2 => {
                let det = self.det();
                if det.abs() < 1e-10 {
                    return None; // Matrix is singular
                }

                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(1, 0)];
                let d = self[(1, 1)];

                let inv_det = 1.0 / det;
                let data = vec![d * inv_det, -b * inv_det, -c * inv_det, a * inv_det];

                Some(Array::from_vec(data, vec![2, 2]))
            }
            _ => {
                panic!("Matrix inverse only implemented for 1x1 and 2x2 matrices");
            }
        }
    }

    /// QR decomposition (Gram-Schmidt process)
    pub fn qr(&self) -> (Array<f64>, Array<f64>) {
        assert_eq!(self.ndim(), 2, "QR decomposition requires 2D matrix");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut q = Array::zeros(vec![m, n]);
        let mut r = Array::zeros(vec![n, n]);

        // Gram-Schmidt process
        for j in 0..n {
            // Get j-th column of A
            let mut v = Vec::new();
            for i in 0..m {
                v.push(self[(i, j)]);
            }

            // Subtract projections of previous columns
            for k in 0..j {
                let mut dot_product = 0.0;
                for i in 0..m {
                    dot_product += v[i] * q[(i, k)];
                }
                r[(k, j)] = dot_product;

                for i in 0..m {
                    v[i] -= dot_product * q[(i, k)];
                }
            }

            // Normalize
            let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            r[(j, j)] = norm;

            if norm > 1e-10 {
                for i in 0..m {
                    q[(i, j)] = v[i] / norm;
                }
            }
        }

        (q, r)
    }

    /// Solve linear system Ax = b using QR decomposition
    pub fn solve(&self, b: &Array<f64>) -> Option<Array<f64>> {
        assert_eq!(self.ndim(), 2, "A must be 2D matrix");
        assert_eq!(b.ndim(), 1, "b must be 1D vector");
        assert_eq!(self.shape[0], b.shape[0], "Dimensions incompatible");
        assert_eq!(self.shape[0], self.shape[1], "A must be square");

        let n = self.shape[0];
        let (q, r) = self.qr();

        // Solve Rx = Q^T * b
        let mut qtb = Array::zeros(vec![n]);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += q[(j, i)] * b[&[j][..]];
            }
            qtb[&[i][..]] = sum;
        }

        // Back substitution
        let mut x = Array::zeros(vec![n]);
        for i in (0..n).rev() {
            let mut sum = qtb[&[i][..]];
            for j in (i + 1)..n {
                sum -= r[(i, j)] * x[&[j][..]];
            }

            if r[(i, i)].abs() < 1e-10 {
                return None; // Singular matrix
            }

            x[&[i][..]] = sum / r[(i, i)];
        }

        Some(x)
    }

    /// Compute eigenvalues and eigenvectors (2x2 only)
    pub fn eig(&self) -> Option<(Array<f64>, Array<f64>)> {
        assert_eq!(self.ndim(), 2, "Eigendecomposition requires 2D matrix");
        assert_eq!(self.shape[0], self.shape[1], "Matrix must be square");

        if self.shape[0] == 2 {
            let a = self[(0, 0)];
            let b = self[(0, 1)];
            let c = self[(1, 0)];
            let d = self[(1, 1)];

            // Characteristic polynomial: λ² - (a+d)λ + (ad-bc) = 0
            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant < 0.0 {
                return None; // Complex eigenvalues
            }

            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;

            let eigenvalues = Array::from_vec(vec![lambda1, lambda2], vec![2]);

            // Compute eigenvectors
            let mut eigenvectors = Array::zeros(vec![2, 2]);

            // Eigenvector for λ₁
            if b.abs() > 1e-10 {
                eigenvectors[(0, 0)] = 1.0;
                eigenvectors[(1, 0)] = (lambda1 - a) / b;
            } else if c.abs() > 1e-10 {
                eigenvectors[(0, 0)] = (lambda1 - d) / c;
                eigenvectors[(1, 0)] = 1.0;
            } else {
                eigenvectors[(0, 0)] = 1.0;
                eigenvectors[(1, 0)] = 0.0;
            }

            // Eigenvector for λ₂
            if b.abs() > 1e-10 {
                eigenvectors[(0, 1)] = 1.0;
                eigenvectors[(1, 1)] = (lambda2 - a) / b;
            } else if c.abs() > 1e-10 {
                eigenvectors[(0, 1)] = (lambda2 - d) / c;
                eigenvectors[(1, 1)] = 1.0;
            } else {
                eigenvectors[(0, 1)] = 0.0;
                eigenvectors[(1, 1)] = 1.0;
            }

            // Normalize eigenvectors
            for j in 0..2 {
                let norm = (eigenvectors[(0, j)].powi(2) + eigenvectors[(1, j)].powi(2)).sqrt();
                if norm > 1e-10 {
                    eigenvectors[(0, j)] /= norm;
                    eigenvectors[(1, j)] /= norm;
                }
            }

            Some((eigenvalues, eigenvectors))
        } else {
            panic!("Eigendecomposition only implemented for 2x2 matrices");
        }
    }

    /// Singular Value Decomposition (SVD) - simplified version for 2x2 matrices
    pub fn svd(&self) -> (Array<f64>, Array<f64>, Array<f64>) {
        assert_eq!(self.ndim(), 2, "SVD requires 2D matrix");

        if self.shape[0] == 2 && self.shape[1] == 2 {
            // For 2x2 matrix A, compute A^T * A and A * A^T
            let at = self.transpose();
            let ata = at.dot(self);
            let aat = self.dot(&at);

            // Get eigenvalues and eigenvectors
            let (s_squared, v) = ata.eig().expect("Failed to compute eigenvalues");
            let (_, u) = aat.eig().expect("Failed to compute eigenvalues");

            // Singular values are square roots of eigenvalues
            let s1 = s_squared[&[0][..]].max(0.0).sqrt();
            let s2 = s_squared[&[1][..]].max(0.0).sqrt();
            let s = Array::from_vec(vec![s1, s2], vec![2]);

            (u, s, v.transpose())
        } else {
            panic!("SVD only implemented for 2x2 matrices");
        }
    }

    /// Compute matrix norm (Frobenius norm)
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Compute matrix rank (approximate, using SVD)
    pub fn rank(&self, tolerance: Option<f64>) -> usize {
        let tol = tolerance.unwrap_or(1e-10);

        if self.shape[0] == 2 && self.shape[1] == 2 {
            let (_, s, _) = self.svd();
            s.data.iter().filter(|&&x| x > tol).count()
        } else {
            // For non-square matrices, use determinant-based approach
            let min_dim = self.shape[0].min(self.shape[1]);
            if min_dim == 1 {
                if self.data.iter().any(|&x| x.abs() > tol) {
                    1
                } else {
                    0
                }
            } else {
                // Simplified: just check if determinant is non-zero for square submatrices
                if self.shape[0] == self.shape[1] {
                    if self.det().abs() > tol {
                        self.shape[0]
                    } else {
                        self.shape[0] - 1
                    }
                } else {
                    min_dim // Conservative estimate
                }
            }
        }
    }

    /// Check if matrix is symmetric
    pub fn is_symmetric(&self, tolerance: Option<f64>) -> bool {
        assert_eq!(self.ndim(), 2, "Symmetry check requires 2D matrix");
        assert_eq!(self.shape[0], self.shape[1], "Matrix must be square");

        let tol = tolerance.unwrap_or(1e-10);
        let n = self.shape[0];

        for i in 0..n {
            for j in 0..n {
                if (self[(i, j)] - self[(j, i)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is orthogonal
    pub fn is_orthogonal(&self, tolerance: Option<f64>) -> bool {
        assert_eq!(self.ndim(), 2, "Orthogonality check requires 2D matrix");
        assert_eq!(self.shape[0], self.shape[1], "Matrix must be square");

        let tol = tolerance.unwrap_or(1e-10);
        let at = self.transpose();
        let should_be_identity = self.dot(&at);
        let n = self.shape[0];

        // Check if A * A^T = I
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (should_be_identity[(i, j)] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}
