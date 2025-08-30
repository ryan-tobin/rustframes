use super::Array;

impl Array<f64> {
    pub fn dot(&self, other: &Array<f64>) -> Array<f64> {
        assert_eq!(self.shape.1, other.shape.0);
        let (m, n) = self.shape;
        let (_p, q) = other.shape;
        let mut result = Array::zeros((m, q));
        for i in 0..m {
            for j in 0..q {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }
}
