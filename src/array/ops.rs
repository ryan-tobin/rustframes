use super::Array;
use std::ops::{Add, Sub, Mul, Div};

impl<'a> Add for &'a Array<f64> {
    type Output = Array<f64>;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Array::from_vec(data, self.shape)
    }
}

impl<'a> Sub for &'a Array<f64> {
    type Output = Array<f64>;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Array::from_vec(data, self.shape)
    }
}

impl<'a> Mul for &'a Array<f64> {
    type Output = Array<f64>;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Array::from_vec(data, self.shape)
    }
}

impl<'a> Div for &'a Array<f64> {
    type Output = Array<f64>;
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Array::from_vec(data, self.shape)
    }
}