use super::{DataFrame, Series};
use std::collections::VecDeque;

pub struct Window<'a> {
    df: &'a DataFrame,
    window_size: usize,
    partition_by: Option<String>,
    order_by: Option<(String, bool)>, // column, ascending
}

impl<'a> Window<'a> {
    pub fn new(df: &'a DataFrame, window_size: usize) -> Self {
        Window {
            df,
            window_size,
            partition_by: None,
            order_by: None,
        }
    }

    pub fn partition_by(mut self, column: &str) -> Self {
        self.partition_by = Some(column.to_string());
        self
    }

    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        self.order_by = Some((column.to_string(), ascending));
        self
    }

    /// Rolling sum over window
    pub fn rolling_sum(&self, column: &str) -> Series {
        self.apply_rolling_function(column, |window| window.iter().sum())
    }

    /// Rolling mean over window
    pub fn rolling_mean(&self, column: &str) -> Series {
        self.apply_rolling_function(column, |window| {
            let sum: f64 = window.iter().sum();
            sum / window.len() as f64
        })
    }

    /// Rolling standard deviation
    pub fn rolling_std(&self, column: &str) -> Series {
        self.apply_rolling_function(column, |window| {
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            variance.sqrt()
        })
    }

    /// Rolling minimum
    pub fn rolling_min(&self, column: &str) -> Series {
        self.apply_rolling_function(column, |window| {
            window.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
        })
    }

    /// Rolling maximum
    pub fn rolling_max(&self, column: &str) -> Series {
        self.apply_rolling_function(column, |window| {
            window.iter().fold(f64::INFINITY, |acc, &x| acc.max(x))
        })
    }

    /// Exponentially weighted moving average
    pub fn ewm(&self, column: &str, alpha: f64) -> Series {
        let col_idx = self
            .df
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        let mut result = Vec::new();

        if let Series::Float64(values) = &self.df.data[col_idx] {
            let mut ewm = values[0];
            result.push(ewm);

            for &value in &values[1..] {
                ewm = alpha * value + (1.0 - alpha) * ewm;
                result.push(ewm);
            }
        }

        Series::Float64(result)
    }

    /// Lag operation (shift values)
    pub fn lag(&self, column: &str, periods: usize) -> Series {
        let col_idx = self
            .df
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        match &self.df.data[col_idx] {
            Series::Float64(values) => {
                let mut result = vec![f64::NAN; periods];
                result.extend_from_slice(&values[..values.len().saturating_sub(periods)]);
                Series::Float64(result)
            }
            Series::Int64(values) => {
                let mut result = vec![0; periods]; // Use 0 as null for integers
                result.extend_from_slice(&values[..values.len().saturating_sub(periods)]);
                Series::Int64(result)
            }
            Series::Utf8(values) => {
                let mut result = vec!["".to_string(); periods];
                result.extend(
                    values[..values.len().saturating_sub(periods)]
                        .iter()
                        .cloned(),
                );
                Series::Utf8(result)
            }
            Series::Bool(values) => {
                let mut result = vec![false; periods];
                result.extend_from_slice(&values[..values.len().saturating_sub(periods)]);
                Series::Bool(result)
            }
        }
    }

    /// Lead operation (negative shift)
    pub fn lead(&self, column: &str, periods: usize) -> Series {
        let col_idx = self
            .df
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        match &self.df.data[col_idx] {
            Series::Float64(values) => {
                let mut result = values[periods..].to_vec();
                result.extend(vec![f64::NAN; periods]);
                Series::Float64(result)
            }
            Series::Int64(values) => {
                let mut result = values[periods..].to_vec();
                result.extend(vec![0; periods]);
                Series::Int64(result)
            }
            Series::Utf8(values) => {
                let mut result = values[periods..].to_vec();
                result.extend(vec!["".to_string(); periods]);
                Series::Utf8(result)
            }
            Series::Bool(values) => {
                let mut result = values[periods..].to_vec();
                result.extend(vec![false; periods]);
                Series::Bool(result)
            }
        }
    }

    /// Percent change
    pub fn pct_change(&self, column: &str) -> Series {
        let col_idx = self
            .df
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        match &self.df.data[col_idx] {
            Series::Float64(values) => {
                let mut result = vec![f64::NAN]; // First value is NaN
                for i in 1..values.len() {
                    let pct = (values[i] - values[i - 1]) / values[i - 1];
                    result.push(pct);
                }
                Series::Float64(result)
            }
            Series::Int64(values) => {
                let mut result = vec![f64::NAN]; // First value is NaN
                for i in 1..values.len() {
                    let pct = (values[i] - values[i - 1]) as f64 / values[i - 1] as f64;
                    result.push(pct);
                }
                Series::Float64(result)
            }
            _ => panic!("Percent change only supported for numeric columns"),
        }
    }

    fn apply_rolling_function<F>(&self, column: &str, func: F) -> Series
    where
        F: Fn(&[f64]) -> f64,
    {
        let col_idx = self
            .df
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        if let Series::Float64(values) = &self.df.data[col_idx] {
            let mut result = Vec::new();
            let mut window: VecDeque<f64> = VecDeque::with_capacity(self.window_size);

            for &value in values {
                window.push_back(value);

                if window.len() > self.window_size {
                    window.pop_front();
                }

                if window.len() == self.window_size {
                    let window_slice: Vec<f64> = window.iter().cloned().collect();
                    result.push(func(&window_slice));
                } else {
                    result.push(f64::NAN);
                }
            }

            Series::Float64(result)
        } else {
            panic!("Rolling function only supported for Float64 columns");
        }
    }
}
