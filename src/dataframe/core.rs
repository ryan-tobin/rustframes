use super::Series;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    pub columns: Vec<String>,
    pub data: Vec<Series>,
}

impl DataFrame {
    pub fn new(columns: Vec<(String, Series)>) -> Self {
        if !columns.is_empty() {
            let first_len = columns[0].1.len();
            for (name, series) in &columns {
                if series.len() != first_len {
                    panic!("All columns must have the same length. Column '{}' has length {}, expected {}", name, series.len(), first_len);
                }
            }
        }

        let (names, series): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        DataFrame {
            columns: names,
            data: series,
        }
    }

    /// Create empty DataFrame with specified column names and types
    pub fn empty(columns: Vec<(String, SeriesType)>) -> Self {
        let series: Vec<Series> = columns
            .iter()
            .map(|(_, dtype)| match dtype {
                SeriesType::Int64 => Series::Int64(Vec::new()),
                SeriesType::Float64 => Series::Float64(Vec::new()),
                SeriesType::Bool => Series::Bool(Vec::new()),
                SeriesType::Utf8 => Series::Utf8(Vec::new()),
            })
            .collect();

        let names: Vec<String> = columns.into_iter().map(|(name, _)| name).collect();
        DataFrame {
            columns: names,
            data: series,
        }
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }

    /// Check if DataFrame is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get shape (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        (self.len(), self.columns.len())
    }

    /// Get first n rows
    pub fn head(&self, n: usize) -> DataFrame {
        let new_data: Vec<Series> = self
            .data
            .iter()
            .map(|s| match s {
                Series::Int64(v) => Series::Int64(v.iter().take(n).cloned().collect()),
                Series::Float64(v) => Series::Float64(v.iter().take(n).cloned().collect()),
                Series::Bool(v) => Series::Bool(v.iter().take(n).cloned().collect()),
                Series::Utf8(v) => Series::Utf8(v.iter().take(n).cloned().collect()),
            })
            .collect();

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
        }
    }

    /// Get last n rows
    pub fn tail(&self, n: usize) -> DataFrame {
        let len = self.len();
        let start = len.saturating_sub(n);

        let new_data: Vec<Series> = self
            .data
            .iter()
            .map(|s| match s {
                Series::Int64(v) => Series::Int64(v.iter().skip(start).cloned().collect()),
                Series::Float64(v) => Series::Float64(v.iter().skip(start).cloned().collect()),
                Series::Bool(v) => Series::Bool(v.iter().skip(start).cloned().collect()),
                Series::Utf8(v) => Series::Utf8(v.iter().skip(start).cloned().collect()),
            })
            .collect();

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
        }
    }

    /// Select specific columns
    pub fn select(&self, cols: &[&str]) -> DataFrame {
        let mut new_cols = Vec::new();
        let mut new_data = Vec::new();

        for col in cols {
            if let Some(pos) = self.columns.iter().position(|c| c == col) {
                new_cols.push(self.columns[pos].clone());
                new_data.push(self.data[pos].clone());
            } else {
                panic!("Column '{}' not found", col);
            }
        }

        DataFrame {
            columns: new_cols,
            data: new_data,
        }
    }

    /// Get a single column as a Series
    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns
            .iter()
            .position(|c| c == name)
            .map(|pos| &self.data[pos])
    }

    /// Filter rows based on a boolean mask
    pub fn filter(&self, mask: &[bool]) -> DataFrame {
        assert_eq!(
            mask.len(),
            self.len(),
            "Mask length must match DataFrame length"
        );

        let new_data: Vec<Series> = self
            .data
            .iter()
            .map(|s| match s {
                Series::Int64(v) => Series::Int64(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(&val, &keep)| if keep { Some(val) } else { None })
                        .collect(),
                ),
                Series::Float64(v) => Series::Float64(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(&val, &keep)| if keep { Some(val) } else { None })
                        .collect(),
                ),
                Series::Bool(v) => Series::Bool(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(&val, &keep)| if keep { Some(val) } else { None })
                        .collect(),
                ),
                Series::Utf8(v) => Series::Utf8(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(val, &keep)| if keep { Some(val.clone()) } else { None })
                        .collect(),
                ),
            })
            .collect();

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
        }
    }

    /// Sort by column
    pub fn sort_by(&self, column: &str, ascending: bool) -> DataFrame {
        let col_idx = self
            .columns
            .iter()
            .position(|c| c == column)
            .expect("Column not found");

        let mut indices: Vec<usize> = (0..self.len()).collect();

        match &self.data[col_idx] {
            Series::Int64(values) => {
                indices.sort_by(|&a, &b| {
                    if ascending {
                        values[a].cmp(&values[b])
                    } else {
                        values[b].cmp(&values[a])
                    }
                });
            }
            Series::Float64(values) => {
                indices.sort_by(|&a, &b| {
                    if ascending {
                        values[a].partial_cmp(&values[b]).unwrap()
                    } else {
                        values[b].partial_cmp(&values[a]).unwrap()
                    }
                });
            }
            Series::Bool(values) => {
                indices.sort_by(|&a, &b| {
                    if ascending {
                        values[a].cmp(&values[b])
                    } else {
                        values[b].cmp(&values[a])
                    }
                });
            }
            Series::Utf8(values) => {
                indices.sort_by(|&a, &b| {
                    if ascending {
                        values[a].cmp(&values[b])
                    } else {
                        values[b].cmp(&values[a])
                    }
                });
            }
        }

        let new_data: Vec<Series> = self
            .data
            .iter()
            .map(|s| match s {
                Series::Int64(v) => Series::Int64(indices.iter().map(|&i| v[i]).collect()),
                Series::Float64(v) => Series::Float64(indices.iter().map(|&i| v[i]).collect()),
                Series::Bool(v) => Series::Bool(indices.iter().map(|&i| v[i]).collect()),
                Series::Utf8(v) => Series::Utf8(indices.iter().map(|&i| v[i].clone()).collect()),
            })
            .collect();

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
        }
    }

    /// Add a new column
    pub fn with_column(&self, name: String, series: Series) -> DataFrame {
        assert_eq!(
            series.len(),
            self.len(),
            "New column length must match DataFrame length"
        );

        let mut new_columns = self.columns.clone();
        let mut new_data = self.data.clone();

        // Check if column already exists
        if let Some(pos) = new_columns.iter().position(|c| c == &name) {
            new_data[pos] = series;
        } else {
            new_columns.push(name);
            new_data.push(series);
        }

        DataFrame {
            columns: new_columns,
            data: new_data,
        }
    }

    /// Drop columns
    pub fn drop(&self, cols: &[&str]) -> DataFrame {
        let cols_to_drop: HashSet<&str> = cols.iter().cloned().collect();
        let mut new_columns = Vec::new();
        let mut new_data = Vec::new();

        for (i, col_name) in self.columns.iter().enumerate() {
            if !cols_to_drop.contains(col_name.as_str()) {
                new_columns.push(col_name.clone());
                new_data.push(self.data[i].clone());
            }
        }

        DataFrame {
            columns: new_columns,
            data: new_data,
        }
    }

    /// Inner join with another DataFrame
    pub fn join(&self, other: &DataFrame, on: &str, how: JoinType) -> DataFrame {
        let left_col_idx = self
            .columns
            .iter()
            .position(|c| c == on)
            .expect("Join column not found in left DataFrame");
        let right_col_idx = other
            .columns
            .iter()
            .position(|c| c == on)
            .expect("Join column not found in right DataFrame");

        match how {
            JoinType::Inner => self.inner_join(other, left_col_idx, right_col_idx, on),
            JoinType::Left => self.left_join(other, left_col_idx, right_col_idx, on),
            JoinType::Right => other.left_join(self, right_col_idx, left_col_idx, on),
            JoinType::Outer => self.outer_join(other, left_col_idx, right_col_idx, on),
        }
    }

    fn inner_join(
        &self,
        other: &DataFrame,
        left_col_idx: usize,
        right_col_idx: usize,
        _on: &str,
    ) -> DataFrame {
        let mut result_columns = self.columns.clone();

        // Add columns from right DataFrame (excluding join column)
        for (i, col) in other.columns.iter().enumerate() {
            if i != right_col_idx {
                let mut new_name = col.clone();
                if result_columns.contains(&new_name) {
                    new_name = format!("{}_y", col);
                }
                result_columns.push(new_name);
            }
        }

        // Build hash map for right DataFrame
        let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
        if let Series::Utf8(right_values) = &other.data[right_col_idx] {
            for (idx, value) in right_values.iter().enumerate() {
                right_map
                    .entry(value.clone())
                    .or_default()
                    .push(idx);
            }
        }

        let mut result_data: Vec<Vec<String>> = vec![Vec::new(); result_columns.len()];

        // Process left DataFrame
        if let Series::Utf8(left_values) = &self.data[left_col_idx] {
            for (left_idx, left_value) in left_values.iter().enumerate() {
                if let Some(right_indices) = right_map.get(left_value) {
                    for &right_idx in right_indices {
                        // Add left row
                        for (col_idx, series) in self.data.iter().enumerate() {
                            let value = match series {
                                Series::Int64(v) => v[left_idx].to_string(),
                                Series::Float64(v) => v[left_idx].to_string(),
                                Series::Bool(v) => v[left_idx].to_string(),
                                Series::Utf8(v) => v[left_idx].clone(),
                            };
                            result_data[col_idx].push(value);
                        }

                        // Add right row (excluding join column)
                        let mut result_col_idx = self.columns.len();
                        for (col_idx, series) in other.data.iter().enumerate() {
                            if col_idx != right_col_idx {
                                let value = match series {
                                    Series::Int64(v) => v[right_idx].to_string(),
                                    Series::Float64(v) => v[right_idx].to_string(),
                                    Series::Bool(v) => v[right_idx].to_string(),
                                    Series::Utf8(v) => v[right_idx].clone(),
                                };
                                result_data[result_col_idx].push(value);
                                result_col_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        // Convert result to DataFrame
        let result_series: Vec<Series> = result_data
            .into_iter()
            .map(Series::Utf8)
            .collect();

        DataFrame {
            columns: result_columns,
            data: result_series,
        }
    }

    fn left_join(
        &self,
        other: &DataFrame,
        left_col_idx: usize,
        right_col_idx: usize,
        on: &str,
    ) -> DataFrame {
        // Similar to inner join but includes all left rows
        // Implementation would be similar but always include left rows, padding with nulls
        self.inner_join(other, left_col_idx, right_col_idx, on) // Simplified for now
    }

    fn outer_join(
        &self,
        other: &DataFrame,
        left_col_idx: usize,
        right_col_idx: usize,
        on: &str,
    ) -> DataFrame {
        // Full outer join - includes all rows from both DataFrames
        // Implementation would combine left and right joins
        self.inner_join(other, left_col_idx, right_col_idx, on) // Simplified for now
    }

    /// Describe numeric columns (basic statistics)
    pub fn describe(&self) -> DataFrame {
        let mut stats_data: Vec<(String, Series)> = Vec::new();
        let stats = vec!["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

        for stat in stats {
            let mut values = Vec::new();

            for series in &self.data {
                let value = match series {
                    Series::Float64(v) if !v.is_empty() => match stat {
                        "count" => v.len() as f64,
                        "mean" => v.iter().sum::<f64>() / v.len() as f64,
                        "std" => {
                            let mean = v.iter().sum::<f64>() / v.len() as f64;
                            let variance =
                                v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
                            variance.sqrt()
                        }
                        "min" => v.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                        "max" => v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                        "25%" | "50%" | "75%" => {
                            let mut sorted = v.clone();
                            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let idx = match stat {
                                "25%" => sorted.len() / 4,
                                "50%" => sorted.len() / 2,
                                "75%" => 3 * sorted.len() / 4,
                                _ => 0,
                            };
                            sorted.get(idx).copied().unwrap_or(0.0)
                        }
                        _ => 0.0,
                    },
                    Series::Int64(v) if !v.is_empty() => match stat {
                        "count" => v.len() as f64,
                        "mean" => v.iter().sum::<i64>() as f64 / v.len() as f64,
                        "std" => {
                            let mean = v.iter().sum::<i64>() as f64 / v.len() as f64;
                            let variance =
                                v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>()
                                    / v.len() as f64;
                            variance.sqrt()
                        }
                        "min" => *v.iter().min().unwrap() as f64,
                        "max" => *v.iter().max().unwrap() as f64,
                        _ => 0.0,
                    },
                    _ => f64::NAN, // Non-numeric or empty series
                };

                values.push(value);
            }

            stats_data.push((stat.to_string(), Series::Float64(values)));
        }

        DataFrame::new(stats_data)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SeriesType {
    Int64,
    Float64,
    Bool,
    Utf8,
}
