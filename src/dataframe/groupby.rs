use super::{DataFrame, Series};
use std::collections::HashMap;

pub struct GroupBy<'a> {
    df: &'a DataFrame,
    by_column: String,
    groups: HashMap<String, Vec<usize>>,
}

impl<'a> GroupBy<'a> {
    pub fn new(df: &'a DataFrame, by: &str) -> Self {
        let by_column = by.to_string();
        let col_idx = df
            .columns
            .iter()
            .position(|c| c == by)
            .expect("GroupBy column not found");

        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        match &df.data[col_idx] {
            Series::Utf8(values) => {
                for (idx, value) in values.iter().enumerate() {
                    groups.entry(value.clone()).or_default().push(idx);
                }
            }
            Series::Int64(values) => {
                for (idx, &value) in values.iter().enumerate() {
                    groups.entry(value.to_string()).or_default().push(idx);
                }
            }
            Series::Float64(values) => {
                for (idx, &value) in values.iter().enumerate() {
                    groups.entry(value.to_string()).or_default().push(idx);
                }
            }
            Series::Bool(values) => {
                for (idx, &value) in values.iter().enumerate() {
                    groups.entry(value.to_string()).or_default().push(idx);
                }
            }
        }

        GroupBy {
            df,
            by_column,
            groups,
        }
    }

    /// Count occurrences in each group
    pub fn count(&self) -> DataFrame {
        let mut keys = Vec::new();
        let mut counts = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(key.clone());
            counts.push(indices.len() as i64);
        }

        DataFrame::new(vec![
            (self.by_column.clone(), Series::Utf8(keys)),
            ("count".to_string(), Series::Int64(counts)),
        ])
    }

    /// Sum numeric columns by group
    pub fn sum(&self) -> DataFrame {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue; // Skip the groupby column
            }

            let mut group_sums = Vec::new();

            match &self.df.data[col_idx] {
                Series::Int64(values) => {
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let sum: i64 = indices.iter().map(|&i| values[i]).sum();
                        group_sums.push(sum);
                    }
                    result_columns.push((col_name.clone(), Series::Int64(group_sums)));
                }
                Series::Float64(values) => {
                    let mut group_sums = Vec::new();
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let sum: f64 = indices.iter().map(|&i| values[i]).sum();
                        group_sums.push(sum);
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_sums)));
                }
                _ => {
                    // Skip non-numeric columns for sum operation
                    continue;
                }
            }
        }

        DataFrame::new(result_columns)
    }

    /// Mean of numeric columns by group
    pub fn mean(&self) -> DataFrame {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue;
            }

            let mut group_means = Vec::new();

            match &self.df.data[col_idx] {
                Series::Int64(values) => {
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let sum: i64 = indices.iter().map(|&i| values[i]).sum();
                        let mean = sum as f64 / indices.len() as f64;
                        group_means.push(mean);
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_means)));
                }
                Series::Float64(values) => {
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let sum: f64 = indices.iter().map(|&i| values[i]).sum();
                        let mean = sum / indices.len() as f64;
                        group_means.push(mean);
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_means)));
                }
                _ => continue,
            }
        }

        DataFrame::new(result_columns)
    }

    /// Standard deviation of numeric columns by group
    pub fn std(&self) -> DataFrame {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue;
            }

            let mut group_stds = Vec::new();

            match &self.df.data[col_idx] {
                Series::Int64(values) => {
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let values_in_group: Vec<f64> =
                            indices.iter().map(|&i| values[i] as f64).collect();
                        let mean: f64 =
                            values_in_group.iter().sum::<f64>() / values_in_group.len() as f64;
                        let variance = values_in_group
                            .iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>()
                            / values_in_group.len() as f64;
                        group_stds.push(variance.sqrt());
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_stds)));
                }
                Series::Float64(values) => {
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let values_in_group: Vec<f64> =
                            indices.iter().map(|&i| values[i]).collect();
                        let mean: f64 =
                            values_in_group.iter().sum::<f64>() / values_in_group.len() as f64;
                        let variance = values_in_group
                            .iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>()
                            / values_in_group.len() as f64;
                        group_stds.push(variance.sqrt());
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_stds)));
                }
                _ => continue,
            }
        }

        DataFrame::new(result_columns)
    }

    /// Min of numeric columns by group
    pub fn min(&self) -> DataFrame {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue;
            }

            match &self.df.data[col_idx] {
                Series::Int64(values) => {
                    let mut group_mins = Vec::new();
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let min_val = indices.iter().map(|&i| values[i]).min().unwrap_or(0);
                        group_mins.push(min_val);
                    }
                    result_columns.push((col_name.clone(), Series::Int64(group_mins)));
                }
                Series::Float64(values) => {
                    let mut group_mins = Vec::new();
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let min_val = indices
                            .iter()
                            .map(|&i| values[i])
                            .fold(f64::INFINITY, |acc, x| acc.min(x));
                        group_mins.push(min_val);
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_mins)));
                }
                _ => continue,
            }
        }

        DataFrame::new(result_columns)
    }

    /// Max of numeric columns by group
    pub fn max(&self) -> DataFrame {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue;
            }

            match &self.df.data[col_idx] {
                Series::Int64(values) => {
                    let mut group_maxs = Vec::new();
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let max_val = indices.iter().map(|&i| values[i]).max().unwrap_or(0);
                        group_maxs.push(max_val);
                    }
                    result_columns.push((col_name.clone(), Series::Int64(group_maxs)));
                }
                Series::Float64(values) => {
                    let mut group_maxs = Vec::new();
                    for key in self.groups.keys() {
                        let indices = &self.groups[key];
                        let max_val = indices
                            .iter()
                            .map(|&i| values[i])
                            .fold(f64::NEG_INFINITY, |acc, x| acc.max(x));
                        group_maxs.push(max_val);
                    }
                    result_columns.push((col_name.clone(), Series::Float64(group_maxs)));
                }
                _ => continue,
            }
        }

        DataFrame::new(result_columns)
    }

    /// Apply custom aggregation function
    pub fn agg<F>(&self, func: F) -> DataFrame
    where
        F: Fn(&[usize], &Series) -> f64,
    {
        let mut result_columns = vec![(
            self.by_column.clone(),
            Series::Utf8(self.groups.keys().cloned().collect()),
        )];

        for (col_idx, col_name) in self.df.columns.iter().enumerate() {
            if col_name == &self.by_column {
                continue;
            }

            let mut group_results = Vec::new();
            for key in self.groups.keys() {
                let indices = &self.groups[key];
                let result = func(indices, &self.df.data[col_idx]);
                group_results.push(result);
            }

            result_columns.push((col_name.clone(), Series::Float64(group_results)));
        }

        DataFrame::new(result_columns)
    }

    /// Get the first row of each group
    pub fn first(&self) -> DataFrame {
        let mut result_data = vec![Vec::new(); self.df.columns.len()];

        for key in self.groups.keys() {
            let first_idx = self.groups[key][0]; // Get first index in group

            for (col_idx, series) in self.df.data.iter().enumerate() {
                let value = match series {
                    Series::Int64(v) => v[first_idx].to_string(),
                    Series::Float64(v) => v[first_idx].to_string(),
                    Series::Bool(v) => v[first_idx].to_string(),
                    Series::Utf8(v) => v[first_idx].clone(),
                };
                result_data[col_idx].push(value);
            }
        }

        let result_series: Vec<Series> = result_data.into_iter().map(Series::Utf8).collect();

        DataFrame {
            columns: self.df.columns.clone(),
            data: result_series,
        }
    }

    /// Get the last row of each group  
    pub fn last(&self) -> DataFrame {
        let mut result_data = vec![Vec::new(); self.df.columns.len()];

        for key in self.groups.keys() {
            let last_idx = *self.groups[key].last().unwrap(); // Get last index in group

            for (col_idx, series) in self.df.data.iter().enumerate() {
                let value = match series {
                    Series::Int64(v) => v[last_idx].to_string(),
                    Series::Float64(v) => v[last_idx].to_string(),
                    Series::Bool(v) => v[last_idx].to_string(),
                    Series::Utf8(v) => v[last_idx].clone(),
                };
                result_data[col_idx].push(value);
            }
        }

        let result_series: Vec<Series> = result_data.into_iter().map(Series::Utf8).collect();

        DataFrame {
            columns: self.df.columns.clone(),
            data: result_series,
        }
    }

    /// Get size of each group
    pub fn size(&self) -> HashMap<String, usize> {
        self.groups
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Get groups as separate DataFrames
    pub fn get_group(&self, key: &str) -> Option<DataFrame> {
        if let Some(indices) = self.groups.get(key) {
            let mask: Vec<bool> = (0..self.df.len()).map(|i| indices.contains(&i)).collect();
            Some(self.df.filter(&mask))
        } else {
            None
        }
    }
}

// Add groupby method to DataFrame
impl DataFrame {
    /// Group DataFrame by column
    pub fn groupby<'a>(&'a self, by: &str) -> GroupBy<'a> {
        GroupBy::new(self, by)
    }

    /// Convenience method for groupby count (maintains backward compatibility)
    pub fn groupby_count(&self, by: &str) -> DataFrame {
        self.groupby(by).count()
    }
}
