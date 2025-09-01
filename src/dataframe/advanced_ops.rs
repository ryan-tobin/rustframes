use wide::f64x4;

use crate::{dataframe::window::Window, DataFrame, Series};

impl DataFrame {
    /// Create window for rolling operations
    pub fn window<'a>(&'a self, window_size: usize) -> Window<'a> {
        Window::new(self, window_size)
    }

    /// Pivot table functionality
    pub fn pivot_table(
        &self,
        index: &str,
        columns: &str,
        values: &str,
        aggfunc: PivotAggFunc,
    ) -> DataFrame {
        use std::collections::{BTreeSet, HashMap};

        let index_idx = self
            .columns
            .iter()
            .position(|c| c == index)
            .expect("Index column not found");
        let columns_idx = self
            .columns
            .iter()
            .position(|c| c == columns)
            .expect("Columns column not found");
        let values_idx = self
            .columns
            .iter()
            .position(|c| c == values)
            .expect("Values column not found");

        let mut unique_columns: BTreeSet<String> = BTreeSet::new();
        let mut unique_indices: BTreeSet<String> = BTreeSet::new();

        match (&self.data[index_idx], &self.data[columns_idx]) {
            (Series::Utf8(idx_vals), Series::Utf8(col_vals)) => {
                unique_indices.extend(idx_vals.iter().cloned());
                unique_columns.extend(col_vals.iter().cloned());
            }
            _ => panic!("Pivot currently only supports string indices and columns"),
        }

        let mut result_columns = vec![index.to_string()];
        result_columns.extend(unique_columns.iter().cloned());

        let mut result_data: Vec<Vec<String>> = vec![Vec::new(); result_columns.len()];

        for idx_val in &unique_indices {
            result_data[0].push(idx_val.clone());
        }

        let mut agg_map: HashMap<(String, String), Vec<f64>> = HashMap::new();

        if let (Series::Utf8(idx_vals), Series::Utf8(col_vals), Series::Float64(val_vals)) = (
            &self.data[index_idx],
            &self.data[columns_idx],
            &self.data[values_idx],
        ) {
            for i in 0..idx_vals.len() {
                let key = (idx_vals[i].clone(), col_vals[i].clone());
                agg_map.entry(key).or_default().push(val_vals[i]);
            }
        }

        for (row_idx, idx_val) in unique_indices.iter().enumerate() {
            for (col_idx, col_val) in unique_columns.iter().enumerate() {
                let key = (idx_val.clone(), col_val.clone());
                let value = if let Some(values) = agg_map.get(&key) {
                    match aggfunc {
                        PivotAggFunc::Sum => simd_sum(values).to_string(),
                        PivotAggFunc::Mean => (simd_sum(values) / values.len() as f64).to_string(),
                        PivotAggFunc::Count => values.len().to_string(),
                        PivotAggFunc::Min => values
                            .iter()
                            .fold(f64::INFINITY, |acc, &x| acc.min(x))
                            .to_string(),
                        PivotAggFunc::Max => values
                            .iter()
                            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
                            .to_string(),
                    }
                } else {
                    "0".to_string()
                };

                while result_data[col_idx + 1].len() <= row_idx {
                    result_data[col_idx + 1].push("0".to_string());
                }
                result_data[col_idx + 1][row_idx] = value;
            }
        }

        let series_data: Vec<Series> = result_data.into_iter().map(Series::Utf8).collect();

        DataFrame {
            columns: result_columns,
            data: series_data,
        }
    }

    /// Melt operation (unpivot)
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: &[&str],
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> DataFrame {
        let var_name = var_name.unwrap_or("variable");
        let value_name = value_name.unwrap_or("value");

        let id_indices: Vec<usize> = id_vars
            .iter()
            .map(|col| {
                self.columns
                    .iter()
                    .position(|c| c == col)
                    .expect("ID column not found")
            })
            .collect();

        let value_indices: Vec<usize> = value_vars
            .iter()
            .map(|col| {
                self.columns
                    .iter()
                    .position(|c| c == col)
                    .expect("Value column not found")
            })
            .collect();

        let mut result_columns = Vec::new();
        let mut result_data: Vec<Vec<String>> = Vec::new();

        for &col in id_vars {
            result_columns.push(col.to_string());
            result_data.push(Vec::new());
        }

        result_columns.push(var_name.to_string());
        result_columns.push(value_name.to_string());
        result_data.push(Vec::new());
        result_data.push(Vec::new());

        for row_idx in 0..self.len() {
            for (value_col_idx, &value_idx) in value_indices.iter().enumerate() {
                for (id_col_idx, &id_idx) in id_indices.iter().enumerate() {
                    let value = match &self.data[id_idx] {
                        Series::Int64(v) => v[row_idx].to_string(),
                        Series::Float64(v) => v[row_idx].to_string(),
                        Series::Bool(v) => v[row_idx].to_string(),
                        Series::Utf8(v) => v[row_idx].clone(),
                    };
                    result_data[id_col_idx].push(value);
                }

                result_data[id_vars.len()].push(value_vars[value_col_idx].to_string());

                let value = match &self.data[value_idx] {
                    Series::Int64(v) => v[row_idx].to_string(),
                    Series::Float64(v) => v[row_idx].to_string(),
                    Series::Bool(v) => v[row_idx].to_string(),
                    Series::Utf8(v) => v[row_idx].clone(),
                };
                result_data[id_vars.len() + 1].push(value);
            }
        }

        let series_data: Vec<Series> = result_data.into_iter().map(Series::Utf8).collect();

        DataFrame {
            columns: result_columns,
            data: series_data,
        }
    }

    /// Cross-validation split
    pub fn cv_split(&self, n_folds: usize, shuffle: bool) -> Vec<(DataFrame, DataFrame)> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..self.len()).collect();

        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        let fold_size = self.len() / n_folds;
        let mut folds = Vec::new();

        for fold in 0..n_folds {
            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                self.len()
            } else {
                start + fold_size
            };

            let test_indices: std::collections::HashSet<usize> =
                indices[start..end].iter().cloned().collect();

            let mask: Vec<bool> = (0..self.len()).map(|i| test_indices.contains(&i)).collect();

            let test_df = self.filter(&mask);
            let train_df = self.filter(&mask.iter().map(|b| !b).collect::<Vec<_>>());

            folds.push((train_df, test_df));
        }

        folds
    }

    /// Sample rows randomly
    pub fn sample(&self, n: usize, replace: bool) -> DataFrame {
        use rand::seq::SliceRandom;
        use rand::Rng;

        let mut rng = rand::rng();
        let indices: Vec<usize> = if replace {
            (0..n).map(|_| rng.random_range(0..self.len())).collect()
        } else {
            let mut all_indices: Vec<usize> = (0..self.len()).collect();
            all_indices.shuffle(&mut rng);
            all_indices.into_iter().take(n.min(self.len())).collect()
        };

        let mask: Vec<bool> = (0..self.len()).map(|i| indices.contains(&i)).collect();

        self.filter(&mask)
    }
}

#[derive(Debug, Clone)]
pub enum PivotAggFunc {
    Sum,
    Mean,
    Count,
    Min,
    Max,
}

/// SIMD accelerated sum using `wide::f64x4`
fn simd_sum(values: &[f64]) -> f64 {
    let mut acc = f64x4::from([0.0; 4]);
    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        acc += f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    let arr: [f64; 4] = acc.into();
    let mut total: f64 = arr.iter().sum();
    for &r in remainder {
        total += r;
    }

    total
}
