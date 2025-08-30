use super::{DataFrame, Series};
use std::collections::HashMap;

impl DataFrame {
    pub fn groupby_count(&self, by: &str) -> DataFrame {
        let idx = self
            .columns
            .iter()
            .position(|c| c == by)
            .expect("column not found");
        let mut counts: HashMap<String, usize> = HashMap::new();
        if let Series::Utf8(values) = &self.data[idx] {
            for v in values {
                *counts.entry(v.clone()).or_insert(0) += 1;
            }
        }
        let keys: Vec<String> = counts.keys().cloned().collect();
        let vals: Vec<i64> = counts.values().map(|&v| v as i64).collect();
        DataFrame::new(vec![
            (by.to_string(), Series::Utf8(keys)),
            ("count".to_string(), Series::Int64(vals)),
        ])
    }
}
