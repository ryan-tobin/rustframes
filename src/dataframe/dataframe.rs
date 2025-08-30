use super::Series;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    pub columns: Vec<String>,
    pub data: Vec<Series>,
}

impl DataFrame {
    pub fn new(columns: Vec<(String, Series)>) -> Self {
        let (names, series): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        DataFrame { columns: names, data: series }
    }

    pub fn head(&self, n: usize) -> Vec<(String, Series)> {
        self.columns
            .iter()
            .cloned()
            .zip(self.data.iter().cloned())
            .map(|(name, s)| match s {
                Series::Int64(v) => (name, Series::Int64(v.into_iter().take(n).collect())),
                Series::Float64(v) => (name, Series::Float64(v.into_iter().take(n).collect())),
                Series::Bool(v) => (name, Series::Bool(v.into_iter().take(n).collect())),
                Series::Utf8(v) => (name, Series::Utf8(v.into_iter().take(n).collect())),
            })
            .collect()
    }

    pub fn select(&self, cols: &[&str]) -> DataFrame {
        let mut new_cols = Vec::new();
        let mut new_data = Vec::new();
        for col in cols {
            if let Some(pos) = self.columns.iter().position(|c| c == col) {
                new_cols.push(self.columns[pos].clone());
                new_data.push(self.data[pos].clone());
            }
        }
        DataFrame { columns: new_cols, data: new_data }
    }
}