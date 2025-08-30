use super::{DataFrame, Series};
use csv::ReaderBuilder;

impl DataFrame {
    pub fn from_csv(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = ReaderBuilder::new().from_path(path)?;
        let headers = rdr.headers()?.clone();
        let mut cols: Vec<Vec<String>> = vec![Vec::new(); headers.len()];

        for result in rdr.records() {
            let record = result?;
            for (i, field) in record.iter().enumerate() {
                cols[i].push(field.to_string());
            }
        }

        let mut series = Vec::new();
        for col in cols.into_iter() {
            series.push(Series::Utf8(col));
        }

        Ok(DataFrame::new(headers.iter().map(|h| h.to_string()).zip(series).collect()))
    }
}