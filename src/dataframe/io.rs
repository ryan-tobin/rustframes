use super::{DataFrame, Series};
use crate::dataframe::core::SeriesType;
use csv::{ReaderBuilder, WriterBuilder};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Debug)]
pub struct BoolParseError;

impl DataFrame {
    /// Read CSV with automatic type inference
    pub fn from_csv(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_csv_with_options(path, CsvReadOptions::default())
    }

    /// Read CSV with custom options
    pub fn from_csv_with_options(
        path: &str,
        options: CsvReadOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(options.delimiter)
            .has_headers(options.has_headers)
            .from_reader(BufReader::new(file));

        let headers = if options.has_headers {
            rdr.headers()?.clone()
        } else {
            // Generate default column names
            csv::StringRecord::from(
                (0..rdr.headers()?.len())
                    .map(|i| format!("column_{}", i))
                    .collect::<Vec<_>>(),
            )
        };

        // First pass: collect all data as strings and infer types
        let mut raw_data: Vec<Vec<String>> = vec![Vec::new(); headers.len()];
        for result in rdr.records() {
            let record = result?;
            for (i, field) in record.iter().enumerate() {
                if i < raw_data.len() {
                    raw_data[i].push(field.to_string());
                }
            }
        }

        // Infer column types
        let mut column_types = Vec::new();
        for col_data in &raw_data {
            column_types.push(Self::infer_column_type(col_data));
        }

        // Convert to appropriate Series types
        let mut series_data = Vec::new();
        for (i, col_data) in raw_data.into_iter().enumerate() {
            let series = match column_types[i] {
                SeriesType::Int64 => {
                    let parsed: Result<Vec<i64>, _> =
                        col_data.iter().map(|s| s.trim().parse::<i64>()).collect();
                    match parsed {
                        Ok(values) => Series::Int64(values),
                        Err(_) => Series::Utf8(col_data), // Fallback to string
                    }
                }
                SeriesType::Float64 => {
                    let parsed: Result<Vec<f64>, _> =
                        col_data.iter().map(|s| s.trim().parse::<f64>()).collect();
                    match parsed {
                        Ok(values) => Series::Float64(values),
                        Err(_) => Series::Utf8(col_data), // Fallback to string
                    }
                }
                SeriesType::Bool => {
                    let parsed: Result<Vec<bool>, _> = col_data
                        .iter()
                        .map(|s| Self::parse_bool(s.trim()))
                        .collect();
                    match parsed {
                        Ok(values) => Series::Bool(values),
                        Err(_) => Series::Utf8(col_data), // Fallback to string
                    }
                }
                SeriesType::Utf8 => Series::Utf8(col_data),
            };
            series_data.push(series);
        }

        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        Ok(DataFrame::new(
            column_names.into_iter().zip(series_data).collect(),
        ))
    }

    /// Write DataFrame to CSV
    pub fn to_csv(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.to_csv_with_options(path, CsvWriteOptions::default())
    }

    /// Write DataFrame to CSV with custom options
    pub fn to_csv_with_options(
        &self,
        path: &str,
        options: CsvWriteOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut wtr = WriterBuilder::new()
            .delimiter(options.delimiter)
            .from_writer(BufWriter::new(file));

        // Write headers
        if options.write_headers {
            wtr.write_record(&self.columns)?;
        }

        // Write data rows
        for row_idx in 0..self.len() {
            let mut record = Vec::new();
            for series in &self.data {
                let value = match series {
                    Series::Int64(v) => v[row_idx].to_string(),
                    Series::Float64(v) => {
                        if options.float_precision > 0 {
                            format!("{:.prec$}", v[row_idx], prec = options.float_precision)
                        } else {
                            v[row_idx].to_string()
                        }
                    }
                    Series::Bool(v) => v[row_idx].to_string(),
                    Series::Utf8(v) => v[row_idx].clone(),
                };
                record.push(value);
            }
            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Read from JSON Lines format
    pub fn from_jsonl(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs;
        let content = fs::read_to_string(path)?;

        let mut all_columns: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut records: Vec<HashMap<String, serde_json::Value>> = Vec::new();

        // Parse each line and collect all possible column names
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let record: HashMap<String, serde_json::Value> = serde_json::from_str(line)?;
            for key in record.keys() {
                all_columns.insert(key.clone());
            }
            records.push(record);
        }

        let columns: Vec<String> = all_columns.into_iter().collect();
        let mut column_data: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize column data
        for col in &columns {
            column_data.insert(col.clone(), Vec::new());
        }

        // Fill data, handling missing values
        for record in records {
            for col in &columns {
                let value = match record.get(col) {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Number(n)) => n.to_string(),
                    Some(serde_json::Value::Bool(b)) => b.to_string(),
                    Some(serde_json::Value::Null) => "".to_string(),
                    Some(_) => "".to_string(), // Arrays, objects -> empty string
                    None => "".to_string(),    // Missing field
                };
                column_data.get_mut(col).unwrap().push(value);
            }
        }

        // Convert to Series with type inference
        let mut series_data = Vec::new();
        let mut final_columns = Vec::new();

        for col in columns {
            let col_values = column_data.remove(&col).unwrap();
            let col_type = Self::infer_column_type(&col_values);

            let series = match col_type {
                SeriesType::Int64 => {
                    let parsed: Vec<i64> = col_values
                        .iter()
                        .map(|s| s.trim().parse::<i64>().unwrap_or(0))
                        .collect();
                    Series::Int64(parsed)
                }
                SeriesType::Float64 => {
                    let parsed: Vec<f64> = col_values
                        .iter()
                        .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                        .collect();
                    Series::Float64(parsed)
                }
                SeriesType::Bool => {
                    let parsed: Vec<bool> = col_values
                        .iter()
                        .map(|s| Self::parse_bool(s.trim()).unwrap_or(false))
                        .collect();
                    Series::Bool(parsed)
                }
                SeriesType::Utf8 => Series::Utf8(col_values),
            };

            final_columns.push(col);
            series_data.push(series);
        }

        Ok(DataFrame::new(
            final_columns.into_iter().zip(series_data).collect(),
        ))
    }

    /// Write DataFrame to JSON Lines format
    pub fn to_jsonl(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        for row_idx in 0..self.len() {
            let mut record = serde_json::Map::new();

            for (col_idx, col_name) in self.columns.iter().enumerate() {
                let value = match &self.data[col_idx] {
                    Series::Int64(v) => {
                        serde_json::Value::Number(serde_json::Number::from(v[row_idx]))
                    }
                    Series::Float64(v) => {
                        if let Some(n) = serde_json::Number::from_f64(v[row_idx]) {
                            serde_json::Value::Number(n)
                        } else {
                            serde_json::Value::Null
                        }
                    }
                    Series::Bool(v) => serde_json::Value::Bool(v[row_idx]),
                    Series::Utf8(v) => serde_json::Value::String(v[row_idx].clone()),
                };
                record.insert(col_name.clone(), value);
            }

            let line = serde_json::to_string(&record)?;
            writeln!(file, "{}", line)?;
        }

        Ok(())
    }

    /// Write DataFrame to regular JSON format (array of objects)
    pub fn to_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let mut records = Vec::new();

        for row_idx in 0..self.len() {
            let mut record = serde_json::Map::new();

            for (col_idx, col_name) in self.columns.iter().enumerate() {
                let value = match &self.data[col_idx] {
                    Series::Int64(v) => {
                        serde_json::Value::Number(serde_json::Number::from(v[row_idx]))
                    }
                    Series::Float64(v) => {
                        if let Some(n) = serde_json::Number::from_f64(v[row_idx]) {
                            serde_json::Value::Number(n)
                        } else {
                            serde_json::Value::Null
                        }
                    }
                    Series::Bool(v) => serde_json::Value::Bool(v[row_idx]),
                    Series::Utf8(v) => serde_json::Value::String(v[row_idx].clone()),
                };
                record.insert(col_name.clone(), value);
            }

            records.push(serde_json::Value::Object(record));
        }

        let json_array = serde_json::Value::Array(records);
        let mut file = File::create(path)?;
        writeln!(file, "{}", serde_json::to_string_pretty(&json_array)?)?;

        Ok(())
    }

    /// Infer the type of a column from string data
    pub fn infer_column_type(data: &[String]) -> SeriesType {
        if data.is_empty() {
            return SeriesType::Utf8;
        }

        let mut int_count = 0;
        let mut float_count = 0;
        let mut bool_count = 0;
        let total = data.len();

        for value in data {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.parse::<i64>().is_ok() {
                int_count += 1;
            } else if trimmed.parse::<f64>().is_ok() {
                float_count += 1;
            } else if Self::parse_bool(trimmed).is_ok() {
                bool_count += 1;
            }
        }

        let threshold = (total as f64 * 0.8).ceil() as usize; // 80% threshold

        if bool_count >= threshold {
            SeriesType::Bool
        } else if int_count >= threshold {
            SeriesType::Int64
        } else if (int_count + float_count) >= threshold {
            SeriesType::Float64
        } else {
            SeriesType::Utf8
        }
    }

    /// Parse boolean from string
    pub fn parse_bool(s: &str) -> Result<bool, BoolParseError> {
        match s.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "1" => Ok(true),
            "false" | "f" | "no" | "n" | "0" => Ok(false),
            _ => Err(BoolParseError),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    pub delimiter: u8,
    pub has_headers: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        CsvReadOptions {
            delimiter: b',',
            has_headers: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CsvWriteOptions {
    pub delimiter: u8,
    pub write_headers: bool,
    pub float_precision: usize,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        CsvWriteOptions {
            delimiter: b',',
            write_headers: true,
            float_precision: 0, // 0 means no special formatting
        }
    }
}
