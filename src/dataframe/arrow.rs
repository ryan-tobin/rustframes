use super::{DataFrame, Series};
use arrow::array::{
    Array as ArrowArray, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;

impl DataFrame {
    /// Convert DataFrame to Apache Arrow RecordBatch
    pub fn to_arrow(&self) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for (i, column_name) in self.columns.iter().enumerate() {
            match &self.data[i] {
                Series::Int64(values) => {
                    fields.push(Field::new(column_name, DataType::Int64, false));
                    let array = Int64Array::from(values.clone());
                    arrays.push(Arc::new(array));
                }
                Series::Float64(values) => {
                    fields.push(Field::new(column_name, DataType::Float64, false));
                    let array = Float64Array::from(values.clone());
                    arrays.push(Arc::new(array));
                }
                Series::Bool(values) => {
                    fields.push(Field::new(column_name, DataType::Boolean, false));
                    let array = BooleanArray::from(values.clone());
                    arrays.push(Arc::new(array));
                }
                Series::Utf8(values) => {
                    fields.push(Field::new(column_name, DataType::Utf8, false));
                    let array = StringArray::from(values.clone());
                    arrays.push(Arc::new(array));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let record_batch = RecordBatch::try_new(schema, arrays)?;
        Ok(record_batch)
    }

    /// Create DataFrame from Apache Arrow RecordBatch
    pub fn from_arrow(batch: &RecordBatch) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = batch.schema();
        let mut columns = Vec::new();
        let mut data = Vec::new();

        for (i, field) in schema.fields().iter().enumerate() {
            let column_name = field.name().clone();
            let array = batch.column(i);

            let series = match field.data_type() {
                DataType::Int64 => {
                    let int_array = array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or("Failed to downcast to Int64Array")?;
                    let values: Vec<i64> =
                        (0..int_array.len()).map(|i| int_array.value(i)).collect();
                    Series::Int64(values)
                }
                DataType::Float64 => {
                    let float_array = array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or("Failed to downcast to Float64Array")?;
                    let values: Vec<f64> = (0..float_array.len())
                        .map(|i| float_array.value(i))
                        .collect();
                    Series::Float64(values)
                }
                DataType::Boolean => {
                    let bool_array = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or("Failed to downcast to BooleanArray")?;
                    let values: Vec<bool> =
                        (0..bool_array.len()).map(|i| bool_array.value(i)).collect();
                    Series::Bool(values)
                }
                DataType::Utf8 => {
                    let string_array = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("Failed to downcast to StringArray")?;
                    let values: Vec<String> = (0..string_array.len())
                        .map(|i| string_array.value(i).to_string())
                        .collect();
                    Series::Utf8(values)
                }
                _ => return Err(format!("Unsupported data type: {:?}", field.data_type()).into()),
            };

            columns.push(column_name);
            data.push(series);
        }

        Ok(DataFrame { columns, data })
    }

    /// Read Parquet file using Arrow
    pub fn from_parquet(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs::File;

        let file = File::open(path)?;
        let mut arrow_reader =
            parquet::arrow::arrow_reader::ArrowReaderBuilder::try_new(file)?.build()?;

        if let Some(batch_result) = arrow_reader.next() {
            let batch = batch_result?;
            Self::from_arrow(&batch)
        } else {
            Err("No data in Parquet file".into())
        }
    }

    /// Write DataFrame to Parquet file
    pub fn to_parquet(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;

        let batch = self.to_arrow()?;
        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;

        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Create DataFrame from Arrow IPC (Feather) file
    pub fn from_ipc(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use arrow::ipc::reader::FileReader;
        use std::fs::File;

        let file = File::open(path)?;
        let mut reader = FileReader::try_new(file, None)?;

        if let Some(batch_result) = reader.next() {
            let batch = batch_result?;
            Self::from_arrow(&batch)
        } else {
            Err("No data in IPC file".into())
        }
    }

    /// Write DataFrame to Arrow IPC (Feather) file
    pub fn to_ipc(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use arrow::ipc::writer::FileWriter;
        use std::fs::File;

        let batch = self.to_arrow()?;
        let file = File::create(path)?;
        let mut writer = FileWriter::try_new(file, &batch.schema())?;

        // propagate any error from write
        writer.write(&batch)?;
        writer.finish()?;

        Ok(())
    }

    /// Convert to Arrow and perform operations using Arrow Compute
    pub fn arrow_filter(
        &self,
        column: &str,
        predicate: ArrowPredicate,
    ) -> Result<DataFrame, Box<dyn std::error::Error>> {
        use arrow::array::{BooleanArray, Float64Array, Int64Array};
        use arrow::compute;
        use arrow::datatypes::DataType;

        let batch = self.to_arrow()?;
        let col_index = batch
            .schema()
            .column_with_name(column)
            .ok_or("Column not found")?
            .0;
        let array = batch.column(col_index);

        let filter_array: BooleanArray = match predicate {
            ArrowPredicate::Gt(value) => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(float_array.len());
                    for i in 0..float_array.len() {
                        mask.push(float_array.value(i) > value);
                    }
                    BooleanArray::from(mask)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(int_array.len());
                    for i in 0..int_array.len() {
                        mask.push((int_array.value(i) as f64) > value);
                    }
                    BooleanArray::from(mask)
                }
                _ => return Err("Unsupported type for comparison".into()),
            },
            ArrowPredicate::Lt(value) => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(float_array.len());
                    for i in 0..float_array.len() {
                        mask.push(float_array.value(i) < value);
                    }
                    BooleanArray::from(mask)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(int_array.len());
                    for i in 0..int_array.len() {
                        mask.push((int_array.value(i) as f64) < value);
                    }
                    BooleanArray::from(mask)
                }
                _ => return Err("Unsupported type for comparison".into()),
            },
            ArrowPredicate::Eq(value) => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(float_array.len());
                    for i in 0..float_array.len() {
                        mask.push(float_array.value(i) == value);
                    }
                    BooleanArray::from(mask)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    let mut mask: Vec<bool> = Vec::with_capacity(int_array.len());
                    for i in 0..int_array.len() {
                        mask.push((int_array.value(i) as f64) == value);
                    }
                    BooleanArray::from(mask)
                }
                _ => return Err("Unsupported type for comparison".into()),
            },
        };

        let filtered_arrays: Result<Vec<ArrayRef>, _> = batch
            .columns()
            .iter()
            .map(|array| compute::filter(array, &filter_array))
            .collect();

        let filtered_batch = RecordBatch::try_new(batch.schema(), filtered_arrays?)?;
        Self::from_arrow(&filtered_batch)
    }

    /// Aggregation using Arrow compute
    pub fn arrow_agg(
        &self,
        column: &str,
        agg_func: ArrowAggFunc,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        use arrow::compute;

        let batch = self.to_arrow()?;
        let col_index = batch
            .schema()
            .column_with_name(column)
            .ok_or("Column not found")?
            .0;
        let array = batch.column(col_index);

        let result = match agg_func {
            ArrowAggFunc::Sum => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    compute::sum(float_array).unwrap_or(0.0)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    compute::sum(int_array).unwrap_or(0) as f64
                }
                _ => return Err("Sum not supported for this type".into()),
            },
            ArrowAggFunc::Min => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    compute::min(float_array).unwrap_or(f64::NAN)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    compute::min(int_array).unwrap_or(0) as f64
                }
                _ => return Err("Min not supported for this type".into()),
            },
            ArrowAggFunc::Max => match array.data_type() {
                DataType::Float64 => {
                    let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    compute::max(float_array).unwrap_or(f64::NAN)
                }
                DataType::Int64 => {
                    let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    compute::max(int_array).unwrap_or(0) as f64
                }
                _ => return Err("Max not supported for this type".into()),
            },
        };

        Ok(result)
    }

    /// Zero-copy slice using Arrow
    pub fn arrow_slice(
        &self,
        offset: usize,
        length: usize,
    ) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let batch = self.to_arrow()?;
        let sliced_arrays: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|array| array.slice(offset, length))
            .collect();

        let sliced_batch = RecordBatch::try_new(batch.schema(), sliced_arrays)?;
        Self::from_arrow(&sliced_batch)
    }
}

#[derive(Debug, Clone)]
pub enum ArrowPredicate {
    Gt(f64),
    Lt(f64),
    Eq(f64),
}

#[derive(Debug, Clone)]
pub enum ArrowAggFunc {
    Sum,
    Min,
    Max,
}

// Integration with NumPy (requires Python bindings)
#[cfg(feature = "python")]
pub mod numpy_interop {
    use super::*;
    use numpy::{PyArray, PyReadonlyArray1};
    use pyo3::prelude::*;
    use pyo3::types::PyArray1;

    impl DataFrame {
        /// Convert Series to NumPy array
        pub fn series_to_numpy<'py>(
            &self,
            py: Python<'py>,
            column: &str,
        ) -> PyResult<&'py PyArray1<f64>> {
            let series = self
                .get_column(column)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Column not found"))?;

            match series {
                Series::Float64(values) => Ok(PyArray::from_slice(py, values)),
                Series::Int64(values) => {
                    let float_values: Vec<f64> = values.iter().map(|&x| x as f64).collect();
                    Ok(PyArray::from_vec(py, float_values))
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Only numeric columns can be converted to NumPy arrays",
                )),
            }
        }

        /// Create DataFrame from NumPy array
        pub fn from_numpy(array: PyReadonlyArray1<f64>, column_name: &str) -> Self {
            let values: Vec<f64> = array.as_slice().unwrap().to_vec();
            DataFrame::new(vec![(column_name.to_string(), Series::Float64(values))])
        }
    }
}

// Memory mapping for large files
pub mod memory_mapped {
    use super::*;
    use memmap2::MmapOptions;
    use std::fs::File;

    impl DataFrame {
        /// Memory-mapped CSV reading for large files
        pub fn from_csv_map(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let file = File::open(path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };

            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(&mmap[..]);

            let headers = rdr.headers()?.clone();
            let mut raw_data: Vec<Vec<String>> = vec![Vec::new(); headers.len()];

            for result in rdr.records() {
                let record = result?;
                for (i, field) in record.iter().enumerate() {
                    if i < raw_data.len() {
                        raw_data[i].push(field.to_string());
                    }
                }
            }

            let mut series_data = Vec::new();
            for col_data in raw_data {
                let col_type = Self::infer_column_type(&col_data);
                let series = match col_type {
                    crate::dataframe::core::SeriesType::Float64 => {
                        let parsed: Vec<f64> = col_data
                            .iter()
                            .map(|s| s.trim().parse().unwrap_or(0.0))
                            .collect();
                        Series::Float64(parsed)
                    }
                    crate::dataframe::core::SeriesType::Int64 => {
                        let parsed: Vec<i64> = col_data
                            .iter()
                            .map(|s| s.trim().parse().unwrap_or(0))
                            .collect();
                        Series::Int64(parsed)
                    }
                    crate::dataframe::core::SeriesType::Bool => {
                        let parsed: Vec<bool> = col_data
                            .iter()
                            .map(|s| Self::parse_bool(s.trim()).unwrap_or(false))
                            .collect();
                        Series::Bool(parsed)
                    }
                    crate::dataframe::core::SeriesType::Utf8 => Series::Utf8(col_data),
                };
                series_data.push(series);
            }

            let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
            Ok(DataFrame::new(
                column_names.into_iter().zip(series_data).collect(),
            ))
        }
    }
}
