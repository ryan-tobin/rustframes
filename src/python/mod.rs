// src/python/mod.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use crate::{Array, DataFrame, Series};

/// Python wrapper for RustFrames Array
#[pyclass(name = "Array")]
pub struct PyArray {
    inner: Array<f64>,
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyArray {
            inner: Array::from_vec(data, shape),
        })
    }

    /// Create array from NumPy array
    #[classmethod]
    fn from_numpy(_cls: &PyType, array: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let shape = array.shape().to_vec();
        let data = array.as_slice()?.to_vec();
        Ok(PyArray {
            inner: Array::from_vec(data, shape),
        })
    }

    /// Convert to NumPy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        let shape = &self.inner.shape;
        if shape.len() == 2 {
            let array = PyArray2::from_vec2(py, &vec![self.inner.data.clone()])?
                .reshape([shape[0], shape[1]])?;
            Ok(array)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only 2D arrays can be converted to NumPy arrays currently"
            ))
        }
    }

    #[classmethod]
    fn zeros(_cls: &PyType, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyArray {
            inner: Array::zeros(shape),
        })
    }

    #[classmethod]
    fn ones(_cls: &PyType, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyArray {
            inner: Array::<f64>::ones(shape),
        })
    }

    #[classmethod]
    fn arange(_cls: &PyType, start: f64, stop: f64, step: f64) -> PyResult<Self> {
        Ok(PyArray {
            inner: Array::arange(start, stop, step),
        })
    }

    fn dot(&self, other: &PyArray) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.dot(&other.inner),
        })
    }

    fn transpose(&self) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.transpose(),
        })
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.reshape(new_shape),
        })
    }

    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn max(&self) -> f64 {
        self.inner.max()
    }

    fn min(&self) -> f64 {
        self.inner.min()
    }

    fn __add__(&self, other: &PyArray) -> PyResult<PyArray> {
        match self.inner.add_broadcast(&other.inner) {
            Some(result) => Ok(PyArray { inner: result }),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shapes are not broadcastable"
            )),
        }
    }

    fn __sub__(&self, other: &PyArray) -> PyResult<PyArray> {
        match self.inner.sub_broadcast(&other.inner) {
            Some(result) => Ok(PyArray { inner: result }),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shapes are not broadcastable"
            )),
        }
    }

    fn __mul__(&self, other: &PyArray) -> PyResult<PyArray> {
        match self.inner.mul_broadcast(&other.inner) {
            Some(result) => Ok(PyArray { inner: result }),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shapes are not broadcastable"
            )),
        }
    }

    fn __truediv__(&self, other: &PyArray) -> PyResult<PyArray> {
        match self.inner.div_broadcast(&other.inner) {
            Some(result) => Ok(PyArray { inner: result }),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shapes are not broadcastable"
            )),
        }
    }

    fn __getitem__(&self, indices: &PyAny) -> PyResult<f64> {
        if let Ok(tuple) = indices.downcast::<pyo3::types::PyTuple>() {
            let idx: Vec<usize> = tuple
                .iter()
                .map(|item| item.extract::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            
            match self.inner.get(&idx) {
                Some(value) => Ok(*value),
                None => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Index out of bounds"
                )),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Index must be a tuple"
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(shape={:?}, data={:?})",
            self.inner.shape,
            if self.inner.len() <= 10 {
                self.inner.data.clone()
            } else {
                let mut preview = self.inner.data[..5].to_vec();
                preview.extend_from_slice(&self.inner.data[self.inner.len()-5..]);
                preview
            }
        )
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Python wrapper for RustFrames DataFrame
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    #[new]
    fn new(data: &PyDict) -> PyResult<Self> {
        let mut columns = Vec::new();

        for (key, value) in data.iter() {
            let column_name: String = key.extract()?;
            
            let series = if let Ok(int_list) = value.downcast::<PyList>() {
                // Try to extract as integers first
                if let Ok(ints) = int_list
                    .iter()
                    .map(|item| item.extract::<i64>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Series::Int64(ints)
                } else if let Ok(floats) = int_list
                    .iter()
                    .map(|item| item.extract::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Series::Float64(floats)
                } else if let Ok(bools) = int_list
                    .iter()
                    .map(|item| item.extract::<bool>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Series::Bool(bools)
                } else {
                    // Default to strings
                    let strings = int_list
                        .iter()
                        .map(|item| item.to_string())
                        .collect::<Vec<_>>();
                    Series::Utf8(strings)
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Column data must be a list"
                ));
            };

            columns.push((column_name, series));
        }

        Ok(PyDataFrame {
            inner: DataFrame::new(columns),
        })
    }

    #[classmethod]
    fn from_csv(_cls: &PyType, path: &str) -> PyResult<Self> {
        match DataFrame::from_csv(path) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read CSV: {}", e)
            )),
        }
    }

    #[classmethod]
    fn from_pandas(_cls: &PyType, pandas_df: &PyAny) -> PyResult<Self> {
        // This would integrate with pandas DataFrames
        // For now, we'll extract the data manually
        let columns_method = pandas_df.getattr("columns")?;
        let values_method = pandas_df.getattr("values")?;
        
        let column_names: Vec<String> = columns_method
            .iter()?
            .map(|col| col?.extract::<String>())
            .collect::<Result<Vec<_>, _>>()?;

        // This is a simplified implementation
        // In practice, you'd want to handle different dtypes properly
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Pandas integration not yet implemented"
        ))
    }

    fn to_csv(&self, path: &str) -> PyResult<()> {
        match self.inner.to_csv(path) {
            Ok(()) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to write CSV: {}", e)
            )),
        }
    }

    fn head(&self, n: Option<usize>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.head(n.unwrap_or(5)),
        })
    }

    fn tail(&self, n: Option<usize>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.tail(n.unwrap_or(5)),
        })
    }

    fn select(&self, columns: Vec<&str>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.select(&columns),
        })
    }

    fn filter(&self, mask: Vec<bool>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.filter(&mask),
        })
    }

    fn sort_values(&self, by: &str, ascending: Option<bool>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.sort_by(by, ascending.unwrap_or(true)),
        })
    }

    fn groupby(&self, by: &str) -> PyResult<PyGroupBy> {
        Ok(PyGroupBy {
            inner: self.inner.groupby(by),
        })
    }

    fn describe(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.describe(),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        // Create a simple string representation
        let mut result = String::new();
        
        // Column headers
        result.push_str(&self.inner.columns.join("\t"));
        result.push('\n');
        
        // Data rows (show first 5 and last 5 if more than 10 rows)
        let n_rows = self.inner.len();
        let show_rows = if n_rows <= 10 {
            (0..n_rows).collect::<Vec<_>>()
        } else {
            let mut rows = (0..5).collect::<Vec<_>>();
            rows.extend((n_rows-5)..n_rows);
            rows
        };

        for (i, &row_idx) in show_rows.iter().enumerate() {
            if i == 5 && n_rows > 10 {
                result.push_str("...\n");
            }
            
            let row_data: Vec<String> = self.inner.data
                .iter()
                .map(|series| match series {
                    Series::Int64(v) => v[row_idx].to_string(),
                    Series::Float64(v) => format!("{:.2}", v[row_idx]),
                    Series::Bool(v) => v[row_idx].to_string(),
                    Series::Utf8(v) => v[row_idx].clone(),
                })
                .collect();
            
            result.push_str(&row_data.join("\t"));
            result.push('\n');
        }
        
        result.push_str(&format!("\n[{} rows x {} columns]", n_rows, self.inner.columns.len()));
        result
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns.clone()
    }

    // Dictionary-style access to columns
    fn __getitem__(&self, column: &str) -> PyResult<PyList> {
        match self.inner.get_column(column) {
            Some(series) => {
                Python::with_gil(|py| {
                    match series {
                        Series::Int64(v) => {
                            let items: Vec<PyObject> = v.iter()
                                .map(|&x| x.into_py(py))
                                .collect();
                            Ok(PyList::new(py, items))
                        }
                        Series::Float64(v) => {
                            let items: Vec<PyObject> = v.iter()
                                .map(|&x| x.into_py(py))
                                .collect();
                            Ok(PyList::new(py, items))
                        }
                        Series::Bool(v) => {
                            let items: Vec<PyObject> = v.iter()
                                .map(|&x| x.into_py(py))
                                .collect();
                            Ok(PyList::new(py, items))
                        }
                        Series::Utf8(v) => {
                            let items: Vec<PyObject> = v.iter()
                                .map(|x| x.clone().into_py(py))
                                .collect();
                            Ok(PyList::new(py, items))
                        }
                    }
                })
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Column '{}' not found", column)
            )),
        }
    }
}

/// Python wrapper for GroupBy operations
#[pyclass(name = "GroupBy")]
pub struct PyGroupBy {
    inner: crate::dataframe::groupby::GroupBy<'static>,
}

#[pymethods]
impl PyGroupBy {
    fn count(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.count(),
        })
    }

    fn sum(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.sum(),
        })
    }

    fn mean(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.mean(),
        })
    }

    fn min(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.min(),
        })
    }

    fn max(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.max(),
        })
    }

    fn std(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.std(),
        })
    }
}

/// Module registration for Python
#[pymodule]
fn rustframes(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyGroupBy>()?;
    
    // Add module-level functions
    #[pyfn(m)]
    fn read_csv(path: &str) -> PyResult<PyDataFrame> {
        match DataFrame::from_csv(path) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read CSV: {}", e)
            )),
        }
    }

    #[pyfn(m)]
    fn zeros(shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: Array::zeros(shape),
        })
    }

    #[pyfn(m)]
    fn ones(shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: Array::<f64>::ones(shape),
        })
    }

    #[pyfn(m)]
    fn arange(start: f64, stop: f64, step: Option<f64>) -> PyResult<PyArray> {
        let step = step.unwrap_or(1.0);
        Ok(PyArray {
            inner: Array::arange(start, stop, step),
        })
    }

    Ok(())
}