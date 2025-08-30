#[derive(Debug, Clone, PartialEq)]
pub enum Series {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Bool(Vec<bool>),
    Utf8(Vec<String>),
}

impl Series {
    pub fn len(&self) -> usize {
        match self {
            Series::Int64(v) => v.len(),
            Series::Float64(v) => v.len(),
            Series::Bool(v) => v.len(),
            Series::Utf8(v) => v.len(),
        }
    }
}

impl From<Vec<i64>> for Series {
    fn from(v: Vec<i64>) -> Self {
        Series::Int64(v)
    }
}

impl From<Vec<f64>> for Series {
    fn from(v: Vec<f64>) -> Self {
        Series::Float64(v)
    }
}

impl From<Vec<bool>> for Series {
    fn from(v: Vec<bool>) -> Self {
        Series::Bool(v)
    }
}

impl From<Vec<&str>> for Series {
    fn from(v: Vec<&str>) -> Self {
        Series::Utf8(v.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<Vec<String>> for Series {
    fn from(v: Vec<String>) -> Self {
        Series::Utf8(v)
    }
}