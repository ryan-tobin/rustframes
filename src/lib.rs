//! # RustFrames
//!
//! A blazing-fast, memory-safe alternative to NumPy + Pandas, written in Rust.
//!
//! RustFrames provides:
//! - N-dimensional arrays with broadcasting support
//! - DataFrame operations with groupby, joins, and filtering
//! - Linear algebra operations (matrix multiplication, decompositions, etc.)
//! - CSV and JSON I/O with automatic type inference
//! - Memory-safe operations with zero-cost abstractions
//!
//! ## Quick Start
//!
//! ### Arrays
//! ```rust
//! use rustframes::array::Array;
//!
//! // Create a 2D array
//! let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//!
//! // Element-wise operations with broadcasting
//! let scalar_mult = &arr * 2.0;
//!
//! // Linear algebra
//! let identity = Array::<f64>::ones(vec![2, 2]);
//! let product = arr.dot(&identity);
//!
//! // Reductions
//! println!("Sum: {}", arr.sum());
//! println!("Mean: {}", arr.mean());
//! ```
//!
//! ### DataFrames
//! ```rust
//! use rustframes::dataframe::{DataFrame, Series};
//!
//! // Create DataFrame
//! let df = DataFrame::new(vec![
//!     ("name".to_string(), Series::from(vec!["Alice", "Bob", "Charlie"])),
//!     ("age".to_string(), Series::from(vec![25, 30, 35])),
//!     ("score".to_string(), Series::from(vec![85.5, 92.0, 78.5])),
//! ]);
//!
//! // Operations
//! let filtered = df.filter(&[true, false, true]);
//! let sorted = df.sort_by("age", true);
//! let grouped = df.groupby("age").mean();
//!
//! // I/O
//! let df_from_csv = DataFrame::from_csv("data.csv")?;
//! df.to_csv("output.csv")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())

pub mod array;
pub mod dataframe;

// Re-export main types for convenience
pub use array::Array;
pub use dataframe::core::JoinType;
pub use dataframe::{DataFrame, Series};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_array_n_dimensional() {
        // Test N-dimensional array creation and indexing
        let arr = Array::from_vec((0..24).map(|x| x as f64).collect(), vec![2, 3, 4]);
        assert_eq!(arr.shape, vec![2, 3, 4]);
        assert_eq!(arr.ndim(), 3);
        assert_eq!(arr[&[0, 0, 0][..]], 0.0);
        assert_eq!(arr[&[1, 2, 3][..]], 23.0);

        // Test reshape
        let reshaped = arr.reshape(vec![6, 4]);
        assert_eq!(reshaped.shape, vec![6, 4]);
        assert_eq!(reshaped.data.len(), 24);
    }

    #[test]
    fn test_array_broadcasting() {
        // Test broadcasting with different shapes
        let arr1 = Array::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let arr2 = Array::from_vec(vec![10.0, 20.0], vec![2, 1]);

        // This should broadcast to shape [2, 3]
        if let Some(result) = arr1.add_broadcast(&arr2) {
            assert_eq!(result.shape, vec![2, 3]);
            assert_eq!(result[&[0, 0][..]], 11.0); // 1 + 10
            assert_eq!(result[&[1, 2][..]], 23.0); // 3 + 20
        }

        // Test scalar operations
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let scaled = &arr + 5.0;
        assert_eq!(scaled.data, vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_array_reductions() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Test basic reductions
        assert_eq!(arr.sum(), 21.0);
        assert_eq!(arr.mean(), 3.5);
        assert_eq!(arr.max(), 6.0);
        assert_eq!(arr.min(), 1.0);

        // Test axis reductions
        let sum_axis_0 = arr.sum_axis(0);
        assert_eq!(sum_axis_0.shape, vec![3]);
        assert_eq!(sum_axis_0.data, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        let mean_axis_1 = arr.mean_axis(1);
        assert_eq!(mean_axis_1.shape, vec![2]);
        assert_eq!(mean_axis_1.data, vec![2.0, 5.0]); // [(1+2+3)/3, (4+5+6)/3]
    }

    #[test]
    fn test_linear_algebra() {
        let matrix = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test determinant
        assert_eq!(matrix.det(), -2.0);

        // Test trace
        assert_eq!(matrix.trace(), 5.0);

        // Test matrix multiplication
        let other = Array::from_vec(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
        let product = matrix.dot(&other);
        assert_eq!(product.data, vec![4.0, 6.0, 10.0, 12.0]);

        // Test matrix inverse (for 2x2)
        if let Some(inv) = matrix.inv() {
            // matrix * inv should be approximately identity
            let should_be_identity = matrix.dot(&inv);
            assert!((should_be_identity[(0, 0)] - 1.0).abs() < 1e-10);
            assert!((should_be_identity[(1, 1)] - 1.0).abs() < 1e-10);
            assert!(should_be_identity[(0, 1)].abs() < 1e-10);
            assert!(should_be_identity[(1, 0)].abs() < 1e-10);
        }

        // Test QR decomposition
        let (q, r) = matrix.qr();
        assert_eq!(q.shape, vec![2, 2]);
        assert_eq!(r.shape, vec![2, 2]);

        // Q should be orthogonal (Q * Q^T = I)
        let qt = q.transpose();
        let should_be_identity = q.dot(&qt);
        assert!((should_be_identity[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((should_be_identity[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dataframe_enhanced() {
        let df = DataFrame::new(vec![
            ("id".to_string(), Series::from(vec![1, 2, 3, 4])),
            (
                "name".to_string(),
                Series::from(vec!["Alice", "Bob", "Charlie", "Diana"]),
            ),
            (
                "score".to_string(),
                Series::from(vec![85.5, 92.0, 78.5, 88.0]),
            ),
            (
                "active".to_string(),
                Series::from(vec![true, true, false, true]),
            ),
        ]);

        // Test shape
        assert_eq!(df.shape(), (4, 4));
        assert_eq!(df.len(), 4);
        assert!(!df.is_empty());

        // Test head/tail
        let head = df.head(2);
        assert_eq!(head.len(), 2);

        let tail = df.tail(2);
        assert_eq!(tail.len(), 2);

        // Test filtering
        let mask = vec![true, false, true, false];
        let filtered = df.filter(&mask);
        assert_eq!(filtered.len(), 2);

        // Test sorting
        let sorted = df.sort_by("score", true); // ascending
        if let Some(Series::Float64(scores)) = sorted.get_column("score") {
            assert!(scores[0] < scores[1]); // Should be sorted ascending
        }

        // Test column operations
        let with_bonus = df.with_column(
            "bonus".to_string(),
            Series::from(vec![100.0, 150.0, 75.0, 120.0]),
        );
        assert_eq!(with_bonus.shape().1, 5); // One more column

        let dropped = df.drop(&["active"]);
        assert_eq!(dropped.shape().1, 3); // One less column
    }

    #[test]
    fn test_groupby_enhanced() {
        let df = DataFrame::new(vec![
            (
                "department".to_string(),
                Series::from(vec!["IT", "HR", "IT", "Finance", "HR"]),
            ),
            (
                "salary".to_string(),
                Series::from(vec![75000, 65000, 80000, 70000, 68000]),
            ),
            ("experience".to_string(), Series::from(vec![3, 5, 7, 4, 6])),
        ]);

        let grouped = df.groupby("department");

        // Test count
        let counts = grouped.count();
        assert_eq!(counts.len(), 3); // 3 unique departments

        // Test sum
        let sums = grouped.sum();
        assert_eq!(sums.columns.len(), 3); // department + 2 numeric columns

        // Test mean
        let means = grouped.mean();
        assert_eq!(means.columns.len(), 3);

        // Test first/last
        let first = grouped.first();
        assert_eq!(first.len(), 3);

        let last = grouped.last();
        assert_eq!(last.len(), 3);
    }

    #[test]
    fn test_joins() {
        let left = DataFrame::new(vec![
            ("id".to_string(), Series::from(vec!["1", "2", "3"])),
            (
                "name".to_string(),
                Series::from(vec!["Alice", "Bob", "Charlie"]),
            ),
        ]);

        let right = DataFrame::new(vec![
            ("id".to_string(), Series::from(vec!["1", "2", "4"])),
            ("score".to_string(), Series::from(vec!["85", "92", "78"])),
        ]);

        // Inner join should match on ids 1 and 2
        let joined = left.join(&right, "id", JoinType::Inner);
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.columns.len(), 3); // id, name, score
    }

    #[test]
    fn test_csv_io_with_inference() -> Result<(), Box<dyn std::error::Error>> {
        // Create temporary CSV with mixed types
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "name,age,salary,active")?;
        writeln!(temp_file, "Alice,25,50000.5,true")?;
        writeln!(temp_file, "Bob,30,60000.0,false")?;
        writeln!(temp_file, "Charlie,35,70000.25,true")?;

        let df = DataFrame::from_csv(temp_file.path().to_str().unwrap())?;

        // Check shape
        assert_eq!(df.shape(), (3, 4));

        // Check type inference
        match df.get_column("age") {
            Some(Series::Int64(_)) => {} // Expected
            _ => panic!("Age should be inferred as Int64"),
        }

        match df.get_column("salary") {
            Some(Series::Float64(_)) => {} // Expected
            _ => panic!("Salary should be inferred as Float64"),
        }

        match df.get_column("active") {
            Some(Series::Bool(_)) => {} // Expected
            _ => panic!("Active should be inferred as Bool"),
        }

        // Test writing back to CSV
        let output_file = NamedTempFile::new()?;
        df.to_csv(output_file.path().to_str().unwrap())?;

        // Read it back
        let df2 = DataFrame::from_csv(output_file.path().to_str().unwrap())?;
        assert_eq!(df2.shape(), df.shape());

        Ok(())
    }

    #[test]
    fn test_json_io() -> Result<(), Box<dyn std::error::Error>> {
        let df = DataFrame::new(vec![
            ("name".to_string(), Series::from(vec!["Alice", "Bob"])),
            ("age".to_string(), Series::from(vec![25, 30])),
            ("active".to_string(), Series::from(vec![true, false])),
        ]);

        // Test JSON Lines format
        let jsonl_file = NamedTempFile::new()?;
        df.to_jsonl(jsonl_file.path().to_str().unwrap())?;

        let df_from_jsonl = DataFrame::from_jsonl(jsonl_file.path().to_str().unwrap())?;
        assert_eq!(df_from_jsonl.shape(), (2, 3));

        // Test regular JSON format
        let json_file = NamedTempFile::new()?;
        df.to_json(json_file.path().to_str().unwrap())?;

        Ok(())
    }

    #[test]
    fn test_statistical_summary() {
        let df = DataFrame::new(vec![
            (
                "values".to_string(),
                Series::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ),
            (
                "integers".to_string(),
                Series::from(vec![10, 20, 30, 40, 50]),
            ),
            (
                "text".to_string(),
                Series::from(vec!["a", "b", "c", "d", "e"]),
            ),
        ]);

        let stats = df.describe();

        // Should have statistics for numeric columns only
        assert_eq!(
            stats.columns,
            vec!["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        );

        // Check some basic statistics
        if let Some(Series::Float64(means)) = stats.data.get(1) {
            assert_eq!(means[0], 3.0); // Mean of [1,2,3,4,5]
            assert_eq!(means[1], 30.0); // Mean of [10,20,30,40,50]
        }
    }

    #[test]
    fn test_mathematical_functions() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test element-wise functions
        let exp_arr = arr.exp();
        assert!((exp_arr[&[0, 0][..]] - 1.0_f64.exp()).abs() < 1e-10);

        let ln_arr = arr.ln();
        assert!((ln_arr[(0, 0)] - 1.0_f64.ln()).abs() < 1e-10);

        let sin_arr = arr.sin();
        assert!((sin_arr[(0, 0)] - 1.0_f64.sin()).abs() < 1e-10);

        let sqrt_arr = arr.sqrt();
        assert!((sqrt_arr[(0, 0)] - 1.0).abs() < 1e-10);

        let pow_arr = arr.pow(2.0);
        assert_eq!(pow_arr[(0, 0)], 1.0);
        assert_eq!(pow_arr[(1, 1)], 16.0);
    }
}
