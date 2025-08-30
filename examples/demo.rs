use rustframes::array::Array;
use rustframes::dataframe::{DataFrame, Series};
use rustframes::JoinType;

fn main() {
    println!("=== RustFrames Enhanced Demo ===\n");

    // ==== Enhanced N-Dimensional Arrays ====
    println!("1. N-Dimensional Arrays:");

    // Create 3D array
    let arr_3d = Array::from_vec((0..24).map(|x| x as f64).collect(), vec![2, 3, 4]);
    println!("3D Array shape: {:?}", arr_3d.shape);
    println!("3D Array element at [1,2,3]: {}", arr_3d[&[1, 2, 3][..]]);

    // Array operations
    let arr_1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let arr_2 = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("\nArray 1: {:?}", arr_1.data);
    println!("Array 2: {:?}", arr_2.data);

    // Broadcasting operations
    let sum = &arr_1 + &arr_2;
    println!("Sum: {:?}", sum.data);

    // Scalar operations
    let scaled = &arr_1 * 2.0;
    println!("Scaled by 2: {:?}", scaled.data);

    // Reductions
    println!("Sum of all elements: {}", arr_1.sum());
    println!("Mean: {}", arr_1.mean());
    println!("Max: {}", arr_1.max());
    println!("Min: {}", arr_1.min());

    // Mathematical functions
    let exp_arr = arr_1.exp();
    println!("Exponential: {:?}", exp_arr.data);

    // ==== Enhanced Linear Algebra ====
    println!("\n2. Linear Algebra:");

    let matrix_a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let matrix_b = Array::from_vec(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);

    println!("Matrix A: {:?}", matrix_a.data);
    println!("Matrix B: {:?}", matrix_b.data);

    // Matrix multiplication
    let product = matrix_a.dot(&matrix_b);
    println!("A * B = {:?}", product.data);

    // Matrix properties
    println!("Determinant of A: {}", matrix_a.det());
    println!("Trace of A: {}", matrix_a.trace());
    println!("Norm of A: {}", matrix_a.norm());
    println!("Is A symmetric? {}", matrix_a.is_symmetric(None));

    // Matrix inverse
    if let Some(inv_a) = matrix_a.inv() {
        println!("Inverse of A: {:?}", inv_a.data);
    }

    // QR decomposition
    let (q, r) = matrix_a.qr();
    println!("Q matrix: {:?}", q.data);
    println!("R matrix: {:?}", r.data);

    // ==== Enhanced DataFrames ====
    println!("\n3. Enhanced DataFrames:");

    // Create sample data
    let df = DataFrame::new(vec![
        ("id".to_string(), Series::from(vec![1, 2, 3, 4, 5])),
        (
            "name".to_string(),
            Series::from(vec!["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        ),
        ("age".to_string(), Series::from(vec![25, 30, 35, 28, 32])),
        (
            "score".to_string(),
            Series::from(vec![85.5, 92.0, 78.5, 88.0, 95.5]),
        ),
        (
            "active".to_string(),
            Series::from(vec![true, true, false, true, true]),
        ),
    ]);

    println!("Original DataFrame:");
    println!("Shape: {:?}", df.shape());
    println!("Columns: {:?}", df.columns);

    // DataFrame operations
    println!("\nHead (3 rows):");
    let head = df.head(3);
    for (i, col) in head.columns.iter().enumerate() {
        print!("{}: ", col);
        match &head.data[i] {
            Series::Int64(v) => println!("{:?}", v),
            Series::Float64(v) => println!("{:?}", v),
            Series::Bool(v) => println!("{:?}", v),
            Series::Utf8(v) => println!("{:?}", v),
        }
    }

    // Filtering
    println!("\nFiltering active users:");
    if let Some(Series::Bool(active_mask)) = df.get_column("active") {
        let filtered = df.filter(active_mask);
        println!("Filtered shape: {:?}", filtered.shape());
    }

    // Sorting
    println!("\nSorting by age:");
    let sorted = df.sort_by("age", true);
    if let Some(Series::Int64(ages)) = sorted.get_column("age") {
        println!("Sorted ages: {:?}", ages);
    }

    // Adding new column
    let df_with_bonus = df.with_column(
        "bonus".to_string(),
        Series::from(vec![100.0, 150.0, 75.0, 120.0, 200.0]),
    );
    println!(
        "\nAdded bonus column, new shape: {:?}",
        df_with_bonus.shape()
    );

    // ==== GroupBy Operations ====
    println!("\n4. GroupBy Operations:");

    // Create data for grouping
    let group_df = DataFrame::new(vec![
        (
            "department".to_string(),
            Series::from(vec!["IT", "HR", "IT", "Finance", "HR", "IT"]),
        ),
        (
            "salary".to_string(),
            Series::from(vec![75000, 65000, 80000, 70000, 68000, 85000]),
        ),
        (
            "experience".to_string(),
            Series::from(vec![3, 5, 7, 4, 6, 8]),
        ),
    ]);

    println!("Group DataFrame:");
    println!("Departments: {:?}", group_df.get_column("department"));

    // GroupBy operations
    let grouped = group_df.groupby("department");

    println!("\nGroupBy Count:");
    let count_result = grouped.count();
    println!("Count columns: {:?}", count_result.columns);

    println!("\nGroupBy Mean:");
    let mean_result = grouped.mean();
    println!("Mean columns: {:?}", mean_result.columns);

    println!("\nGroupBy Sum:");
    let sum_result = grouped.sum();
    println!("Sum columns: {:?}", sum_result.columns);

    // ==== Join Operations ====
    println!("\n5. Join Operations:");

    let left_df = DataFrame::new(vec![
        ("id".to_string(), Series::from(vec!["1", "2", "3"])),
        (
            "name".to_string(),
            Series::from(vec!["Alice", "Bob", "Charlie"]),
        ),
    ]);

    let right_df = DataFrame::new(vec![
        ("id".to_string(), Series::from(vec!["1", "2", "4"])),
        ("score".to_string(), Series::from(vec!["85", "92", "78"])),
    ]);

    println!("Left DataFrame columns: {:?}", left_df.columns);
    println!("Right DataFrame columns: {:?}", right_df.columns);

    let joined = left_df.join(&right_df, "id", JoinType::Inner);
    println!("Inner join result columns: {:?}", joined.columns);
    println!("Inner join shape: {:?}", joined.shape());

    // ==== Statistical Summary ====
    println!("\n6. Statistical Summary:");

    let stats_df = DataFrame::new(vec![
        (
            "values".to_string(),
            Series::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ),
        (
            "categories".to_string(),
            Series::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
    ]);

    let description = stats_df.describe();
    println!("Statistical summary:");
    println!("Description columns: {:?}", description.columns);

    // ==== Advanced Array Operations ====
    println!("\n7. Advanced Array Operations:");

    // Reshape operations
    let arr_1d = Array::from_vec((1..13).map(|x| x as f64).collect(), vec![12]);
    println!("1D array: {:?}", arr_1d.shape);

    let reshaped = arr_1d.reshape(vec![3, 4]);
    println!("Reshaped to 3x4: {:?}", reshaped.shape);
    println!("Reshaped data: {:?}", reshaped.data);

    // Transpose
    let transposed = reshaped.transpose();
    println!("Transposed shape: {:?}", transposed.shape);

    // Axis operations
    let sum_axis0 = reshaped.sum_axis(0);
    println!("Sum along axis 0: {:?}", sum_axis0.data);

    let mean_axis1 = reshaped.mean_axis(1);
    println!("Mean along axis 1: {:?}", mean_axis1.data);

    // ==== Broadcasting Examples ====
    println!("\n8. Broadcasting Examples:");

    let arr_2x3 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let arr_1x3 = Array::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3]);

    println!("Array 2x3: {:?}", arr_2x3.data);
    println!("Array 1x3: {:?}", arr_1x3.data);

    if let Some(broadcast_sum) = arr_2x3.add_broadcast(&arr_1x3) {
        println!("Broadcast addition result: {:?}", broadcast_sum.data);
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_enhanced_features() {
        // Test N-dimensional arrays
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr[&[1, 2]], 6.0);

        // Test broadcasting
        let arr1 = Array::from_vec(vec![1.0, 2.0], vec![2]);
        let arr2 = Array::from_vec(vec![3.0, 4.0], vec![2]);
        let sum = arr1.add_broadcast(&arr2).unwrap();
        assert_eq!(sum.data, vec![4.0, 6.0]);

        // Test DataFrame operations
        let df = DataFrame::new(vec![
            ("a".to_string(), Series::from(vec![1, 2, 3])),
            ("b".to_string(), Series::from(vec![4.0, 5.0, 6.0])),
        ]);

        let filtered = df.filter(&[true, false, true]);
        assert_eq!(filtered.len(), 2);

        let sorted = df.sort_by("a", false); // descending
        if let Some(Series::Int64(values)) = sorted.get_column("a") {
            assert_eq!(values, &vec![3, 2, 1]);
        }

        // Test groupby
        let group_df = DataFrame::new(vec![
            ("group".to_string(), Series::from(vec!["A", "B", "A", "B"])),
            ("value".to_string(), Series::from(vec![1, 2, 3, 4])),
        ]);

        let grouped = group_df.groupby("group");
        let counts = grouped.count();
        assert_eq!(counts.len(), 2); // Two groups
    }

    #[test]
    fn test_csv_io() {
        // Create temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,score").unwrap();
        writeln!(temp_file, "Alice,25,85.5").unwrap();
        writeln!(temp_file, "Bob,30,92.0").unwrap();
        writeln!(temp_file, "Charlie,35,78.5").unwrap();

        // Read CSV
        let df = DataFrame::from_csv(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(df.shape(), (3, 3));
        assert_eq!(df.columns, vec!["name", "age", "score"]);

        // Verify type inference worked
        match df.get_column("age").unwrap() {
            Series::Int64(_) => {} // Expected
            _ => panic!("Age should be inferred as Int64"),
        }

        match df.get_column("score").unwrap() {
            Series::Float64(_) => {} // Expected
            _ => panic!("Score should be inferred as Float64"),
        }
    }

    #[test]
    fn test_linear_algebra() {
        let matrix = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test determinant
        let det = matrix.det();
        assert_eq!(det, -2.0);

        // Test trace
        let trace = matrix.trace();
        assert_eq!(trace, 5.0);

        // Test matrix multiplication
        let identity = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let product = matrix.dot(&identity);
        assert_eq!(product.data, matrix.data);

        // Test QR decomposition
        let (q, r) = matrix.qr();
        assert_eq!(q.shape, vec![2, 2]);
        assert_eq!(r.shape, vec![2, 2]);
    }
}
