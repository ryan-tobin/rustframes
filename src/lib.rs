//! RustFrames - core library

pub mod array;
pub mod dataframe;

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::dataframe::DataFrame;
    use crate::dataframe::Series;

    #[test]
    fn array_basic_ops() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let b = Array::<f64>::ones((2, 2));
        let c = &a + &b;
        assert_eq!(c[(0, 0)], 2.0);
    }

    #[test]
    fn dataframe_head_select() {
        let df = DataFrame::new(vec![
            ("x".to_string(), Series::from(vec![1, 2, 3])),
            ("y".to_string(), Series::from(vec!["a", "b", "c"])),
        ]);
        let head = df.head(2);
        assert_eq!(head[0].1.len(), 2);
    }
}
