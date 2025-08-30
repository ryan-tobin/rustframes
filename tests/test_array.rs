use rustframes::array::Array;
use rustframes::dataframe::{DataFrame, Series};

#[test]
fn array_addition() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = Array::<f64>::ones((2, 2));
    let c = &a + &b;

    assert_eq!(c[(0, 0)], 2.0);
    assert_eq!(c[(1, 1)], 5.0);
}

#[test]
fn array_dot() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = Array::<f64>::ones((2, 2));
    let d = a.dot(&b);

    assert_eq!(d[(0, 0)], 3.0);
    assert_eq!(d[(1, 1)], 7.0);
}

#[test]
fn dataframe_head_select() {
    let df = DataFrame::new(vec![
        ("x".to_string(), Series::from(vec![1, 2, 3])),
        ("y".to_string(), Series::from(vec!["a", "b", "c"])),
    ]);

    let head = df.head(2);
    assert_eq!(head[0].1.len(), 2);
    assert_eq!(head[1].1.len(), 2);

    let selected = df.select(&["y"]);
    assert_eq!(selected.columns, vec!["y".to_string()]);
    assert_eq!(selected.data.len(), 1);
}

#[test]
fn dataframe_groupby_count() {
    let df = DataFrame::new(vec![(
        "group".to_string(),
        Series::from(vec!["a", "b", "a", "c", "b"]),
    )]);

    let grouped = df.groupby_count("group");
    if let Series::Utf8(labels) = &grouped.data[0] {
        assert!(labels.contains(&"a".to_string()));
        assert!(labels.contains(&"b".to_string()));
        assert!(labels.contains(&"c".to_string()));
    }
    if let Series::Int64(counts) = &grouped.data[1] {
        assert_eq!(counts.iter().sum::<i64>(), 5);
    }
}
