use rustframes::dataframe::DataFrame;

#[test]
fn test_dataframe_head() {
    let df = DataFrame::from_csv("tests/data/test.csv").unwrap();
    let head = df.head(2);
    assert_eq!(head.data.len(), 2);
}
