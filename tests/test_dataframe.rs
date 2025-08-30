use rustframes::dataframe::DataFrame;

#[test]
fn test_dataframe_head() {
    let df = DataFrame::from_csv("dummy.csv").unwrap();
    let head = df.head(1);
    assert_eq!(head.len(), 1);
}
