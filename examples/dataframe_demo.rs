use rustframes::dataframe::{DataFrame, Series};


fn main() {
    let df = DataFrame::new(vec![
        ("x".to_string(), Series::from(vec![1, 2, 3, 4])),
        ("y".to_string(), Series::from(vec!["a", "b", "c", "d"])),
    ]);


    println!("DataFrame:\n{:?}", df);


    let head = df.head(2);
    println!("Head 2 rows:\n{:?}", head);


    let selected = df.select(&["y"]);
    println!("Selected column 'y':\n{:?}", selected);


    let grouped = df.groupby_count("y");
    println!("Grouped count by 'y':\n{:?}", grouped);
}