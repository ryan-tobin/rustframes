use rustframes::array::Array;

#[test]
fn test_array_creation() {
    let arr = Array::from_vec(vec![1, 2, 3], (3,));
    assert_eq!(arr.len(), 3);
}
