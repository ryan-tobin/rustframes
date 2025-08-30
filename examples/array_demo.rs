use rustframes::array::Array;

fn main() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = Array::<f64>::ones((2, 2));
    let c = &a + &b;

    println!("Array a: {:?}", a);
    println!("Array b: {:?}", b);
    println!("a + b = {:?}", c);

    let d = a.dot(&b);
    println!("a.dot(b) = {:?}", d);
}
