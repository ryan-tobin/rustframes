#[derive(Debug, Clone, PartialEq)]
pub struct Array<T> {
    data: Vec<T>,
    shape: (usize, ), // placeholder
}

impl<T> Array<T> {
    pub fn from_vec(data: Vec<T>, shape: (usize, )) -> Self {
        assert_eq!(data.len(), shape.0, "Shape does not match data length");
        Array { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}