#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    columns: Vec<String>,
    data: Vec<Vec<String>>, // naive placeholder
}

impl DataFrame {
    pub fn from_csv(_path: &str) -> Result<Self, &'static str> {
        // Placeholder for now
        Ok(DataFrame {
            columns: vec!["col1".to_string(), "col2".to_string()],
            data: vec![
                vec!["1".to_string(), "a".to_string()],
                vec!["2".to_string(), "b".to_string()],
            ],
        })
    }

    pub fn head(&self, n: usize) -> Vec<Vec<String>> {
        self.data.iter().take(n).cloned().collect()
    }
}