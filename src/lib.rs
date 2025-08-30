//! RustFrames
//! A blazing-fast, memory-safe alternative to NumPy + Pandas, written in Rust.
//! 
//! # Quick Start
//! ```
//! use rustframes::array::Array;
//! 
//! let arr = Array:from_vec(vec![1.0,2.0,3.0], (3,));
//! println!("{:?}", arr);
//! ```

pub mod array;
pub mod dataframe;