<div align="center">
  <img src="logo/RustFrames.svg" alt="RustFrames" width="280">
</div>

# RustFrames

*A blazing fast, memory-safe alternative to NumPy + Pandas, written in Rust*

[![CI](https://github.com/ryan-tobin/rustframes/actions/workflows/ci.yml/badge.svg)](https://github.com/ryan-tobin/rustframes/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/rustframes.svg)](https://crates.io/crates/rustframes)
[![Docs.rs](https://docs.rs/rustframes/badge.svg)](https://docs.rs/rustframes)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Why RustFrames?
Rustframes aims to become the **foundational data & array library for Rust**, much like NumPy and Pandas are for Python.

- ***Rust Safety** -> No segfaults, no undefined behavior
- **Performance-first** -> SIMD, multithreading, and GPU acceleration
- **Interop** -> Apache Arrow, Parquet, CSV, NumPy arrays.
- **DataFrames + Arrays** -> One unifed library for both tabular and n-dimensional data.

## Features (WIP)
* [x] - N-dimensional arrays (`Array`)
* [x] - Basic arithmetic & broadcasting
* [x] - DataFrame & Series abstractions
* [x] - CSV I/O
* [ ] - GroupBy & joins
* [ ] - Arrow backend
* [ ] - GPU acceleration (CUDA/ROCm)

## Installation
Add to your `Cargo.toml`:
```toml
[dependencies]
rustframes = "0.1"
```

## Quick Start
### Arrays
```rust
use rustframes::array::Array;

fn main() {
    let arr = Array::from_vec(vec![1.0,2.0,3.0], (3,));
    println!("{:?}", arr + 2.0); // Broadcasting
}
```

### DataFrames
```rust
use rustframes::dataframe::DataFrame;

fn main() {
    let df = DataFrame::from_csv("data.csv").unwrap();
    println("{:?}", df.head(5));
}
```

## Roadmap
* [ ] Phase 1: Core Array Library
* [ ] Phase 2: DataFrame Layer
* [ ] Phase 3: Arrow backend + optimizations
* [ ] Phase 4: GPU support
* [ ] Phase 5: Python bindings + ecosystem adaption

## Contributing
Contributions welcome! Please check out our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
Licensed under either of:
* Apache License, Version 2.0
* MIT license

