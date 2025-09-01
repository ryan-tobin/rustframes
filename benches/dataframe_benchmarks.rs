use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{thread_rng, Rng};
use rustframes::{DataFrame, Series};

fn bench_dataframe_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_ops");

    // Create test data
    let n_rows = 100_000usize;
    let mut rng = thread_rng();

    let ids: Vec<i64> = (0..n_rows).map(|i| i as i64).collect();
    let categories: Vec<String> = (0..n_rows)
        .map(|_| format!("cat_{}", rng.gen_range(0..10)))
        .collect();
    let value1s: Vec<f64> = (0..n_rows).map(|_| rng.gen::<f64>()).collect();
    let value2s: Vec<f64> = (0..n_rows).map(|_| rng.gen::<f64>()).collect();

    let df = DataFrame::new(vec![
        ("id".to_string(), Series::Int64(ids)),
        ("category".to_string(), Series::Utf8(categories)),
        ("value1".to_string(), Series::Float64(value1s)),
        ("value2".to_string(), Series::Float64(value2s)),
    ]);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_function("groupby_sum", |bench| {
        bench.iter(|| black_box(df.groupby("category").sum()));
    });

    group.bench_function("groupby_mean", |bench| {
        bench.iter(|| black_box(df.groupby("category").mean()));
    });

    group.bench_function("filter_operation", |bench| {
        bench.iter(|| {
            let mask: Vec<bool> = (0..n_rows).map(|i| i % 2 == 0).collect();
            black_box(df.filter(&mask))
        });
    });

    group.bench_function("sort_by_numeric", |bench| {
        bench.iter(|| black_box(df.sort_by("value1", true)));
    });

    group.bench_function("sort_by_categorical", |bench| {
        bench.iter(|| black_box(df.sort_by("category", true)));
    });

    group.finish();
}

fn bench_io_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_operations");

    // Create test CSV data
    let n_rows = 50_000;
    let temp_path = "/tmp/bench_data.csv";

    let df = DataFrame::new(vec![
        ("id".to_string(), Series::Int64((0..n_rows).collect())),
        (
            "name".to_string(),
            Series::Utf8((0..n_rows).map(|i| format!("name_{}", i)).collect()),
        ),
        (
            "score".to_string(),
            Series::Float64((0..n_rows).map(|i| (i as f64) * 0.1).collect()),
        ),
    ]);

    // Write test data
    df.to_csv(temp_path).unwrap();

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_function("csv_read", |bench| {
        bench.iter(|| black_box(DataFrame::from_csv(temp_path).unwrap()));
    });

    group.bench_function("csv_write", |bench| {
        bench.iter(|| black_box(df.to_csv("/tmp/bench_output.csv").unwrap()));
    });

    #[cfg(feature = "arrow")]
    group.bench_function("parquet_read", |bench| {
        // First write to parquet
        df.to_parquet("/tmp/bench_data.parquet").unwrap();

        bench.iter(|| black_box(DataFrame::from_parquet("/tmp/bench_data.parquet").unwrap()));
    });

    #[cfg(feature = "arrow")]
    group.bench_function("parquet_write", |bench| {
        bench.iter(|| black_box(df.to_parquet("/tmp/bench_output.parquet").unwrap()));
    });

    group.finish();
}

criterion_group!(
    dataframe_benches,
    bench_dataframe_operations,
    bench_io_operations
);

// Comparative benchmarks against other libraries
fn bench_vs_numpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_numpy");

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let numpy = py.import("numpy").unwrap();

            for size in [256, 512, 1024].iter() {
                let rust_a = Array::from_vec(
                    (0..size * size).map(|x| x as f64).collect(),
                    vec![*size, *size],
                );
                let rust_b = Array::from_vec(
                    (0..size * size).map(|x| (x * 2) as f64).collect(),
                    vec![*size, *size],
                );

                let np_a = numpy.call_method1("random.rand", (*size, *size)).unwrap();
                let np_b = numpy.call_method1("random.rand", (*size, *size)).unwrap();

                group.bench_with_input(
                    BenchmarkId::new("rustframes_matmul", size),
                    size,
                    |bench, _| {
                        bench.iter(|| black_box(rust_a.dot(&rust_b)));
                    },
                );

                group.bench_with_input(BenchmarkId::new("numpy_matmul", size), size, |bench, _| {
                    bench.iter(|| black_box(numpy.call_method1("matmul", (np_a, np_b)).unwrap()));
                });
            }
        });
    }

    group.finish();
}

criterion_group!(comparison_benches, bench_vs_numpy);

// Memory usage profiling
pub mod memory_profiler {
    use rustframes::Array;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    pub struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ret = System.alloc(layout);
            if !ret.is_null() {
                ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            }
            ret
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
        }
    }

    #[global_allocator]
    static GLOBAL: TrackingAllocator = TrackingAllocator;

    pub fn current_memory_usage() -> usize {
        ALLOCATED.load(Ordering::SeqCst)
    }

    pub fn benchmark_memory_efficiency() {
        println!("=== Memory Efficiency Benchmarks ===");

        let initial_memory = current_memory_usage();

        // Test array creation
        {
            let start_mem = current_memory_usage();
            let arr = Array::zeros(vec![1000, 1000]);
            let peak_mem = current_memory_usage();
            drop(arr);
            let end_mem = current_memory_usage();

            println!("Array creation (1000x1000):");
            println!("  Peak memory: {} MB", (peak_mem - start_mem) / 1_000_000);
            println!(
                "  Memory after drop: {} MB",
                (end_mem - start_mem) / 1_000_000
            );
        }

        // Test broadcasting memory usage
        {
            let start_mem = current_memory_usage();
            let a = Array::from_vec(vec![1.0; 1000], vec![1, 1000]);
            let b = Array::from_vec(vec![2.0; 1000], vec![1000, 1]);
            let peak_before = current_memory_usage();

            let _result = a.add_broadcast(&b).unwrap();
            let peak_after = current_memory_usage();

            println!("Broadcasting (1x1000) + (1000x1):");
            println!(
                "  Input arrays: {} MB",
                (peak_before - start_mem) / 1_000_000
            );
            println!("  With result: {} MB", (peak_after - start_mem) / 1_000_000);
        }

        let final_memory = current_memory_usage();
        println!(
            "Memory leaked: {} bytes",
            final_memory.saturating_sub(initial_memory)
        );
    }
}

// Performance regression testing
pub mod regression_tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    pub struct PerformanceBaseline {
        benchmarks: HashMap<String, Duration>,
    }

    impl PerformanceBaseline {
        pub fn new() -> Self {
            PerformanceBaseline {
                benchmarks: HashMap::new(),
            }
        }

        pub fn record_benchmark(&mut self, name: &str, duration: Duration) {
            self.benchmarks.insert(name.to_string(), duration);
        }

        pub fn check_regression(&self, name: &str, current: Duration, threshold: f64) -> bool {
            if let Some(&baseline) = self.benchmarks.get(name) {
                let ratio = current.as_secs_f64() / baseline.as_secs_f64();
                ratio > threshold
            } else {
                false // No baseline to compare against
            }
        }

        pub fn run_regression_suite(&mut self) -> Vec<String> {
            let mut regressions = Vec::new();

            // Matrix multiplication regression test
            let sizes = [128, 256, 512];
            for size in sizes {
                let a = Array::from_vec(
                    (0..size * size).map(|x| x as f64).collect(),
                    vec![size, size],
                );
                let b = a.clone();

                let start = Instant::now();
                let _result = a.dot(&b);
                let duration = start.elapsed();

                let benchmark_name = format!("matmul_{}", size);
                if self.check_regression(&benchmark_name, duration, 1.2) {
                    regressions.push(format!(
                        "Regression detected in {}: {}ms vs baseline",
                        benchmark_name,
                        duration.as_millis()
                    ));
                }
                self.record_benchmark(&benchmark_name, duration);
            }

            // DataFrame operations regression test
            let n_rows = 50_000;
            let df = DataFrame::new(vec![
                ("id".to_string(), Series::Int64((0..n_rows).collect())),
                (
                    "category".to_string(),
                    Series::Utf8((0..n_rows).map(|i| format!("cat_{}", i % 10)).collect()),
                ),
                (
                    "value".to_string(),
                    Series::Float64((0..n_rows).map(|i| i as f64 * 0.1).collect()),
                ),
            ]);

            let start = Instant::now();
            let _grouped = df.groupby("category").sum();
            let duration = start.elapsed();

            let benchmark_name = "groupby_sum_50k";
            if self.check_regression(benchmark_name, duration, 1.2) {
                regressions.push(format!(
                    "Regression detected in {}: {}ms",
                    benchmark_name,
                    duration.as_millis()
                ));
            }
            self.record_benchmark(benchmark_name, duration);

            regressions
        }

        pub fn save_baseline(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            let json = serde_json::to_string_pretty(&self.benchmarks)?;
            std::fs::write(path, json)?;
            Ok(())
        }

        pub fn load_baseline(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            let content = std::fs::read_to_string(path)?;
            let benchmarks: HashMap<String, Duration> = serde_json::from_str(&content)?;
            self.benchmarks = benchmarks;
            Ok(())
        }
    }
}

// Cross-platform benchmark runner
pub struct BenchmarkRunner {
    pub system_info: SystemInfo,
    pub results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub memory_gb: u64,
    pub gpu: Option<String>,
    pub rust_version: String,
    pub rustframes_version: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration: Duration,
    pub throughput: Option<f64>, // operations per second
    pub memory_used: Option<usize>,
    pub cpu_usage: Option<f32>,
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        BenchmarkRunner {
            system_info: Self::gather_system_info(),
            results: Vec::new(),
        }
    }

    fn gather_system_info() -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu: Self::get_cpu_info(),
            memory_gb: Self::get_memory_info(),
            gpu: Self::get_gpu_info(),
            rust_version: env!("RUST_VERSION").to_string(),
            rustframes_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn get_cpu_info() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                "x86_64 with AVX2".to_string()
            } else if is_x86_feature_detected!("sse4.1") {
                "x86_64 with SSE4.1".to_string()
            } else {
                "x86_64".to_string()
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            "ARM64".to_string()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Unknown".to_string()
        }
    }

    fn get_memory_info() -> u64 {
        // Simplified - in practice would use system APIs
        8 // GB, placeholder
    }

    fn get_gpu_info() -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            if cudarc::driver::CudaDevice::new(0).is_ok() {
                return Some("NVIDIA CUDA".to_string());
            }
        }
        #[cfg(feature = "rocm")]
        {
            return Some("AMD ROCm".to_string());
        }
        #[cfg(feature = "metal")]
        {
            if metal::Device::system_default().is_some() {
                return Some("Apple Metal".to_string());
            }
        }
        None
    }

    pub fn run_comprehensive_suite(&mut self) {
        println!("Running comprehensive benchmark suite...");
        println!("System: {:?}", self.system_info);

        // Array benchmarks
        self.benchmark_array_operations();

        // DataFrame benchmarks
        self.benchmark_dataframe_operations();

        // Memory benchmarks
        self.benchmark_memory_operations();

        // I/O benchmarks
        self.benchmark_io_operations();

        self.print_summary();
    }

    fn benchmark_array_operations(&mut self) {
        println!("Benchmarking array operations...");

        let sizes = [256, 512, 1024];
        for size in sizes {
            let a = Array::from_vec(
                (0..size * size).map(|x| x as f64).collect(),
                vec![size, size],
            );
            let b = a.clone();

            let start = Instant::now();
            let _result = a.dot(&b);
            let duration = start.elapsed();

            self.results.push(BenchmarkResult {
                name: format!("matmul_{}", size),
                duration,
                throughput: Some((size * size * size) as f64 / duration.as_secs_f64()),
                memory_used: Some(size * size * 8 * 3), // Rough estimate
                cpu_usage: None,
            });
        }
    }

    fn benchmark_dataframe_operations(&mut self) {
        println!("Benchmarking DataFrame operations...");

        let n_rows = 100_000;
        let df = DataFrame::new(vec![
            ("id".to_string(), Series::Int64((0..n_rows).collect())),
            (
                "category".to_string(),
                Series::Utf8((0..n_rows).map(|i| format!("cat_{}", i % 100)).collect()),
            ),
            (
                "value".to_string(),
                Series::Float64((0..n_rows).map(|i| i as f64 * 0.1).collect()),
            ),
        ]);

        let start = Instant::now();
        let _grouped = df.groupby("category").sum();
        let duration = start.elapsed();

        self.results.push(BenchmarkResult {
            name: "groupby_sum_100k".to_string(),
            duration,
            throughput: Some(n_rows as f64 / duration.as_secs_f64()),
            memory_used: None,
            cpu_usage: None,
        });
    }

    fn print_summary(&self) {
        println!("\n=== Benchmark Results Summary ===");
        for result in &self.results {
            println!(
                "{}: {:.2}ms (throughput: {:.0} ops/sec)",
                result.name,
                result.duration.as_millis(),
                result.throughput.unwrap_or(0.0)
            );
        }
    }

    pub fn export_results(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

criterion_main!(array_benches, dataframe_benches, comparison_benches);
