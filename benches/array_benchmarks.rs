use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rustframes::Array;

fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    for size in [64, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements((*size * *size * *size) as u64));

        let a = Array::from_vec(
            (0..size * size).map(|x| x as f64).collect(),
            vec![*size, *size],
        );
        let b = Array::from_vec(
            (0..size * size).map(|x| (x * 2) as f64).collect(),
            vec![*size, *size],
        );

        group.bench_with_input(BenchmarkId::new("naive", size), size, |bench, _| {
            bench.iter(|| black_box(a.dot(&b)));
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("blocked", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul_blocked(&b, 64)));
        });

        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul_parallel(&b)));
        });

        #[cfg(feature = "gpu")]
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul_gpu(&b).unwrap()));
        });
    }

    group.finish();
}

fn bench_broadcasting_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    for size in [1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = Array::from_vec((0..*size).map(|x| x as f64).collect(), vec![*size]);
        let b = Array::from_vec(vec![2.0], vec![1]);

        group.bench_with_input(
            BenchmarkId::new("element_wise_add", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(a.add_broadcast(&b).unwrap()));
            },
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd_add", size), size, |bench, _| {
            bench.iter(|| black_box(a.add_simd(&b)));
        });

        group.bench_with_input(BenchmarkId::new("scalar_add", size), size, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(2.0)));
        });
    }

    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    for size in [1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = Array::from_vec(
            (0..*size).map(|_| thread_rng().gen::<f64>()).collect(),
            vec![*size],
        );

        group.bench_with_input(BenchmarkId::new("sum", size), size, |bench, _| {
            bench.iter(|| black_box(a.sum()));
        });

        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| black_box(a.mean()));
        });

        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel_sum", size), size, |bench, _| {
            bench.iter(|| black_box(a.reduce(0.0, |x| x, |a, b| a + b)));
        });
    }

    group.finish();
}

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    let size = 1024;
    let a = Array::from_vec(
        (0..size * size).map(|x| x as f64).collect(),
        vec![size, size],
    );

    group.bench_function("row_major_access", |bench| {
        bench.iter(|| {
            let mut sum = 0.0;
            for i in 0..size {
                for j in 0..size {
                    sum += a[(i, j)];
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("col_major_access", |bench| {
        bench.iter(|| {
            let mut sum = 0.0;
            for j in 0..size {
                for i in 0..size {
                    sum += a[(i, j)];
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("transpose", |bench| {
        bench.iter(|| black_box(a.transpose()));
    });

    group.finish();
}

criterion_group!(
    array_benches,
    bench_matrix_multiplication,
    bench_broadcasting_operations,
    bench_reductions,
    bench_memory_patterns
);
