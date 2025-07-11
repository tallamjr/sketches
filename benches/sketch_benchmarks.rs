use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use sketches::{
    aod::AodSketch, bloom::BloomFilter, countmin::CountMinSketch, cpc::CpcSketch, hll::HllSketch,
    quantiles::KllSketch, sampling::ReservoirSamplerA, tdigest::TDigest, theta::ThetaSketch,
};

fn generate_test_strings(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("item_{:08}", i)).collect()
}

fn generate_test_numbers(count: usize) -> Vec<f64> {
    (0..count).map(|i| i as f64 * 1.7 + 0.3).collect()
}

fn bench_hll_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("hll_updates");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("hll_basic", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = HllSketch::new(12);
                for item in &data {
                    sketch.update(black_box(item));
                }
                black_box(sketch.estimate())
            })
        });

        group.bench_with_input(BenchmarkId::new("hll_plus_plus", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = sketches::hll::HllPlusPlusSketch::new(12);
                for item in &data {
                    sketch.update(black_box(item));
                }
                black_box(sketch.estimate())
            })
        });
    }
    group.finish();
}

fn bench_theta_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("theta_updates");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("theta", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = ThetaSketch::new(4096);
                for item in &data {
                    sketch.update(black_box(item));
                }
                black_box(sketch.estimate())
            })
        });
    }
    group.finish();
}

fn bench_cpc_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpc_updates");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("cpc", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = CpcSketch::new(12);
                for item in &data {
                    sketch.update(black_box(item));
                }
                black_box(sketch.estimate())
            })
        });
    }
    group.finish();
}

fn bench_bloom_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("bloom_updates", size), size, |b, _| {
            b.iter(|| {
                let mut filter = BloomFilter::new(*size, 0.01, false);
                for item in &data {
                    filter.add(black_box(item));
                }
                // Test some lookups
                for i in 0..100 {
                    black_box(filter.contains(&format!("item_{:08}", i)));
                }
            })
        });
    }
    group.finish();
}

fn bench_count_min_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_min_sketch");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("cms_updates", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = CountMinSketch::new(1000, 5, false, false);
                for item in &data {
                    sketch.increment(black_box(item));
                }
                // Test some queries
                for i in 0..100 {
                    black_box(sketch.estimate(&format!("item_{:08}", i)));
                }
            })
        });
    }
    group.finish();
}

fn bench_kll_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kll_sketch");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_numbers(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("kll_updates", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = KllSketch::new(200);
                for &item in &data {
                    sketch.update(black_box(item));
                }
                black_box(sketch.quantile(0.5));
                black_box(sketch.quantile(0.95));
                black_box(sketch.quantile(0.99))
            })
        });
    }
    group.finish();
}

fn bench_reservoir_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("reservoir_sampling");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("reservoir_a", size), size, |b, _| {
            b.iter(|| {
                let mut sampler = ReservoirSamplerA::new(1000);
                for item in &data {
                    sampler.add(black_box(item.clone()));
                }
                black_box(sampler.sample().len())
            })
        });
    }
    group.finish();
}

fn bench_tdigest(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdigest");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_numbers(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("tdigest_updates", size), size, |b, _| {
            b.iter(|| {
                let mut digest = TDigest::new();
                for &item in &data {
                    digest.add(black_box(item));
                }
                black_box(digest.quantile(0.5));
                black_box(digest.quantile(0.95));
                black_box(digest.quantile(0.99))
            })
        });
    }
    group.finish();
}

fn bench_aod_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("aod_sketch");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_test_strings(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("aod_updates", size), size, |b, _| {
            b.iter(|| {
                let mut sketch = AodSketch::with_capacity_and_values(4096, 3);
                for (i, item) in data.iter().enumerate() {
                    let values = [i as f64, (i * 2) as f64, (i * 3) as f64];
                    sketch.update(black_box(item), black_box(&values)).unwrap();
                }
                black_box(sketch.estimate());
                black_box(sketch.column_means())
            })
        });
    }
    group.finish();
}

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");

    let data1 = generate_test_strings(50_000);
    let data2 = generate_test_strings(50_000);

    // Pre-populate sketches
    let mut theta1 = ThetaSketch::new(4096);
    let mut theta2 = ThetaSketch::new(4096);

    for item in &data1 {
        theta1.update(item);
    }
    for item in &data2 {
        theta2.update(item);
    }

    group.bench_function("theta_union", |b| {
        b.iter(|| {
            let result = ThetaSketch::union(black_box(&theta1), black_box(&theta2));
            black_box(result.estimate())
        })
    });

    group.bench_function("theta_intersection", |b| {
        b.iter(|| {
            let result = ThetaSketch::intersect(black_box(&theta1), black_box(&theta2));
            black_box(result.estimate())
        })
    });

    group.bench_function("theta_difference", |b| {
        b.iter(|| {
            let result = ThetaSketch::difference(black_box(&theta1), black_box(&theta2), 4096);
            black_box(result.estimate())
        })
    });

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    // Prepare sketches with data
    let data = generate_test_strings(100_000);

    let mut hll = HllSketch::new(12);
    let mut theta = ThetaSketch::new(4096);

    for item in &data {
        hll.update(item);
        theta.update(item);
    }

    group.bench_function("hll_serialize", |b| b.iter(|| black_box(hll.to_bytes())));

    // Note: HLL deserialization not implemented yet
    // let hll_bytes = hll.to_bytes();
    // group.bench_function("hll_deserialize", |b| {
    //     b.iter(|| {
    //         black_box(HllSketch::from_bytes(black_box(&hll_bytes)))
    //     })
    // });

    group.finish();
}

criterion_group!(
    benches,
    bench_hll_updates,
    bench_theta_updates,
    bench_cpc_updates,
    bench_bloom_filter,
    bench_count_min_sketch,
    bench_kll_sketch,
    bench_reservoir_sampling,
    bench_tdigest,
    bench_aod_sketch,
    bench_set_operations,
    bench_serialization
);

criterion_main!(benches);
