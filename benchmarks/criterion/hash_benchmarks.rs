use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sketches::hash::murmur3::Murmur3Hasher;
use sketches::hash::xxh3::Xxh3Hasher;
use sketches::hash::{DEFAULT_SEED, SketchHasher};

fn bench_single_hash(c: &mut Criterion) {
    let key8 = 0x0123_4567_89ab_cdefu64.to_le_bytes();
    let key64: Vec<u8> = (0..64u8).collect();
    let xx = Xxh3Hasher;
    let mm = Murmur3Hasher;

    let mut g = c.benchmark_group("single_hash64");
    g.bench_function("xxh3/8B", |b| {
        b.iter(|| black_box(xx.hash64(black_box(&key8), DEFAULT_SEED)))
    });
    g.bench_function("murmur3/8B", |b| {
        b.iter(|| black_box(mm.hash64(black_box(&key8), DEFAULT_SEED)))
    });
    g.bench_function("xxh3/64B", |b| {
        b.iter(|| black_box(xx.hash64(black_box(&key64), DEFAULT_SEED)))
    });
    g.bench_function("murmur3/64B", |b| {
        b.iter(|| black_box(mm.hash64(black_box(&key64), DEFAULT_SEED)))
    });
    g.finish();
}

criterion_group!(benches, bench_single_hash);
criterion_main!(benches);
