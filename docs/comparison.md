# Comparison and prior art

[Back to the README](../README.md)

This library stands on a large body of prior work. This page sets out the main alternatives, how `sketches` relates to each, and where it is and is not differentiated, stating plainly where established libraries lead.

## Apache DataSketches (the reference)

[Apache DataSketches](https://datasketches.apache.org) is the dominant library in this space and the one `sketches` benchmarks and cross-checks against. It is an Apache Software Foundation top-level project (Apache-2.0), and it ships the same four sketch families this crate targets: cardinality (CPC, HLL, Theta, Tuple), quantiles (KLL, REQ, classic), frequencies and frequent items, and sampling (reservoir, VarOpt), plus t-digest, count-min, KDE and a Kolmogorov-Smirnov test.

Where it leads `sketches`, decisively:

- Language reach: Java, C++, Python, Rust and Go, versus this crate's single Rust core with Python bindings.
- Byte-compatible interchange: a sketch built in one language serialises and is read, merged and queried in another (same-endianness). This crate uses its own compact little-endian codec and is deliberately not byte-compatible.
- Production track record and ecosystem: deep integrations including Apache Druid, Hive, Pig, Pinot and a PostgreSQL extension ([apache/datasketches-postgresql](https://github.com/apache/datasketches-postgresql)).
- Institutional backing: maintained by the ASF rather than a single author.

Its Python package ([pypi `datasketches`](https://pypi.org/project/datasketches/), source [apache/datasketches-python](https://github.com/apache/datasketches-python)) is a Nanobind binding over the C++ core, so its Python API mirrors C++. This crate, by contrast, exposes a native Rust implementation through PyO3.

How `sketches` differs: a single native-Rust implementation of the whole suite, with PyO3 bindings, default xxh3 hashing, and an accuracy methodology based on multi-trial RMSE against the 1/sqrt(k) floor. The throughput and accuracy benchmarks in [benchmarks.md](benchmarks.md) run directly against `datasketches-cpp` and the Apache Rust port.

## Python: similarity search and filters

- [`datasketch`](https://github.com/ekzhu/datasketch) (ekzhu): a pure-Python similarity-search library (MinHash, b-bit and weighted MinHash, HLL and HLL++, several LSH variants, HNSW). It overlaps with `sketches` only on HLL; its focus is Jaccard similarity and nearest-neighbour search, which `sketches` does not yet cover (MinHash, SimHash and LSH are on the roadmap, see [design.md](design.md)).
- [`pyprobables`](https://github.com/barrust/pyprobables) (barrust): pure-Python filters and counting (Bloom and its counting/scalable/on-disk/rotating variants, Cuckoo and counting Cuckoo, Quotient filter, Count-Min and its mean variants). No Theta, CPC, KLL, t-digest or sampling.
- [`python-hyperloglog`](https://github.com/svpcom/hyperloglog) (svpcom): a single-family pure-Python cardinality library (HyperLogLog and sliding HyperLogLog with HLL++ bias correction).

## Rust crates

- [`probabilistic-collections`](https://crates.io/crates/probabilistic-collections) (jeffrey-xiao): the canonical Rust multi-structure crate (Bloom and partitioned/scalable/stream variants, Cuckoo, Quotient, Count-Min, HyperLogLog, MinHash, SimHash). It is filter and similarity oriented and does not provide Theta, CPC, KLL, t-digest or reservoir sampling.
- [`hyperloglogplus`](https://crates.io/crates/hyperloglogplus): a single-purpose crate implementing only HyperLogLog and HyperLogLog++.
- The Apache Rust port (`datasketches-rust`) is the Rust peer this crate uses as the `apache-rust` benchmark plane.

The differentiator for `sketches` among Rust crates is breadth: it consolidates cardinality, quantiles, frequency, membership, sampling and multi-dimensional sketches into one crate, where the existing crates are either filter-focused or single-family.

## Database-embedded implementations

Cardinality and quantile sketches are also embedded directly in data systems: Redis (HyperLogLog, and Bloom via RedisBloom), ClickHouse (the `uniq*` and quantile-sketch functions), PostgreSQL (`postgresql-hll` and the Apache `datasketches-postgresql` extension), and the DataSketches integrations in Apache Druid and Spark. These are a different product category (engine-embedded rather than a standalone library), and a frontier where `sketches` has no presence.

## Positioning

- Differentiated: the breadth of the four sketch families plus sampling and t-digest in one actively maintained, native-Rust crate, exposed to Python through PyO3, with xxh3 hashing and an interval-reported, multi-trial-RMSE accuracy methodology.
- Where established libraries lead: Apache DataSketches wins on language reach, byte-compatible interchange, ASF backing, production integrations and breadth-with-maturity; `datasketch` leads on similarity search (MinHash and LSH), which this crate does not yet implement.
- Deliberate non-goal: byte compatibility with Apache DataSketches. This crate uses its own compact codec, trading interchange for a uniform internal format.

## References

- Apache DataSketches: https://datasketches.apache.org (ports: [cpp](https://github.com/apache/datasketches-cpp), [python](https://github.com/apache/datasketches-python), rust, [postgresql](https://github.com/apache/datasketches-postgresql))
- datasketch (MinHash, LSH): https://github.com/ekzhu/datasketch
- pyprobables (filters, count-min): https://github.com/barrust/pyprobables
- probabilistic-collections (Rust): https://crates.io/crates/probabilistic-collections
- hyperloglogplus (Rust): https://crates.io/crates/hyperloglogplus
- python-hyperloglog: https://github.com/svpcom/hyperloglog
