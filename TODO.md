# Sketches: High-Performance Probabilistic Data Structures

The `sketches` repo is to be a high-performant implementation of various
probabilistic data structures.

In terms of features it should implement as part of its library all those
outlined here: https://github.com/gakhov/pdsa

Furthermore, it would be great if it also ensure to implement those found here:
https://github.com/apache/datasketches-postgresql/blob/master/README.md

In terms of the repository layout, I have moved towards preferring the structure
found at ../evlib better where we have a single crate that has all functionality
implemented in src and the bindings defined in lib.rs. Like evlib there should
be an examples folder that is able to showcase the functionality and notebooks
users can play with.

All implementation should be verified with `cargo check` and `cargo test` as well as
`pytest` for python code and `pytest --nbmake` to verify notebooks.

## Implementation Roadmap

### Phase 1: Foundation & Core Structures (Current Sprint)

#### 1.1 Repository Restructuring

- [ ] Align repository structure with `evlib` pattern
  - [ ] Create `examples/` directory with Rust examples
  - [ ] Create `notebooks/` directory with Jupyter notebooks
  - [ ] Update module organisation in `src/`
  - [ ] Ensure all bindings are centralised in `lib.rs`

#### 1.2 Complete In-Progress Implementations

- [ ] **CPC (Compressed Probabilistic Counting)**
  - Implement proper CPC algorithm (not HLL delegation)
  - Add compression techniques
  - SIMD optimisation for hash operations
  - Python bindings and tests

### Phase 2: Essential Membership & Frequency Structures

#### 2.1 Bloom Filter Family

- [ ] **Standard Bloom Filter**
  - SIMD-optimised bit operations
  - Multiple hash function strategies
  - Optimal parameter calculation
- [ ] **Counting Bloom Filter**
  - Counter overflow handling
  - Memory-efficient counter storage

#### 2.2 Frequency Estimation

- [ ] **Count-Min Sketch**
  - SIMD parallel updates
  - Conservative update variant
  - Heavy hitters detection
- [ ] **Count Sketch**
  - Median estimation
  - Range query support

### Phase 3: Quantiles & Advanced Structures

#### 3.1 Distribution Sketches

- [ ] **KLL Sketch** (K-Minimum Values)
  - Quantile queries
  - Rank queries
  - Merge operations
- [ ] **q-digest**
  - Compressed quantile summaries
  - Range sum queries

#### 3.2 Cardinality Estimation

- [ ] **Linear Counter**
  - Small cardinality optimisation
  - Transition to HLL for large sets
- [ ] **Probabilistic Counter** (Flajolet–Martin)
  - Historical implementation
  - Educational examples

### Phase 4: Specialised Structures

#### 4.1 Apache DataSketches Compatibility

- [ ] **Frequent Strings Sketch**
  - String-specific optimisations
  - Error bounds guarantees
- [ ] **Array of Doubles (AOD) Sketch**
  - Tuple sketch implementation
  - Aggregation operations

#### 4.2 Sampling

- [ ] **Reservoir Sampling**
  - Weighted and unweighted variants
  - Distributed sampling support

### Phase 5: Performance & Advanced Features

#### 5.1 Hardware Acceleration

- [ ] **SIMD Optimisations**
  - AVX2/AVX-512 for x86_64
  - NEON for ARM
  - Portable SIMD abstraction layer
- [ ] **GPU Acceleration** (Optional)
  - Metal compute shaders for macOS
  - CUDA kernels for NVIDIA GPUs
  - Batch operations interface

#### 5.2 Integration & Ecosystem

- [ ] **Polars Integration**
  - Custom expressions
  - Lazy evaluation support
- [ ] **Distributed Computing**
  - Serialisation formats
  - Merge protocols
  - Network-efficient updates

## Technical Requirements

### Performance Goals

- All core operations should leverage SIMD where beneficial
- Memory layout optimised for cache efficiency
- Zero-copy serialisation where possible
- Benchmarks against reference implementations

### API Design

- Consistent API across all sketch types
- Builder pattern for configuration
- Streaming and batch interfaces
- Type-safe error handling

### Testing Strategy

- Property-based testing for correctness
- Benchmarks for performance regression
- Integration tests with real datasets
- Notebook examples as documentation

### Documentation

- Comprehensive rustdoc for all public APIs
- Theory & implementation notes
- Performance characteristics
- Use case examples

## Priority Order

1. Complete CPC implementation
2. Bloom Filter (most widely used)
3. Count-Min Sketch (essential for frequency)
4. KLL Sketch (modern quantiles solution)
5. Repository restructuring throughout

Please refer to the @TODO.md file to develop a plan for implementing the
features described therein. You should think hard about what a roadmap should
look like and then when that is in place you should update the TODO.md with an
outline of this plan.

When you are ready to tackle this please go ahead and begin the restructuring of
the codebase and implementation of the functionality. You should commit your
changes often and make sure a good message is given.

Where possible all implementation should leverage SIMD and if at all possible
have an option to leverage GPU acceleration via CUDA or ideally Metal for macOS.
