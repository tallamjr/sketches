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

### Phase 1: Foundation & Core Structures ✅ **COMPLETED**

#### 1.1 Repository Restructuring ✅

- [x] Align repository structure with `evlib` pattern
  - [x] Create `examples/` directory with Rust examples
  - [x] Create `notebooks/` directory with Jupyter notebooks
  - [x] Update module organisation in `src/`
  - [x] Ensure all bindings are centralised in `lib.rs`

#### 1.2 Complete In-Progress Implementations ✅

- [x] **CPC (Compressed Probabilistic Counting)**
  - [x] Implement proper CPC algorithm (not HLL delegation)
  - [x] Add compression techniques (RLE for table mode)
  - [x] SIMD optimisation framework (scalars with SIMD-ready structure)
  - [x] Python bindings and tests

### Phase 2: Essential Membership & Frequency Structures ✅ **COMPLETED**

#### 2.1 Bloom Filter Family ✅

- [x] **Standard Bloom Filter**
  - [x] SIMD-optimised bit operations (framework ready)
  - [x] Multiple hash function strategies (double hashing)
  - [x] Optimal parameter calculation
- [x] **Counting Bloom Filter**
  - [x] Counter overflow handling (saturating arithmetic)
  - [x] Memory-efficient counter storage (u8 counters)

#### 2.2 Frequency Estimation ✅

- [x] **Count-Min Sketch**
  - [x] SIMD parallel updates (framework ready)
  - [x] Conservative update variant
  - [x] Heavy hitters detection
- [x] **Count Sketch**
  - [x] Median estimation
  - [x] Signed counter support for better accuracy

### Phase 3: Quantiles & Advanced Structures 🚧 **IN PROGRESS**

#### 3.1 Distribution Sketches

- [ ] **KLL Sketch** (K-Minimum Values) - **NEXT PRIORITY**
  - [ ] Quantile queries (0.5, 0.95, 0.99, etc.)
  - [ ] Rank queries (position of value in sorted order)
  - [ ] Merge operations for distributed computing
  - [ ] Optimal compaction policies
- [ ] **q-digest**
  - [ ] Compressed quantile summaries
  - [ ] Range sum queries
  - [ ] Tree-based compression

#### 3.2 Cardinality Estimation

- [ ] **Linear Counter**
  - [ ] Small cardinality optimisation (better than HLL for n < 1000)
  - [ ] Automatic transition to HLL for large sets
  - [ ] Cache-friendly bit array implementation
- [ ] **Probabilistic Counter** (Flajolet–Martin)
  - [ ] Historical implementation for educational purposes
  - [ ] Bit-pattern analysis
  - [ ] Geometric mean estimation

### Phase 4: Specialised Structures ⏳ **PENDING**

#### 4.1 Apache DataSketches Compatibility

- [ ] **Frequent Strings Sketch**
  - [ ] String-specific optimisations
  - [ ] Error bounds guarantees (ε-δ parameters)
  - [ ] Space-efficient string storage
- [ ] **Array of Doubles (AOD) Sketch**
  - [ ] Tuple sketch implementation
  - [ ] Aggregation operations (sum, mean, etc.)
  - [ ] Multi-dimensional data support

#### 4.2 Sampling

- [ ] **Reservoir Sampling**
  - [ ] Weighted and unweighted variants
  - [ ] Algorithm R and Algorithm A implementations
  - [ ] Distributed sampling support
  - [ ] Stream processing optimizations

### Phase 5: Performance & Advanced Features ⏳ **FRAMEWORK READY**

#### 5.1 Hardware Acceleration

- [ ] **SIMD Optimisations** 🔧 **FRAMEWORK IMPLEMENTED**
  - [ ] AVX2/AVX-512 for x86_64 (actual implementations)
  - [ ] NEON for ARM64 (actual implementations) 
  - [x] Portable SIMD abstraction layer (structure ready)
  - [ ] Batch processing APIs
- [ ] **GPU Acceleration** (Optional)
  - [ ] Metal compute shaders for macOS
  - [ ] CUDA kernels for NVIDIA GPUs
  - [ ] Batch operations interface
  - [ ] Memory transfer optimization

#### 5.2 Integration & Ecosystem

- [ ] **Polars Integration**
  - [ ] Custom expressions for sketches
  - [ ] Lazy evaluation support
  - [ ] DataFrame sketch operations
- [ ] **Distributed Computing**
  - [x] Basic serialisation formats (implemented)
  - [ ] Advanced merge protocols
  - [ ] Network-efficient updates
  - [ ] Apache Arrow integration

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

## Current Implementation Status (December 2024)

### ✅ **Completed (Phases 1-2)**
- **Repository Structure**: Full evlib-style layout with examples/ and notebooks/
- **Core Cardinality**: HyperLogLog (3 variants), CPC, Theta Sketch
- **Membership Testing**: Standard & Counting Bloom Filters  
- **Frequency Estimation**: Count-Min Sketch, Count Sketch
- **Python Bindings**: Complete PyO3 integration for all structures
- **SIMD Framework**: Ready for acceleration (scalar fallbacks implemented)

### 🚧 **Current Priority Order**

1. **KLL Sketch** (modern quantiles - next implementation)
2. **Linear Counter** (small cardinality optimization) 
3. **Frequent Strings Sketch** (Apache DataSketches compatibility)
4. **Reservoir Sampling** (sampling algorithms)
5. **q-digest** (range sum queries)
6. **Actual SIMD Implementation** (replace scalar fallbacks)
7. **GPU Acceleration** (Metal/CUDA kernels)

### 📊 **Scope Achievement**
- **pdsa compatibility**: ~70% (missing quantiles, sampling)
- **Apache DataSketches**: ~60% (missing specialised structures)
- **Production Ready**: Core cardinality and frequency estimation ✅

Please refer to the @TODO.md file to develop a plan for implementing the
features described therein. You should think hard about what a roadmap should
look like and then when that is in place you should update the TODO.md with an
outline of this plan.

When you are ready to tackle this please go ahead and begin the restructuring of
the codebase and implementation of the functionality. You should commit your
changes often and make sure a good message is given.

Where possible all implementation should leverage SIMD and if at all possible
have an option to leverage GPU acceleration via CUDA or ideally Metal for macOS.
