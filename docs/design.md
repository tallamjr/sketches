# Design and architecture

[Back to the README](../README.md)

## Extending HLL++: Sparse Buffer, Variable-Length Encoding, and Hybrid Representation

Beyond the built-in dense and simple sparse sketches, HLL++ can be optimised further:

- **Unsorted Insertion Buffer**: For high-throughput updates, buffer `(index, rank)` pairs in a small `Vec`, and flush into the main map once full.

  ```rust
  struct SparseBuffer {
      p: u8,
      buffer: Vec<(usize, u8)>,      // unsorted bucket updates
      map: BTreeMap<usize, u8>,      // current sparse registers
  }
  impl SparseBuffer {
      fn update<T: Hash>(&mut self, item: &T) {
          let hash = hash64(item);
          let idx = (hash >> (64 - self.p)) as usize;
          let rank = (hash << self.p).leading_zeros().saturating_add(1) as u8;
          self.buffer.push((idx, rank));
          if self.buffer.len() > self.buffer.capacity() {
              self.flush();
          }
      }
      fn flush(&mut self) {
          for (idx, rank) in self.buffer.drain(..) {
              let entry = self.map.entry(idx).or_insert(0);
              if *entry < rank { *entry = rank; }
          }
      }
  }
  ```

- **Variable-Length Encoding**: Compact sparse pairs into `u32` words `(idx<<6)|rank`, delta-sort, then LEB128 encode:

  ```rust
  fn pack(j: usize, r: u8) -> u32 { ((j as u32) << 6) | (r as u32) }
  let mut packed: Vec<u32> = map.iter().map(|(&j,&r)| pack(j,r)).collect();
  packed.sort_unstable();
  let mut bytes = Vec::new();
  let mut prev = 0;
  for v in packed {
      let delta = v.wrapping_sub(prev);
      leb128::write::unsigned(&mut bytes, delta as u128).unwrap();
      prev = v;
  }
  ```

- **Hybrid Sparse-to-Dense Switch**: Start in sparse mode; once `map.len() > m/2`, materialise a dense `Vec<u8>` and switch to `HllPlusPlusSketch` for O(1) updates.
  ```rust
  if sparse.map.len() > (1 << p) / 2 {
      let mut dense = HllPlusPlusSketch::new(p);
      for (&j,&r) in &sparse.map { dense.registers[j] = r; }
      // adopt `dense` for further updates...
  }
  ```

These extensions deliver fast, memory-efficient, and scalable HLL++ sketches across workloads.

## Architecture

The cardinality and set-operation sketches share a small, deliberately simple core:

- **Pluggable hashing.** A single canonical `SketchHasher` (64- and 128-bit) is used in every build, defaulting to [xxh3](https://github.com/Cyan4973/xxHash). This is a deliberate choice, not MurmurHash3, and the serialised format is our own compact little-endian codec, so sketches are not byte-compatible with Apache DataSketches. That is a chosen tradeoff in favour of a uniform internal format. Because absolute throughput is dominated by the hash function, comparing our xxh3 numbers to Apache's MurmurHash3 numbers compares hash functions as much as sketches.
- **Shared codec and serialisation.** All sketches serialise through a shared little-endian codec behind a uniform `Serializable` trait (the long-tail sketches use postcard). Serialisation round-trips are covered by tests.
- **Pure scalar Rust.** No SIMD, no jemalloc, no rayon. The earlier `optimized` feature has been removed. The implementation is plain, portable, scalar Rust.
- **HIP estimators.** HLL and CPC both carry a Historical Inverse Probability (HIP) estimator, which is what brings their accuracy to (or below) the theoretical floor.
