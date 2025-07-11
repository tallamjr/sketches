//! Compact memory data structures for maximum memory efficiency.
//!
//! This module provides memory-optimized alternatives to standard Rust collections
//! specifically designed for probabilistic data structures. These implementations
//! focus on minimizing memory overhead and maximizing cache locality.

use std::mem;

/// Bit-packed storage for HyperLogLog registers.
/// Stores multiple small integers in packed format to minimize memory usage.
pub struct PackedRegisters {
    data: Vec<u64>,
    precision: u8,
    register_bits: u8,
}

impl PackedRegisters {
    /// Create a new packed register array for HLL with given precision.
    /// Each register stores `register_bits` bits (typically 6 for HLL).
    pub fn new(precision: u8, register_bits: u8) -> Self {
        assert!(
            register_bits > 0 && register_bits <= 64,
            "register_bits must be between 1 and 64"
        );

        let num_registers = 1 << precision;
        let registers_per_u64 = 64 / register_bits as usize;
        let data_len = (num_registers + registers_per_u64 - 1) / registers_per_u64;

        Self {
            data: vec![0u64; data_len],
            precision,
            register_bits,
        }
    }

    /// Get the value of a register at the given index.
    #[inline(always)]
    pub fn get(&self, index: usize) -> u8 {
        let registers_per_u64 = 64 / self.register_bits as usize;
        let word_index = index / registers_per_u64;
        let bit_index = (index % registers_per_u64) * self.register_bits as usize;
        let mask = if self.register_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.register_bits) - 1
        };

        ((self.data[word_index] >> bit_index) & mask) as u8
    }

    /// Set the value of a register at the given index.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: u8) {
        let registers_per_u64 = 64 / self.register_bits as usize;
        let word_index = index / registers_per_u64;
        let bit_index = (index % registers_per_u64) * self.register_bits as usize;
        let mask = if self.register_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.register_bits) - 1
        };

        // Clear the bits at this position
        self.data[word_index] &= !(mask << bit_index);
        // Set the new value
        self.data[word_index] |= (value as u64 & mask) << bit_index;
    }

    /// Update register with maximum value (branch-free).
    #[inline(always)]
    pub fn update_max(&mut self, index: usize, value: u8) {
        let current = self.get(index);
        // Branch-free maximum using bit manipulation
        let mask = ((value > current) as u8).wrapping_sub(1);
        let new_value = (current & mask) | (value & !mask);
        self.set(index, new_value);
    }

    /// Branch-free batch update for multiple registers.
    #[cfg(feature = "optimized")]
    pub fn update_max_batch(&mut self, updates: &[(usize, u8)]) {
        use rayon::prelude::*;

        // Sort updates by word index for better cache locality
        let mut sorted_updates = updates.to_vec();
        sorted_updates.par_sort_unstable_by_key(|(index, _)| {
            let registers_per_u64 = 64 / self.register_bits as usize;
            index / registers_per_u64
        });

        // Apply updates with prefetching
        for (index, value) in sorted_updates {
            self.update_max(index, value);
        }
    }

    /// Get the number of registers.
    pub fn len(&self) -> usize {
        1 << self.precision
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.data.len() * 8 + mem::size_of::<Self>()
    }

    /// Clear all registers.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Iterator over all register values.
    pub fn iter(&self) -> PackedRegisterIter {
        PackedRegisterIter {
            registers: self,
            index: 0,
        }
    }
}

pub struct PackedRegisterIter<'a> {
    registers: &'a PackedRegisters,
    index: usize,
}

impl<'a> Iterator for PackedRegisterIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.registers.len() {
            let value = self.registers.get(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}

/// Compact hash table optimized for sketch data structures.
/// Uses open addressing with linear probing for better cache locality.
pub struct CompactHashTable<V> {
    buckets: Vec<Option<(u64, V)>>,
    size: usize,
    capacity: usize,
    load_factor_threshold: f64,
}

impl<V: Clone> CompactHashTable<V> {
    /// Create a new compact hash table with initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        let capacity = initial_capacity.next_power_of_two();
        Self {
            buckets: vec![None; capacity],
            size: 0,
            capacity,
            load_factor_threshold: 0.75,
        }
    }

    /// Insert or update a key-value pair.
    pub fn insert(&mut self, key: u64, value: V) -> Option<V> {
        if self.size as f64 / self.capacity as f64 > self.load_factor_threshold {
            self.resize();
        }

        self.insert_internal(key, value)
    }

    fn insert_internal(&mut self, key: u64, value: V) -> Option<V> {
        let mut index = (key as usize) & (self.capacity - 1);

        loop {
            match &mut self.buckets[index] {
                None => {
                    self.buckets[index] = Some((key, value));
                    self.size += 1;
                    return None;
                }
                Some((existing_key, existing_value)) => {
                    if *existing_key == key {
                        return Some(mem::replace(existing_value, value));
                    }
                    index = (index + 1) & (self.capacity - 1);
                }
            }
        }
    }

    /// Get a value by key.
    pub fn get(&self, key: u64) -> Option<&V> {
        let mut index = (key as usize) & (self.capacity - 1);

        loop {
            match &self.buckets[index] {
                None => return None,
                Some((existing_key, value)) => {
                    if *existing_key == key {
                        return Some(value);
                    }
                    index = (index + 1) & (self.capacity - 1);
                }
            }
        }
    }

    /// Get a mutable reference to a value by key.
    pub fn get_mut(&mut self, key: u64) -> Option<&mut V> {
        let mut index = (key as usize) & (self.capacity - 1);
        let start_index = index;

        // First find the correct index
        let found_index = loop {
            match &self.buckets[index] {
                None => return None,
                Some((existing_key, _)) => {
                    if *existing_key == key {
                        break index;
                    }
                }
            }

            index = (index + 1) & (self.capacity - 1);
            if index == start_index {
                return None; // Prevent infinite loop
            }
        };

        // Now borrow mutably using the found index
        if let Some((_, value)) = &mut self.buckets[found_index] {
            Some(value)
        } else {
            None
        }
    }

    /// Remove a key-value pair.
    pub fn remove(&mut self, key: u64) -> Option<V> {
        let mut index = (key as usize) & (self.capacity - 1);

        loop {
            match &self.buckets[index] {
                None => return None,
                Some((existing_key, _)) => {
                    if *existing_key == key {
                        let removed = self.buckets[index].take();
                        self.size -= 1;

                        // Rehash subsequent entries to maintain invariants
                        self.rehash_from(index);

                        return removed.map(|(_, v)| v);
                    }
                    index = (index + 1) & (self.capacity - 1);
                }
            }
        }
    }

    fn rehash_from(&mut self, start_index: usize) {
        let mut index = (start_index + 1) & (self.capacity - 1);

        while let Some((key, value)) = self.buckets[index].take() {
            self.size -= 1; // Will be incremented again in insert_internal
            self.insert_internal(key, value);
            index = (index + 1) & (self.capacity - 1);
        }
    }

    fn resize(&mut self) {
        let old_buckets = mem::replace(&mut self.buckets, vec![None; self.capacity * 2]);
        let old_size = self.size;
        self.capacity *= 2;
        self.size = 0;

        for bucket in old_buckets {
            if let Some((key, value)) = bucket {
                self.insert_internal(key, value);
            }
        }

        debug_assert_eq!(self.size, old_size);
    }

    /// Get current size.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.capacity * mem::size_of::<Option<(u64, V)>>() + mem::size_of::<Self>()
    }

    /// Iterator over key-value pairs.
    pub fn iter(&self) -> CompactHashTableIter<V> {
        CompactHashTableIter {
            buckets: &self.buckets,
            index: 0,
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.buckets.fill(None);
        self.size = 0;
    }
}

pub struct CompactHashTableIter<'a, V> {
    buckets: &'a [Option<(u64, V)>],
    index: usize,
}

impl<'a, V> Iterator for CompactHashTableIter<'a, V> {
    type Item = (u64, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buckets.len() {
            if let Some((key, value)) = &self.buckets[self.index] {
                self.index += 1;
                return Some((*key, value));
            }
            self.index += 1;
        }
        None
    }
}

/// Advanced memory pool with size-based buckets and statistics.
pub struct AdvancedSketchPool<T> {
    small_pool: Vec<T>,  // For small sketches (< 1KB)
    medium_pool: Vec<T>, // For medium sketches (1KB - 64KB)
    large_pool: Vec<T>,  // For large sketches (> 64KB)
    factory: Box<dyn Fn() -> T>,
    stats: PoolStatistics,
    max_pool_size: usize,
}

#[derive(Default, Debug)]
pub struct PoolStatistics {
    pub allocations: u64,
    pub deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_saved_bytes: u64,
}

impl<T> AdvancedSketchPool<T>
where
    T: Resettable + MemoryUsage,
{
    /// Create a new advanced memory pool.
    pub fn new<F>(factory: F, max_pool_size: usize) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            small_pool: Vec::new(),
            medium_pool: Vec::new(),
            large_pool: Vec::new(),
            factory: Box::new(factory),
            stats: PoolStatistics::default(),
            max_pool_size,
        }
    }

    /// Get an object from the appropriate size bucket.
    pub fn get(&mut self, size_hint: usize) -> T {
        let pool = self.select_pool(size_hint);

        if let Some(obj) = pool.pop() {
            self.stats.cache_hits += 1;
            self.stats.memory_saved_bytes += obj.memory_usage() as u64;
            obj
        } else {
            self.stats.cache_misses += 1;
            self.stats.allocations += 1;
            (self.factory)()
        }
    }

    /// Return an object to the appropriate pool.
    pub fn put(&mut self, mut obj: T) {
        obj.reset();
        let size = obj.memory_usage();
        let max_size = self.max_pool_size; // Avoid borrow conflict
        let pool = self.select_pool(size);

        if pool.len() < max_size {
            pool.push(obj);
            self.stats.deallocations += 1;
        }
        // If pool is full, let the object drop to prevent unbounded growth
    }

    fn select_pool(&mut self, size: usize) -> &mut Vec<T> {
        if size < 1024 {
            &mut self.small_pool
        } else if size < 65536 {
            &mut self.medium_pool
        } else {
            &mut self.large_pool
        }
    }

    /// Get pool statistics.
    pub fn statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Get total objects in all pools.
    pub fn total_pooled_objects(&self) -> usize {
        self.small_pool.len() + self.medium_pool.len() + self.large_pool.len()
    }

    /// Clear all pools and reset statistics.
    pub fn clear(&mut self) {
        self.small_pool.clear();
        self.medium_pool.clear();
        self.large_pool.clear();
        self.stats = PoolStatistics::default();
    }
}

/// Memory pool for recycling sketch objects to reduce allocation overhead.
pub struct SketchObjectPool<T> {
    pool: Vec<T>,
    factory: Box<dyn Fn() -> T>,
}

impl<T> SketchObjectPool<T> {
    /// Create a new object pool with a factory function.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            pool: Vec::new(),
            factory: Box::new(factory),
        }
    }

    /// Get an object from the pool or create a new one.
    pub fn get(&mut self) -> T {
        self.pool.pop().unwrap_or_else(|| (self.factory)())
    }

    /// Return an object to the pool for reuse.
    pub fn put(&mut self, mut obj: T)
    where
        T: Resettable,
    {
        obj.reset();
        self.pool.push(obj);
    }

    /// Get current pool size.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }
}

/// Trait for objects that can report their memory usage.
pub trait MemoryUsage {
    fn memory_usage(&self) -> usize;
}

/// Pre-allocated buffer pool for zero-allocation operations.
#[cfg(feature = "optimized")]
pub struct BufferPool {
    u8_buffers: Vec<Vec<u8>>,
    u64_buffers: Vec<Vec<u64>>,
    hash_buffers: Vec<Vec<u64>>,
    max_buffer_size: usize,
    reuse_threshold: usize,
}

#[cfg(feature = "optimized")]
impl BufferPool {
    /// Create a new buffer pool.
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            u8_buffers: Vec::new(),
            u64_buffers: Vec::new(),
            hash_buffers: Vec::new(),
            max_buffer_size,
            reuse_threshold: max_buffer_size / 4,
        }
    }

    /// Get a u8 buffer of at least the specified size.
    pub fn get_u8_buffer(&mut self, min_size: usize) -> Vec<u8> {
        // Try to find a buffer that's big enough
        for i in (0..self.u8_buffers.len()).rev() {
            if self.u8_buffers[i].capacity() >= min_size {
                let mut buffer = self.u8_buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        // No suitable buffer found, create new one
        Vec::with_capacity(min_size.max(self.reuse_threshold))
    }

    /// Return a u8 buffer to the pool.
    pub fn return_u8_buffer(&mut self, buffer: Vec<u8>) {
        if buffer.capacity() >= self.reuse_threshold
            && buffer.capacity() <= self.max_buffer_size
            && self.u8_buffers.len() < 32
        {
            self.u8_buffers.push(buffer);
        }
        // Otherwise let it drop
    }

    /// Get a u64 buffer of at least the specified size.
    pub fn get_u64_buffer(&mut self, min_size: usize) -> Vec<u64> {
        for i in (0..self.u64_buffers.len()).rev() {
            if self.u64_buffers[i].capacity() >= min_size {
                let mut buffer = self.u64_buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        Vec::with_capacity(min_size.max(self.reuse_threshold))
    }

    /// Return a u64 buffer to the pool.
    pub fn return_u64_buffer(&mut self, buffer: Vec<u64>) {
        if buffer.capacity() >= self.reuse_threshold
            && buffer.capacity() <= self.max_buffer_size
            && self.u64_buffers.len() < 32
        {
            self.u64_buffers.push(buffer);
        }
    }

    /// Get a hash buffer for temporary hash storage.
    pub fn get_hash_buffer(&mut self, min_size: usize) -> Vec<u64> {
        for i in (0..self.hash_buffers.len()).rev() {
            if self.hash_buffers[i].capacity() >= min_size {
                let mut buffer = self.hash_buffers.swap_remove(i);
                buffer.clear();
                return buffer;
            }
        }

        Vec::with_capacity(min_size.max(self.reuse_threshold))
    }

    /// Return a hash buffer to the pool.
    pub fn return_hash_buffer(&mut self, buffer: Vec<u64>) {
        if buffer.capacity() >= self.reuse_threshold
            && buffer.capacity() <= self.max_buffer_size
            && self.hash_buffers.len() < 16
        {
            self.hash_buffers.push(buffer);
        }
    }

    /// Get pool statistics.
    pub fn buffer_pool_stats(&self) -> (usize, usize, usize) {
        (
            self.u8_buffers.len(),
            self.u64_buffers.len(),
            self.hash_buffers.len(),
        )
    }

    /// Clear all buffers.
    pub fn clear(&mut self) {
        self.u8_buffers.clear();
        self.u64_buffers.clear();
        self.hash_buffers.clear();
    }
}

/// Trait for objects that can be reset for reuse in pools.
pub trait Resettable {
    fn reset(&mut self);
}

/// Cache-aligned data structure to avoid false sharing in concurrent scenarios.
#[repr(align(64))]
pub struct CacheAligned<T> {
    pub data: T,
    _padding: [u8; 0], // Ensures alignment without wasting space
}

impl<T> CacheAligned<T> {
    pub fn new(data: T) -> Self {
        Self { data, _padding: [] }
    }
}

impl<T> std::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Cache-optimized packed registers with intelligent prefetching.
#[repr(align(64))]
pub struct CacheOptimizedPackedRegisters {
    // Hot data first - frequently accessed registers
    data: Vec<u64>,
    precision: u8,
    register_bits: u8,

    // Cold data after - metadata and statistics
    _padding: [u8; 6],        // Pad to cache line boundary
    access_pattern: Vec<u32>, // Track access patterns for prefetching
    cache_stats: CacheStats,
}

#[derive(Default, Debug)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_requests: u64,
}

impl CacheOptimizedPackedRegisters {
    /// Create cache-optimized packed registers.
    pub fn new(precision: u8, register_bits: u8) -> Self {
        assert!(
            register_bits > 0 && register_bits <= 64,
            "register_bits must be between 1 and 64"
        );

        let num_registers = 1 << precision;
        let registers_per_u64 = 64 / register_bits as usize;
        let data_len = (num_registers + registers_per_u64 - 1) / registers_per_u64;

        Self {
            data: vec![0u64; data_len],
            precision,
            register_bits,
            _padding: [0; 6],
            access_pattern: vec![0; data_len],
            cache_stats: CacheStats::default(),
        }
    }

    /// Get register value with cache optimization.
    #[inline(always)]
    pub fn get(&mut self, index: usize) -> u8 {
        let registers_per_u64 = 64 / self.register_bits as usize;
        let word_index = index / registers_per_u64;

        // Track access pattern for prefetching
        if word_index < self.access_pattern.len() {
            self.access_pattern[word_index] = self.access_pattern[word_index].saturating_add(1);

            // Prefetch next cache line if this word is frequently accessed
            if self.access_pattern[word_index] > 10 && word_index + 8 < self.data.len() {
                self.prefetch_cache_line(word_index + 8);
            }
        }

        let bit_index = (index % registers_per_u64) * self.register_bits as usize;
        let mask = if self.register_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.register_bits) - 1
        };

        ((self.data[word_index] >> bit_index) & mask) as u8
    }

    /// Set register value with cache optimization.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: u8) {
        let registers_per_u64 = 64 / self.register_bits as usize;
        let word_index = index / registers_per_u64;
        let bit_index = (index % registers_per_u64) * self.register_bits as usize;
        let mask = if self.register_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.register_bits) - 1
        };

        // Track write access
        if word_index < self.access_pattern.len() {
            self.access_pattern[word_index] = self.access_pattern[word_index].saturating_add(2); // Writes are more expensive
        }

        // Clear and set bits
        self.data[word_index] &= !(mask << bit_index);
        self.data[word_index] |= (value as u64 & mask) << bit_index;
    }

    /// Prefetch cache line for better performance.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline(always)]
    fn prefetch_cache_line(&mut self, word_index: usize) {
        if word_index < self.data.len() {
            self.cache_stats.prefetch_requests += 1;

            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    self.data.as_ptr().add(word_index) as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                // Use standard prefetch hint instead of unstable intrinsics
                std::ptr::read_volatile(self.data.as_ptr().add(word_index));
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    fn prefetch_cache_line(&mut self, _word_index: usize) {
        // No-op on unsupported architectures
        self.cache_stats.prefetch_requests += 1;
    }

    /// Branch-free batch update optimized for cache locality.
    #[cfg(feature = "optimized")]
    pub fn update_max_batch_cache_optimized(&mut self, updates: &[(usize, u8)]) {
        use rayon::prelude::*;

        // Sort by cache line for optimal access pattern
        let mut sorted_updates = updates.to_vec();
        let registers_per_u64 = 64 / self.register_bits as usize;

        sorted_updates.par_sort_unstable_by_key(|(index, _)| {
            let word_index = index / registers_per_u64;
            word_index / 8 // Group by cache line (64 bytes = 8 u64s)
        });

        // Process updates in cache-friendly order
        let mut last_cache_line = usize::MAX;
        for (index, value) in sorted_updates {
            let word_index = index / registers_per_u64;
            let cache_line = word_index / 8;

            // Prefetch next cache line if we've moved to a new one
            if cache_line != last_cache_line {
                if cache_line + 1 < (self.data.len() + 7) / 8 {
                    self.prefetch_cache_line((cache_line + 1) * 8);
                }
                last_cache_line = cache_line;
            }

            let current = self.get(index);
            let mask = ((value > current) as u8).wrapping_sub(1);
            let new_value = (current & mask) | (value & !mask);
            self.set(index, new_value);
        }
    }

    /// Get cache performance statistics.
    pub fn cache_statistics(&self) -> &CacheStats {
        &self.cache_stats
    }

    /// Get number of registers.
    pub fn len(&self) -> usize {
        1 << self.precision
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_registers() {
        let mut registers = PackedRegisters::new(4, 6); // 16 registers, 6 bits each

        // Test basic set/get
        registers.set(0, 15);
        registers.set(15, 63);
        assert_eq!(registers.get(0), 15);
        assert_eq!(registers.get(15), 63);

        // Test all values are initially zero
        for i in 1..15 {
            assert_eq!(registers.get(i), 0);
        }
    }

    #[test]
    fn test_packed_registers_update_max() {
        let mut registers = PackedRegisters::new(4, 6);

        registers.update_max(0, 10);
        assert_eq!(registers.get(0), 10);

        registers.update_max(0, 5); // Should not update
        assert_eq!(registers.get(0), 10);

        registers.update_max(0, 15); // Should update
        assert_eq!(registers.get(0), 15);
    }

    #[test]
    fn test_compact_hash_table() {
        let mut table = CompactHashTable::new(4);

        // Test insertion
        assert_eq!(table.insert(1, "one"), None);
        assert_eq!(table.insert(2, "two"), None);
        assert_eq!(table.len(), 2);

        // Test retrieval
        assert_eq!(table.get(1), Some(&"one"));
        assert_eq!(table.get(2), Some(&"two"));
        assert_eq!(table.get(3), None);

        // Test update
        assert_eq!(table.insert(1, "ONE"), Some("one"));
        assert_eq!(table.get(1), Some(&"ONE"));
        assert_eq!(table.len(), 2);

        // Test removal
        assert_eq!(table.remove(1), Some("ONE"));
        assert_eq!(table.get(1), None);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_compact_hash_table_resize() {
        let mut table = CompactHashTable::new(2);

        // Insert enough items to trigger resize
        for i in 0..10 {
            table.insert(i, i * 2);
        }

        // Verify all items are still accessible
        for i in 0..10 {
            assert_eq!(table.get(i), Some(&(i * 2)));
        }

        assert_eq!(table.len(), 10);
    }

    #[test]
    fn test_object_pool() {
        #[derive(Debug, PartialEq)]
        struct TestObject {
            value: i32,
        }

        impl TestObject {
            fn new() -> Self {
                Self { value: 0 }
            }
        }

        impl Resettable for TestObject {
            fn reset(&mut self) {
                self.value = 0;
            }
        }

        let mut pool = SketchObjectPool::new(TestObject::new);

        // Get object from empty pool (creates new)
        let mut obj1 = pool.get();
        obj1.value = 42;
        assert_eq!(pool.pool_size(), 0);

        // Return object to pool
        pool.put(obj1);
        assert_eq!(pool.pool_size(), 1);

        // Get object from pool (reused)
        let obj2 = pool.get();
        assert_eq!(obj2.value, 0); // Should be reset
        assert_eq!(pool.pool_size(), 0);
    }
}
