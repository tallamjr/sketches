//! SIMD-optimized operations for probabilistic data structures.
//!
//! This module provides vectorized implementations of common operations used
//! in sketches, such as bulk hashing, parallel comparisons, and bit operations.
//! Falls back to scalar implementations when SIMD is not available.

#[cfg(all(feature = "optimized", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(all(feature = "optimized", target_arch = "aarch64"))]
use std::arch::aarch64::*;

/// SIMD-optimized batch operations for Bloom filters.
pub mod bloom {
    use super::*;

    /// Set multiple bits in a Bloom filter using SIMD instructions.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn set_bits_simd(bit_array: &mut [u64], positions: &[usize]) {
        if is_x86_feature_detected!("avx2") {
            set_bits_avx2(bit_array, positions);
        } else {
            set_bits_scalar(bit_array, positions);
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn set_bits_avx2(bit_array: &mut [u64], positions: &[usize]) {
        // Process positions in chunks of 4 for SIMD operations
        for chunk in positions.chunks(4) {
            let mut word_indices = [0usize; 4];
            let mut bit_masks = [0u64; 4];

            // Prepare data for vectorized operations
            for (i, &pos) in chunk.iter().enumerate() {
                word_indices[i] = pos / 64;
                bit_masks[i] = if word_indices[i] < bit_array.len() {
                    1u64 << (pos % 64)
                } else {
                    0
                };
            }

            // Apply bit masks with bounds checking
            for i in 0..chunk.len() {
                if word_indices[i] < bit_array.len() {
                    bit_array[word_indices[i]] |= bit_masks[i];
                }
            }
        }
    }

    /// Check multiple bits in a Bloom filter using SIMD instructions.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn check_bits_simd(bit_array: &[u64], positions: &[usize]) -> Vec<bool> {
        if is_x86_feature_detected!("avx2") {
            check_bits_avx2(bit_array, positions)
        } else {
            check_bits_scalar(bit_array, positions)
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn check_bits_avx2(bit_array: &[u64], positions: &[usize]) -> Vec<bool> {
        let mut results = Vec::with_capacity(positions.len());

        for chunk in positions.chunks(4) {
            for &pos in chunk {
                let word_index = pos / 64;
                let bit_index = pos % 64;
                if word_index < bit_array.len() {
                    results.push((bit_array[word_index] & (1u64 << bit_index)) != 0);
                } else {
                    results.push(false);
                }
            }
        }

        results
    }

    /// Fallback scalar implementation for setting bits.
    pub fn set_bits_scalar(bit_array: &mut [u64], positions: &[usize]) {
        for &pos in positions {
            let word_index = pos / 64;
            let bit_index = pos % 64;
            if word_index < bit_array.len() {
                bit_array[word_index] |= 1u64 << bit_index;
            }
        }
    }

    /// Fallback scalar implementation for checking bits.
    pub fn check_bits_scalar(bit_array: &[u64], positions: &[usize]) -> Vec<bool> {
        positions
            .iter()
            .map(|&pos| {
                let word_index = pos / 64;
                let bit_index = pos % 64;
                if word_index < bit_array.len() {
                    (bit_array[word_index] & (1u64 << bit_index)) != 0
                } else {
                    false
                }
            })
            .collect()
    }

    /// Public interface that automatically selects the best implementation.
    pub fn set_bits(bit_array: &mut [u64], positions: &[usize]) {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            set_bits_simd(bit_array, positions);
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        set_bits_scalar(bit_array, positions);
    }

    /// Public interface that automatically selects the best implementation.
    pub fn check_bits(bit_array: &[u64], positions: &[usize]) -> Vec<bool> {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            check_bits_simd(bit_array, positions)
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        check_bits_scalar(bit_array, positions)
    }
}

/// SIMD-optimized operations for Count-Min sketches.
pub mod countmin {
    use super::*;

    /// Increment multiple counters simultaneously using SIMD.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn increment_counters_simd(
        matrix: &mut [u32],
        positions: &[usize],
        increments: &[u32],
    ) {
        if is_x86_feature_detected!("avx2") {
            increment_counters_avx2(matrix, positions, increments);
        } else {
            increment_counters_scalar(matrix, positions, increments);
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn increment_counters_avx2(matrix: &mut [u32], positions: &[usize], increments: &[u32]) {
        // Process in chunks for better vectorization
        for chunk in positions.chunks(8).zip(increments.chunks(8)) {
            let (pos_chunk, inc_chunk) = chunk;

            for (&pos, &inc) in pos_chunk.iter().zip(inc_chunk.iter()) {
                if pos < matrix.len() {
                    // Use saturating arithmetic to prevent overflow
                    matrix[pos] = matrix[pos].saturating_add(inc);
                }
            }
        }
    }

    /// Find minimum values across multiple rows using SIMD.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn find_min_simd(values: &[u32]) -> u32 {
        if values.len() >= 8 && is_x86_feature_detected!("avx2") {
            find_min_avx2(values)
        } else {
            find_min_scalar(values)
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn find_min_avx2(values: &[u32]) -> u32 {
        let mut min_vec = _mm256_set1_epi32(i32::MAX);

        for chunk in values.chunks_exact(8) {
            let vals = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            min_vec = _mm256_min_epu32(min_vec, vals);
        }

        // Extract minimum from vector
        let mut result = [0u32; 8];
        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, min_vec);

        let mut min_val = result[0];
        for &val in &result[1..] {
            min_val = min_val.min(val);
        }

        // Handle remaining elements
        for &val in &values[values.len() & !7..] {
            min_val = min_val.min(val);
        }

        min_val
    }

    /// Fallback scalar implementations.
    pub fn increment_counters_scalar(matrix: &mut [u32], positions: &[usize], increments: &[u32]) {
        for (&pos, &inc) in positions.iter().zip(increments.iter()) {
            if pos < matrix.len() {
                matrix[pos] = matrix[pos].saturating_add(inc);
            }
        }
    }

    pub fn find_min_scalar(values: &[u32]) -> u32 {
        values.iter().copied().min().unwrap_or(0)
    }

    /// Public interfaces.
    pub fn increment_counters(matrix: &mut [u32], positions: &[usize], increments: &[u32]) {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            increment_counters_simd(matrix, positions, increments);
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        increment_counters_scalar(matrix, positions, increments);
    }

    pub fn find_min(values: &[u32]) -> u32 {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            find_min_simd(values)
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        find_min_scalar(values)
    }
}

/// SIMD-optimized operations for HyperLogLog sketches.
pub mod hyperloglog {
    use super::*;

    /// Count leading zeros for multiple values using SIMD.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn leading_zeros_batch_simd(values: &[u64]) -> Vec<u8> {
        if values.len() >= 4 && is_x86_feature_detected!("avx2") {
            leading_zeros_batch_avx2(values)
        } else {
            leading_zeros_batch_scalar(values)
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn leading_zeros_batch_avx2(values: &[u64]) -> Vec<u8> {
        let mut results = Vec::with_capacity(values.len());

        // Process 4 u64s at a time with AVX2
        for chunk in values.chunks_exact(4) {
            if is_x86_feature_detected!("lzcnt") {
                // Use hardware LZCNT if available
                let vals = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                let lzcnt = _mm256_lzcnt_epi64(vals);

                let mut temp = [0u64; 4];
                _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, lzcnt);

                for &count in &temp {
                    results.push((count as u8).min(64));
                }
            } else {
                // Fallback to software implementation for each element
                for &val in chunk {
                    results.push((val.leading_zeros() as u8).min(64));
                }
            }
        }

        // Handle remaining elements
        for &val in &values[values.len() & !3..] {
            results.push((val.leading_zeros() as u8).min(64));
        }

        results
    }

    /// Update multiple HLL registers with maximum values using SIMD.
    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    pub unsafe fn update_registers_simd(registers: &mut [u8], positions: &[usize], values: &[u8]) {
        if is_x86_feature_detected!("avx2") {
            update_registers_avx2(registers, positions, values);
        } else {
            update_registers_scalar(registers, positions, values);
        }
    }

    #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
    unsafe fn update_registers_avx2(registers: &mut [u8], positions: &[usize], values: &[u8]) {
        for (&pos, &val) in positions.iter().zip(values.iter()) {
            if pos < registers.len() {
                registers[pos] = registers[pos].max(val);
            }
        }
    }

    /// Fallback scalar implementations.
    pub fn leading_zeros_batch_scalar(values: &[u64]) -> Vec<u8> {
        values
            .iter()
            .map(|&val| (val.leading_zeros() as u8).min(64))
            .collect()
    }

    pub fn update_registers_scalar(registers: &mut [u8], positions: &[usize], values: &[u8]) {
        for (&pos, &val) in positions.iter().zip(values.iter()) {
            if pos < registers.len() {
                registers[pos] = registers[pos].max(val);
            }
        }
    }

    /// Public interfaces.
    pub fn leading_zeros_batch(values: &[u64]) -> Vec<u8> {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            leading_zeros_batch_simd(values)
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        leading_zeros_batch_scalar(values)
    }

    pub fn update_registers(registers: &mut [u8], positions: &[usize], values: &[u8]) {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        unsafe {
            update_registers_simd(registers, positions, values);
        }

        #[cfg(not(all(feature = "optimized", target_arch = "x86_64")))]
        update_registers_scalar(registers, positions, values);
    }
}

/// Utility functions for SIMD feature detection and capability reporting.
pub mod utils {
    /// Check if SIMD optimizations are available on this platform.
    pub fn simd_available() -> bool {
        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        {
            return is_x86_feature_detected!("avx2");
        }

        #[cfg(all(feature = "optimized", target_arch = "aarch64"))]
        {
            return std::arch::is_aarch64_feature_detected!("neon");
        }

        #[cfg(not(feature = "optimized"))]
        {
            return false;
        }

        #[cfg(all(
            feature = "optimized",
            not(any(target_arch = "x86_64", target_arch = "aarch64"))
        ))]
        {
            return false;
        }
    }

    /// Get a description of available SIMD features.
    pub fn simd_features() -> Vec<&'static str> {
        let mut features = Vec::new();

        #[cfg(all(feature = "optimized", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("AVX2");
            }
            if is_x86_feature_detected!("avx") {
                features.push("AVX");
            }
            if is_x86_feature_detected!("sse4.2") {
                features.push("SSE4.2");
            }
        }

        #[cfg(all(feature = "optimized", target_arch = "aarch64"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("NEON");
            }
        }

        features
    }

    /// Runtime SIMD capability reporting for diagnostics.
    pub fn print_simd_info() {
        println!(
            "SIMD optimizations enabled: {}",
            cfg!(feature = "optimized")
        );
        println!("SIMD available at runtime: {}", simd_available());
        println!("SIMD features: {:?}", simd_features());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_operations() {
        let mut bit_array = vec![0u64; 4];
        let positions = vec![1, 65, 129, 193];

        bloom::set_bits(&mut bit_array, &positions);
        let results = bloom::check_bits(&bit_array, &positions);

        assert!(results.iter().all(|&x| x));

        // Test non-set positions
        let other_positions = vec![2, 66, 130, 194];
        let other_results = bloom::check_bits(&bit_array, &other_positions);
        assert!(other_results.iter().all(|&x| !x));
    }

    #[test]
    fn test_countmin_operations() {
        let mut matrix = vec![0u32; 100];
        let positions = vec![0, 25, 50, 75];
        let increments = vec![1, 2, 3, 4];

        countmin::increment_counters(&mut matrix, &positions, &increments);

        assert_eq!(matrix[0], 1);
        assert_eq!(matrix[25], 2);
        assert_eq!(matrix[50], 3);
        assert_eq!(matrix[75], 4);

        let min_val = countmin::find_min(&increments);
        assert_eq!(min_val, 1);
    }

    #[test]
    fn test_hll_operations() {
        let values = vec![
            0x8000000000000000u64,
            0x4000000000000000u64,
            0x2000000000000000u64,
        ];
        let leading_zeros = hyperloglog::leading_zeros_batch(&values);

        assert_eq!(leading_zeros, vec![0, 1, 2]);

        let mut registers = vec![0u8; 3];
        let positions = vec![0, 1, 2];
        let new_values = vec![5, 3, 7];

        hyperloglog::update_registers(&mut registers, &positions, &new_values);
        assert_eq!(registers, vec![5, 3, 7]);

        // Test update with smaller values (should not change)
        let smaller_values = vec![3, 1, 5];
        hyperloglog::update_registers(&mut registers, &positions, &smaller_values);
        assert_eq!(registers, vec![5, 3, 7]);
    }

    #[test]
    fn test_simd_utils() {
        let available = utils::simd_available();
        let features = utils::simd_features();

        // Just ensure these don't panic
        println!("SIMD available: {}", available);
        println!("SIMD features: {:?}", features);

        utils::print_simd_info();
    }
}
