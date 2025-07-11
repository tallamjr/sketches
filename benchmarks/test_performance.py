#!/usr/bin/env python3
"""
Comprehensive benchmark suite comparing our sketches library with Apache DataSketches.

This suite tests:
- Processing throughput (items/second)
- Memory efficiency 
- Accuracy comparison
- Serialization performance
"""

import pytest
import gc
import psutil
import os
import random
import string
from typing import List, Dict, Any

# Import both libraries for comparison
import sketches as our_sketches
import datasketches as apache_ds


def generate_test_data(size: int, unique_ratio: float = 1.0) -> List[str]:
    """Generate test data with controlled uniqueness."""
    unique_count = int(size * unique_ratio)
    base_data = [f"item_{i:08d}" for i in range(unique_count)]
    
    if unique_ratio < 1.0:
        # Add duplicates to reach desired size
        additional = random.choices(base_data, k=size - unique_count)
        data = base_data + additional
        random.shuffle(data)
        return data
    
    return base_data


class TestHyperLogLogPerformance:
    """Benchmark HyperLogLog sketches."""

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000, 1000000])
    def test_hll_update_throughput_our_library(self, benchmark, data_size):
        """Benchmark our HLL update performance."""
        data = generate_test_data(data_size)
        
        def update_sketch():
            sketch = our_sketches.HllSketch(lg_k=12)
            for item in data:
                sketch.update(item)
            return sketch.estimate()
        
        result = benchmark(update_sketch)
        assert result > 0

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000, 1000000])
    def test_hll_update_throughput_apache(self, benchmark, data_size):
        """Benchmark Apache DataSketches HLL update performance."""
        data = generate_test_data(data_size)
        
        def update_sketch():
            sketch = apache_ds.hll_sketch(12)  # lg_k=12
            for item in data:
                sketch.update(item)
            return sketch.get_estimate()
        
        result = benchmark(update_sketch)
        assert result > 0

    def test_hll_memory_efficiency(self):
        """Compare memory usage between implementations."""
        data_size = 1000000
        data = generate_test_data(data_size)
        
        process = psutil.Process(os.getpid())
        
        # Test our library
        gc.collect()
        mem_before = process.memory_info().rss
        our_sketch = our_sketches.HllSketch(lg_k=12)
        for item in data:
            our_sketch.update(item)
        our_memory = process.memory_info().rss - mem_before
        our_estimate = our_sketch.estimate()
        
        # Clear memory
        del our_sketch
        gc.collect()
        
        # Test Apache DataSketches
        mem_before = process.memory_info().rss
        apache_sketch = apache_ds.hll_sketch(12)
        for item in data:
            apache_sketch.update(item)
        apache_memory = process.memory_info().rss - mem_before
        apache_estimate = apache_sketch.get_estimate()
        
        # Results
        print(f"\n=== HLL Memory Comparison ({data_size:,} items) ===")
        print(f"Our library: {our_memory / 1024:.1f} KB, estimate: {our_estimate:,.0f}")
        print(f"Apache DS:   {apache_memory / 1024:.1f} KB, estimate: {apache_estimate:,.0f}")
        memory_ratio = our_memory / apache_memory if apache_memory > 0 else float('inf')
        print(f"Memory ratio: {memory_ratio:.2f}x")
        
        return {
            'our_memory': our_memory,
            'apache_memory': apache_memory,
            'our_estimate': our_estimate,
            'apache_estimate': apache_estimate,
            'data_size': data_size
        }

    def test_hll_accuracy_comparison(self):
        """Compare accuracy across different data sizes."""
        results = []
        
        for size in [1000, 10000, 100000, 1000000]:
            data = generate_test_data(size, unique_ratio=1.0)  # All unique
            true_count = len(set(data))
            
            # Our library
            our_sketch = our_sketches.HllSketch(lg_k=12)
            for item in data:
                our_sketch.update(item)
            our_estimate = our_sketch.estimate()
            our_error = abs(our_estimate - true_count) / true_count * 100 if true_count > 0 else 0.0
            
            # Apache DataSketches
            apache_sketch = apache_ds.hll_sketch(12)
            for item in data:
                apache_sketch.update(item)
            apache_estimate = apache_sketch.get_estimate()
            apache_error = abs(apache_estimate - true_count) / true_count * 100 if true_count > 0 else 0.0
            
            result = {
                'size': size,
                'true_count': true_count,
                'our_estimate': our_estimate,
                'our_error': our_error,
                'apache_estimate': apache_estimate,
                'apache_error': apache_error
            }
            results.append(result)
            
            print(f"\nSize: {size:,}")
            print(f"True count: {true_count:,}")
            print(f"Our estimate: {our_estimate:,.0f} (error: {our_error:.2f}%)")
            print(f"Apache estimate: {apache_estimate:,.0f} (error: {apache_error:.2f}%)")
        
        return results

    def test_hll_serialization_performance(self, benchmark):
        """Compare serialization performance."""
        data = generate_test_data(100000)
        
        # Prepare sketches
        our_sketch = our_sketches.HllSketch(lg_k=12)
        apache_sketch = apache_ds.hll_sketch(12)
        
        for item in data:
            our_sketch.update(item)
            apache_sketch.update(item)
        
        def serialize_our():
            return our_sketch.to_bytes()
        
        def serialize_apache():
            return apache_sketch.serialize_compact()
        
        # Benchmark our serialization
        our_bytes = benchmark.pedantic(serialize_our, rounds=100)
        
        # Test Apache serialization (separate benchmark)
        apache_bytes = serialize_apache()
        
        print(f"\n=== HLL Serialization Comparison ===")
        print(f"Our library: {len(our_bytes)} bytes")
        print(f"Apache DS:   {len(apache_bytes)} bytes")
        size_ratio = len(our_bytes) / len(apache_bytes) if len(apache_bytes) > 0 else float('inf')
        print(f"Size ratio: {size_ratio:.2f}x")
        
        return {
            'our_size': len(our_bytes),
            'apache_size': len(apache_bytes)
        }


class TestThetaSketchPerformance:
    """Benchmark Theta sketches."""

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_theta_update_throughput_our_library(self, benchmark, data_size):
        """Benchmark our Theta sketch performance."""
        data = generate_test_data(data_size)
        
        def update_sketch():
            sketch = our_sketches.ThetaSketch(k=4096)
            for item in data:
                sketch.update(item)
            return sketch.estimate()
        
        result = benchmark(update_sketch)
        assert result > 0

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_theta_update_throughput_apache(self, benchmark, data_size):
        """Benchmark Apache DataSketches Theta performance."""
        data = generate_test_data(data_size)
        
        def update_sketch():
            sketch = apache_ds.update_theta_sketch()
            for item in data:
                sketch.update(item)
            return sketch.get_estimate()
        
        result = benchmark(update_sketch)
        assert result > 0

    def test_theta_set_operations_performance(self):
        """Benchmark set operations."""
        data1 = generate_test_data(50000)
        data2 = generate_test_data(50000)
        
        # Our library
        our_s1 = our_sketches.ThetaSketch(k=4096)
        our_s2 = our_sketches.ThetaSketch(k=4096)
        
        for item in data1:
            our_s1.update(item)
        for item in data2:
            our_s2.update(item)
        
        # Apache DataSketches
        apache_s1 = apache_ds.update_theta_sketch()
        apache_s2 = apache_ds.update_theta_sketch()
        
        for item in data1:
            apache_s1.update(item)
        for item in data2:
            apache_s2.update(item)
        
        print(f"\n=== Theta Set Operations ===")
        
        # Union
        our_union = our_s1.union(our_s2)
        apache_union = apache_ds.theta_union()
        apache_union.update(apache_s1)
        apache_union.update(apache_s2)
        apache_union_result = apache_union.get_result()
        
        print(f"Union - Our: {our_union.estimate():.0f}, Apache: {apache_union_result.get_estimate():.0f}")
        
        # Intersection
        our_intersection = our_s1.intersect(our_s2)
        apache_intersection = apache_ds.theta_intersection()
        apache_intersection.update(apache_s1)
        apache_intersection.update(apache_s2)
        apache_intersection_result = apache_intersection.get_result()
        
        print(f"Intersection - Our: {our_intersection.estimate():.0f}, Apache: {apache_intersection_result.get_estimate():.0f}")


class TestMemoryBenchmarks:
    """Memory-focused benchmarks."""

    def test_memory_scaling(self):
        """Test memory usage as data size increases."""
        sizes = [1000, 10000, 100000, 1000000]
        results = []
        
        for size in sizes:
            data = generate_test_data(size)
            process = psutil.Process(os.getpid())
            
            # Our library
            gc.collect()
            mem_before = process.memory_info().rss
            sketch = our_sketches.HllSketch(lg_k=12)
            for item in data:
                sketch.update(item)
            our_memory = process.memory_info().rss - mem_before
            
            del sketch
            gc.collect()
            
            # Apache DataSketches
            mem_before = process.memory_info().rss
            apache_sketch = apache_ds.hll_sketch(12)
            for item in data:
                apache_sketch.update(item)
            apache_memory = process.memory_info().rss - mem_before
            
            result = {
                'size': size,
                'our_memory': our_memory,
                'apache_memory': apache_memory,
                'ratio': our_memory / apache_memory if apache_memory > 0 else float('inf')
            }
            results.append(result)
            
            ratio = our_memory/apache_memory if apache_memory > 0 else float('inf')
            print(f"Size: {size:,} - Our: {our_memory/1024:.1f}KB, Apache: {apache_memory/1024:.1f}KB, Ratio: {ratio:.2f}x")
        
        return results


class TestThroughputBenchmarks:
    """Throughput-focused benchmarks."""

    def test_streaming_performance(self):
        """Test performance on streaming data."""
        # Generate large streaming dataset
        total_items = 2000000
        batch_size = 10000
        
        our_sketch = our_sketches.HllSketch(lg_k=14)  # Larger sketch for big data
        apache_sketch = apache_ds.hll_sketch(14)
        
        import time
        
        # Our library timing
        start_time = time.time()
        for batch_start in range(0, total_items, batch_size):
            batch = [f"stream_item_{i}" for i in range(batch_start, min(batch_start + batch_size, total_items))]
            for item in batch:
                our_sketch.update(item)
        our_time = time.time() - start_time
        our_estimate = our_sketch.estimate()
        
        # Apache DataSketches timing
        start_time = time.time()
        for batch_start in range(0, total_items, batch_size):
            batch = [f"stream_item_{i}" for i in range(batch_start, min(batch_start + batch_size, total_items))]
            for item in batch:
                apache_sketch.update(item)
        apache_time = time.time() - start_time
        apache_estimate = apache_sketch.get_estimate()
        
        our_throughput = total_items / our_time if our_time > 0 else float('inf')
        apache_throughput = total_items / apache_time if apache_time > 0 else float('inf')
        
        print(f"\n=== Streaming Performance ({total_items:,} items) ===")
        print(f"Our library: {our_time:.2f}s, {our_throughput:,.0f} items/sec, estimate: {our_estimate:,.0f}")
        print(f"Apache DS:   {apache_time:.2f}s, {apache_throughput:,.0f} items/sec, estimate: {apache_estimate:,.0f}")
        speedup = apache_time / our_time if our_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        return {
            'our_time': our_time,
            'apache_time': apache_time,
            'our_throughput': our_throughput,
            'apache_throughput': apache_throughput,
            'speedup': apache_time / our_time if our_time > 0 else float('inf'),
            'total_items': total_items
        }


if __name__ == "__main__":
    # Run individual tests for quick validation
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_memory = TestHyperLogLogPerformance()
        test_memory.test_hll_memory_efficiency()
        test_memory.test_hll_accuracy_comparison()
        
        test_throughput = TestThroughputBenchmarks()
        test_throughput.test_streaming_performance()
    else:
        print("Run with 'python test_performance.py quick' for quick tests")
        print("Or use 'pytest benchmarks/test_performance.py --benchmark-only' for full benchmarks")