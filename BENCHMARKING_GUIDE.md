# 🏁 Benchmarking Guide

**Comprehensive guide for measuring and comparing performance against Apache DataSketches**

---

## 🎯 Benchmarking Objectives

### Primary Goals:
1. **Track Progress**: Monitor improvements throughout roadmap implementation
2. **Validate Claims**: Ensure all performance improvements are real and measurable  
3. **Competitive Analysis**: Continuously compare against Apache DataSketches
4. **Regression Prevention**: Catch performance regressions early

### Key Metrics:
- **Throughput**: Items processed per second
- **Memory Usage**: RAM consumption per sketch
- **Accuracy**: Error rates across different cardinalities
- **Latency**: Time for individual operations

---

## 🛠 Benchmarking Infrastructure

### Required Tools:
```bash
# Rust benchmarking
cargo install cargo-criterion
cargo install cargo-bench
cargo install cargo-flamegraph

# Python benchmarking  
pip install pytest-benchmark
pip install memory_profiler
pip install line_profiler

# System monitoring
# macOS: Install Instruments.app (Xcode)
# Linux: Install perf, valgrind
```

### Benchmark Environment Setup:
```bash
# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable turbo boost for consistent results
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Set CPU affinity for benchmark process
taskset -c 0 cargo bench

# Increase process priority
nice -n -20 cargo bench
```

---

## 📊 Rust Micro-Benchmarks

### 1. Core Operation Benchmarks

Create `benches/core_operations.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use sketches::{HllSketch, ThetaSketch, CpcSketch};

fn bench_hll_update_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hll_update_throughput");
    
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let items: Vec<String> = (0..*size).map(|i| format!("item_{}", i)).collect();
        
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            let mut hll = HllSketch::new(12).with_simd(false);
            b.iter(|| {
                for item in &items {
                    hll.update(black_box(item));
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("chunked", size), size, |b, _| {
            let mut hll = HllSketch::new(12).with_simd(true);
            b.iter(|| {
                for item in &items {
                    hll.update(black_box(item));
                }
            });
        });
        
        // Future: Add SIMD and GPU variants
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            let mut hll = HllSketch::new(12).with_true_simd(true);
            b.iter(|| {
                for item in &items {
                    hll.update(black_box(item));
                }
            });
        });
    }
    
    group.finish();
}

fn bench_hll_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("hll_memory");
    
    for precision in [8, 10, 12, 14, 16].iter() {
        group.bench_with_input(BenchmarkId::new("dense", precision), precision, |b, &p| {
            b.iter(|| {
                let hll = HllSketch::new(p);
                black_box(hll.memory_usage())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sparse", precision), precision, |b, &p| {
            b.iter(|| {
                let hll = HllSketch::new(p).with_sparse_mode(true);
                black_box(hll.memory_usage())
            });
        });
    }
    
    group.finish();
}

fn bench_hll_estimation_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("hll_accuracy");
    
    for true_cardinality in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let items: Vec<String> = (0..*true_cardinality).map(|i| format!("item_{}", i)).collect();
        
        group.bench_with_input(
            BenchmarkId::new("accuracy", true_cardinality), 
            true_cardinality, 
            |b, &true_card| {
                b.iter(|| {
                    let mut hll = HllSketch::new(12);
                    for item in &items {
                        hll.update(item);
                    }
                    let estimate = hll.estimate();
                    let error = (estimate - true_card as f64).abs() / true_card as f64;
                    black_box(error)
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_hll_update_throughput,
    bench_hll_memory_usage,
    bench_hll_estimation_accuracy
);
criterion_main!(benches);
```

### 2. Comparative Benchmarks

Create `benches/apache_comparison.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::process::Command;
use serde_json::Value;

fn bench_against_apache_datasketches(c: &mut Criterion) {
    let mut group = c.benchmark_group("apache_comparison");
    
    for size in [10_000, 100_000, 1_000_000].iter() {
        // Benchmark our implementation
        group.bench_with_input(BenchmarkId::new("our_rust", size), size, |b, &n| {
            let items: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();
            b.iter(|| {
                let mut hll = sketches::HllSketch::new(12);
                for item in &items {
                    hll.update(black_box(item));
                }
                black_box(hll.estimate())
            });
        });
        
        // Benchmark Apache DataSketches via Python subprocess
        group.bench_with_input(BenchmarkId::new("apache_python", size), size, |b, &n| {
            b.iter(|| {
                let output = Command::new("python3")
                    .arg("-c")
                    .arg(&format!(
                        "
import time
from datasketches import hll_sketch
start = time.time()
hll = hll_sketch(12)
for i in range({}):
    hll.update(f'item_{{i}}')
estimate = hll.get_estimate()
duration = time.time() - start
print(f'{{\"estimate\": {estimate}, \"duration\": {duration}}}')
                        ", n
                    ))
                    .output()
                    .expect("Failed to run Python benchmark");
                
                let result: Value = serde_json::from_str(
                    &String::from_utf8(output.stdout).unwrap()
                ).unwrap();
                
                black_box(result["estimate"].as_f64().unwrap())
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_against_apache_datasketches);
criterion_main!(benches);
```

---

## 🐍 Python Benchmarks

### 1. PyTest Benchmark Suite

Create `tests/test_performance.py`:
```python
import pytest
import numpy as np
import psutil
import gc
from memory_profiler import memory_usage
import sketches
try:
    import datasketches
    HAS_APACHE = True
except ImportError:
    HAS_APACHE = False

class TestPerformanceComparison:
    @pytest.mark.parametrize("n_items", [1_000, 10_000, 100_000, 1_000_000])
    @pytest.mark.benchmark(group="hll_throughput")
    def test_hll_update_throughput_our_library(self, benchmark, n_items):
        """Benchmark our HLL implementation throughput."""
        items = [f"item_{i}" for i in range(n_items)]
        hll = sketches.HllSketch(12)
        
        def update_all():
            for item in items:
                hll.update(item)
            return hll.estimate()
        
        result = benchmark(update_all)
        assert 0.8 * n_items <= result <= 1.2 * n_items  # 20% accuracy tolerance
    
    @pytest.mark.skipif(not HAS_APACHE, reason="Apache DataSketches not available")
    @pytest.mark.parametrize("n_items", [1_000, 10_000, 100_000, 1_000_000])
    @pytest.mark.benchmark(group="hll_throughput")
    def test_hll_update_throughput_apache(self, benchmark, n_items):
        """Benchmark Apache DataSketches throughput."""
        items = [f"item_{i}" for i in range(n_items)]
        hll = datasketches.hll_sketch(12)
        
        def update_all():
            for item in items:
                hll.update(item)
            return hll.get_estimate()
        
        result = benchmark(update_all)
        assert 0.8 * n_items <= result <= 1.2 * n_items

    @pytest.mark.parametrize("n_items", [1_000, 10_000, 100_000])
    def test_hll_memory_usage_our_library(self, n_items):
        """Measure memory usage of our HLL implementation."""
        items = [f"item_{i}" for i in range(n_items)]
        
        def create_and_update():
            hll = sketches.HllSketch(12)
            for item in items:
                hll.update(item)
            return hll
        
        # Measure memory usage during creation and updates
        mem_usage = memory_usage(create_and_update, interval=0.1)
        max_memory_mb = max(mem_usage)
        
        print(f"Our Library - {n_items} items: {max_memory_mb:.2f} MB")
        
        # Memory should be reasonable (less than 10MB for 100K items)
        assert max_memory_mb < 10.0 * (n_items / 100_000)

    @pytest.mark.skipif(not HAS_APACHE, reason="Apache DataSketches not available")
    @pytest.mark.parametrize("n_items", [1_000, 10_000, 100_000])
    def test_hll_memory_usage_apache(self, n_items):
        """Measure memory usage of Apache DataSketches."""
        items = [f"item_{i}" for i in range(n_items)]
        
        def create_and_update():
            hll = datasketches.hll_sketch(12)
            for item in items:
                hll.update(item)
            return hll
        
        mem_usage = memory_usage(create_and_update, interval=0.1)
        max_memory_mb = max(mem_usage)
        
        print(f"Apache DataSketches - {n_items} items: {max_memory_mb:.2f} MB")

class TestAccuracyComparison:
    @pytest.mark.parametrize("true_cardinality", [100, 1_000, 10_000, 100_000, 1_000_000])
    def test_hll_accuracy_our_library(self, true_cardinality):
        """Test accuracy of our HLL implementation."""
        items = [f"item_{i}" for i in range(true_cardinality)]
        hll = sketches.HllSketch(12)
        
        for item in items:
            hll.update(item)
        
        estimate = hll.estimate()
        relative_error = abs(estimate - true_cardinality) / true_cardinality
        
        print(f"Our Library - True: {true_cardinality}, Estimate: {estimate:.0f}, Error: {relative_error:.1%}")
        
        # Standard HLL accuracy: ~1.04/sqrt(2^precision) ≈ 1.6% for precision=12
        assert relative_error < 0.05  # 5% tolerance

    @pytest.mark.skipif(not HAS_APACHE, reason="Apache DataSketches not available")
    @pytest.mark.parametrize("true_cardinality", [100, 1_000, 10_000, 100_000, 1_000_000])
    def test_hll_accuracy_apache(self, true_cardinality):
        """Test accuracy of Apache DataSketches."""
        items = [f"item_{i}" for i in range(true_cardinality)]
        hll = datasketches.hll_sketch(12)
        
        for item in items:
            hll.update(item)
        
        estimate = hll.get_estimate()
        relative_error = abs(estimate - true_cardinality) / true_cardinality
        
        print(f"Apache DataSketches - True: {true_cardinality}, Estimate: {estimate:.0f}, Error: {relative_error:.1%}")
        
        assert relative_error < 0.05
```

### 2. Memory Profiling Script

Create `scripts/memory_profile.py`:
```python
#!/usr/bin/env python3
"""
Detailed memory profiling comparison between our library and Apache DataSketches.
"""

import tracemalloc
import gc
import sys
import psutil
import time
from memory_profiler import profile
import sketches

try:
    import datasketches
    HAS_APACHE = True
except ImportError:
    HAS_APACHE = False
    print("Warning: Apache DataSketches not available")

@profile
def profile_our_library(n_items):
    """Profile memory usage of our library."""
    print(f"Profiling our library with {n_items} items...")
    
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create sketch and add items
    hll = sketches.HllSketch(12)
    items = [f"item_{i}" for i in range(n_items)]
    
    for item in items:
        hll.update(item)
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    # Get tracemalloc statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    estimate = hll.estimate()
    error = abs(estimate - n_items) / n_items
    
    print(f"Our Library Results:")
    print(f"  Items: {n_items}")
    print(f"  Estimate: {estimate:.0f} (error: {error:.1%})")
    print(f"  Memory used: {memory_used:.2f} MB")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Memory per item: {memory_used * 1024 * 1024 / n_items:.1f} bytes")
    
    return memory_used, estimate, error

@profile
def profile_apache_library(n_items):
    """Profile memory usage of Apache DataSketches."""
    if not HAS_APACHE:
        print("Apache DataSketches not available")
        return None, None, None
        
    print(f"Profiling Apache DataSketches with {n_items} items...")
    
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create sketch and add items
    hll = datasketches.hll_sketch(12)
    items = [f"item_{i}" for i in range(n_items)]
    
    for item in items:
        hll.update(item)
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    # Get tracemalloc statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    estimate = hll.get_estimate()
    error = abs(estimate - n_items) / n_items
    
    print(f"Apache DataSketches Results:")
    print(f"  Items: {n_items}")
    print(f"  Estimate: {estimate:.0f} (error: {error:.1%})")
    print(f"  Memory used: {memory_used:.2f} MB")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Memory per item: {memory_used * 1024 * 1024 / n_items:.1f} bytes")
    
    return memory_used, estimate, error

def main():
    test_sizes = [1_000, 10_000, 100_000, 1_000_000]
    results = {"our_library": {}, "apache": {}}
    
    for n_items in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_items:,} items")
        print(f"{'='*60}")
        
        # Test our library
        gc.collect()  # Clean up before test
        our_memory, our_estimate, our_error = profile_our_library(n_items)
        results["our_library"][n_items] = {
            "memory_mb": our_memory,
            "estimate": our_estimate,
            "error": our_error
        }
        
        print("\n" + "-"*40)
        
        # Test Apache DataSketches
        gc.collect()  # Clean up before test
        apache_memory, apache_estimate, apache_error = profile_apache_library(n_items)
        if apache_memory is not None:
            results["apache"][n_items] = {
                "memory_mb": apache_memory,
                "estimate": apache_estimate,
                "error": apache_error
            }
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Items':<12} {'Our Memory':<12} {'Apache Memory':<15} {'Memory Ratio':<12} {'Our Error':<10} {'Apache Error':<12}")
    print("-" * 80)
    
    for n_items in test_sizes:
        our_data = results["our_library"][n_items]
        apache_data = results["apache"].get(n_items)
        
        if apache_data:
            memory_ratio = our_data["memory_mb"] / apache_data["memory_mb"]
            print(f"{n_items:<12,} {our_data['memory_mb']:<12.2f} {apache_data['memory_mb']:<15.2f} {memory_ratio:<12.1f}x {our_data['error']:<10.1%} {apache_data['error']:<12.1%}")
        else:
            print(f"{n_items:<12,} {our_data['memory_mb']:<12.2f} {'N/A':<15} {'N/A':<12} {our_data['error']:<10.1%} {'N/A':<12}")

if __name__ == "__main__":
    main()
```

---

## 🚀 Continuous Integration Benchmarks

### GitHub Actions Workflow

Create `.github/workflows/performance.yml`:
```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run weekly performance regression tests
    - cron: '0 0 * * 0'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust-version: [stable]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: ${{ matrix.rust-version }}
        profile: minimal
        override: true
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-benchmark memory-profiler
        pip install datasketches || echo "Apache DataSketches not available"
        
    - name: Build optimized version
      run: cargo build --release --features optimized
      
    - name: Run Rust benchmarks
      run: |
        cargo bench --bench core_operations -- --output-format json | tee benchmark_results.json
        
    - name: Run Python benchmarks
      run: |
        pytest tests/test_performance.py --benchmark-json=python_benchmark_results.json
        
    - name: Compare against baseline
      run: |
        python scripts/compare_benchmarks.py benchmark_results.json python_benchmark_results.json
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ github.sha }}
        path: |
          benchmark_results.json
          python_benchmark_results.json
          
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = fs.readFileSync('python_benchmark_results.json', 'utf8');
          const data = JSON.parse(results);
          
          let comment = '## 📊 Performance Benchmark Results\n\n';
          comment += '| Test | Time | Memory | Improvement |\n';
          comment += '|------|------|--------|-----------|\n';
          
          // Format benchmark results for comment
          // (Implementation depends on benchmark output format)
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

---

## 📈 Performance Tracking Dashboard

### Benchmark Results Analysis

Create `scripts/analyze_performance.py`:
```python
#!/usr/bin/env python3
"""
Analyze benchmark results and generate performance tracking reports.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def load_benchmark_data(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_performance_report(results_file, output_dir="reports"):
    """Generate comprehensive performance report."""
    
    # Load data
    data = load_benchmark_data(results_file)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Throughput comparison
    axes[0, 0].set_title('Throughput Comparison')
    # Plot throughput data
    
    # Memory usage comparison  
    axes[0, 1].set_title('Memory Usage Comparison')
    # Plot memory data
    
    # Accuracy comparison
    axes[1, 0].set_title('Accuracy Comparison')
    # Plot accuracy data
    
    # Performance over time
    axes[1, 1].set_title('Performance Trend')
    # Plot trend data
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_report_{datetime.now().strftime('%Y%m%d')}.png")
    
    # Generate markdown report
    report = f"""
# Performance Report - {datetime.now().strftime('%Y-%m-%d')}

## Summary

### Throughput Results
- Our Library: X.X M items/sec
- Apache DataSketches: Y.Y M items/sec  
- Ratio: Z.Z (ours vs Apache)

### Memory Results
- Our Library: X.X MB
- Apache DataSketches: Y.Y MB
- Ratio: Z.Z (ours vs Apache)

### Accuracy Results
- Our Library: X.X% error
- Apache DataSketches: Y.Y% error

## Detailed Results

[Include detailed tables and analysis]

## Recommendations

[Based on results, what should be prioritized next]
"""
    
    with open(f"{output_dir}/performance_report_{datetime.now().strftime('%Y%m%d')}.md", 'w') as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze performance benchmark results')
    parser.add_argument('results_file', help='JSON file with benchmark results')
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    
    args = parser.parse_args()
    create_performance_report(args.results_file, args.output_dir)
```

---

## 🎯 Success Criteria

### Performance Targets:
- **Throughput**: 2-3x faster than Apache DataSketches
- **Memory**: 20% more efficient than Apache DataSketches  
- **Accuracy**: Match or exceed Apache DataSketches (<1% error)

### Benchmarking Standards:
- All benchmarks run on consistent hardware
- Multiple runs with statistical significance
- Both synthetic and real-world datasets
- Cross-platform validation (x86_64, ARM64)

This comprehensive benchmarking infrastructure ensures that every optimization claim is backed by solid data and that we can track our progress toward surpassing Apache DataSketches performance.