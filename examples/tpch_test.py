#!/usr/bin/env python3
"""
TPC-H testing with sketches library

This script demonstrates using sketches with TPC-H data,
allowing for standardised benchmarking and validation.
"""

import sketches
import pandas as pd
import time
import subprocess
import sys
import os
from pathlib import Path


def generate_tpch_data(scale_factor=0.01, output_dir="tpch_data"):
    """Generate TPC-H data using tpchgen-rs via Rust example"""
    print(f"Generating TPC-H data at scale factor {scale_factor}...")
    
    # First, ensure the Rust example is built
    subprocess.run(["cargo", "build", "--example", "tpch_generate"], check=True)
    
    # Run the generator
    subprocess.run([
        "cargo", "run", "--example", "tpch_generate", "--",
        str(scale_factor), output_dir
    ], check=True)
    
    return output_dir


def test_distinct_counting(data_dir):
    """Test distinct counting on customer data"""
    print("\n=== Testing Distinct Customer Counting ===")
    
    customer_file = Path(data_dir) / "customer.csv"
    if not customer_file.exists():
        print(f"Customer file not found: {customer_file}")
        return
    
    # Load customer data
    df = pd.read_csv(customer_file)
    
    # Exact count
    exact_count = df['c_custkey'].nunique()
    print(f"Exact distinct customers: {exact_count}")
    
    # Test different sketch types
    sketches_to_test = [
        ("HLL", sketches.HllSketch(precision=12)),
        ("HLL++", sketches.HllPlusPlusSketch(precision=12)),
        ("Theta", sketches.ThetaSketch(lg_k=12)),
        ("CPC", sketches.CpcSketch(lg_k=12))
    ]
    
    for name, sketch in sketches_to_test:
        start_time = time.time()
        
        # Add all customer keys
        for custkey in df['c_custkey']:
            sketch.update(str(custkey))
        
        estimate = sketch.estimate()
        elapsed = time.time() - start_time
        error = abs(estimate - exact_count) / exact_count * 100
        
        print(f"\n{name} Sketch:")
        print(f"  Estimate: {estimate:.0f}")
        print(f"  Error: {error:.2f}%")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory: {len(sketch.to_bytes())} bytes")


def test_set_operations(data_dir):
    """Test set operations with order data"""
    print("\n=== Testing Set Operations with Orders ===")
    
    orders_file = Path(data_dir) / "orders.csv"
    if not orders_file.exists():
        print(f"Orders file not found: {orders_file}")
        return
    
    # Load order data
    df = pd.read_csv(orders_file)
    
    # Create two sets: high priority and recent orders
    high_priority = sketches.ThetaSketch(lg_k=12)
    recent_orders = sketches.ThetaSketch(lg_k=12)
    
    # High priority orders
    high_priority_df = df[df['o_orderpriority'].isin(['1-URGENT', '2-HIGH'])]
    for orderkey in high_priority_df['o_orderkey']:
        high_priority.update(str(orderkey))
    
    # Recent orders (last 25% by date)
    df_sorted = df.sort_values('o_orderdate')
    cutoff_idx = int(len(df_sorted) * 0.75)
    recent_df = df_sorted.iloc[cutoff_idx:]
    for orderkey in recent_df['o_orderkey']:
        recent_orders.update(str(orderkey))
    
    # Perform set operations
    both = high_priority.intersect(recent_orders)
    either = high_priority.union(recent_orders)
    high_only = high_priority.difference(recent_orders)
    
    print(f"High priority orders: ~{high_priority.estimate():.0f}")
    print(f"Recent orders: ~{recent_orders.estimate():.0f}")
    print(f"Both high priority AND recent: ~{both.estimate():.0f}")
    print(f"Either high priority OR recent: ~{either.estimate():.0f}")
    print(f"High priority but NOT recent: ~{high_only.estimate():.0f}")


def test_streaming_accuracy(data_dir):
    """Test accuracy with different stream sizes"""
    print("\n=== Testing Streaming Accuracy ===")
    
    lineitem_file = Path(data_dir) / "lineitem.csv"
    if not lineitem_file.exists():
        print(f"LineItem file not found: {lineitem_file}")
        return
    
    # Test with different sample sizes
    sample_sizes = [1000, 10000, 50000, 100000]
    
    print("\nTesting distinct part counting accuracy:")
    print("Sample Size | Exact | HLL Est | Error % | Memory")
    print("-" * 55)
    
    for sample_size in sample_sizes:
        # Read only the first N rows
        df_sample = pd.read_csv(lineitem_file, nrows=sample_size)
        
        # Exact count
        exact = df_sample['l_partkey'].nunique()
        
        # HLL estimate
        hll = sketches.HllSketch(precision=12)
        for partkey in df_sample['l_partkey']:
            hll.update(str(partkey))
        
        estimate = hll.estimate()
        error = abs(estimate - exact) / exact * 100 if exact > 0 else 0
        memory = len(hll.to_bytes())
        
        print(f"{sample_size:11,d} | {exact:5d} | {estimate:7.0f} | {error:7.2f} | {memory:6d}")


def main():
    """Main test runner"""
    # Parse command line arguments
    scale_factor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "tpch_data"
    
    # Generate data if it doesn't exist
    if not Path(data_dir).exists():
        generate_tpch_data(scale_factor, data_dir)
    else:
        print(f"Using existing data in {data_dir}/")
    
    # Run tests
    test_distinct_counting(data_dir)
    test_set_operations(data_dir)
    test_streaming_accuracy(data_dir)
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()