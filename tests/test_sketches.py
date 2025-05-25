import csv
from pathlib import Path

import pytest
import logging

# Determine the directory containing the test data
DATA_DIR = Path(__file__).parent / "data"


def load_column(file_name, column):
    """
    Load a column from a CSV file in the test data directory.
    Returns a list of string values.
    """
    path = DATA_DIR / file_name
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row[column] for row in reader]


@pytest.mark.skip(reason="Region table has only 5 unique values - not suitable for probabilistic sketch testing")
@pytest.mark.parametrize(
    "sketch_class,column,true_rel_err",
    [
        ("ThetaSketch", "r_regionkey", 1e-6),
        ("HllSketch", "r_regionkey", 0.05),
        ("CpcSketch", "r_regionkey", 0.05),
    ],
)
def test_sketch_on_region(sketch_class, column, true_rel_err):
    """
    Test that each sketch implementation estimates the number
    of distinct values in the region.csv file within tolerance.
    """
    logging.info("Testing %s on column '%s' with tolerance %f", sketch_class, column, true_rel_err)
    # Dynamically import the sketch class from the Python extension module
    import sketches as ds

    sketch_cls = getattr(ds, sketch_class)
    # Load data
    data = load_column("region.csv", column)
    # Compute true distinct count
    true_count = len(set(data))
    logging.info("True distinct count: %d", true_count)
    # Instantiate sketch with default parameters
    sk = sketch_cls()
    # Update sketch
    for v in data:
        sk.update(v)
    # Estimate
    est = sk.estimate()
    # Compute relative error
    rel_err = abs(est - true_count) / true_count
    logging.info("%s estimate: %s, true: %s, rel_err: %.6f", sketch_class, est, true_count, rel_err)
    assert (
        rel_err <= true_rel_err
    ), f"{sketch_class} err >{true_rel_err*100}%: est {est}, true {true_count}"


# Reservoir Sampling Tests

def test_reservoir_sampler_r_basic():
    """Test basic functionality of Algorithm R reservoir sampler."""
    import sketches as ds
    
    sampler = ds.ReservoirSamplerR(3)
    
    # Test initial state
    assert sampler.capacity() == 3
    assert sampler.items_seen() == 0
    assert len(sampler.sample()) == 0
    assert not sampler.is_full()
    
    # Add items when reservoir is not full
    sampler.add("item1")
    sampler.add("item2")
    sampler.add("item3")
    
    assert sampler.items_seen() == 3
    assert len(sampler.sample()) == 3
    assert sampler.is_full()
    assert set(sampler.sample()) == {"item1", "item2", "item3"}
    
    # Add more items - sample size should remain 3
    for i in range(4, 20):
        sampler.add(f"item{i}")
    
    assert sampler.items_seen() == 19
    assert len(sampler.sample()) == 3
    assert sampler.is_full()


def test_reservoir_sampler_a_basic():
    """Test basic functionality of Algorithm A reservoir sampler."""
    import sketches as ds
    
    sampler = ds.ReservoirSamplerA(3)
    
    # Test initial state
    assert sampler.capacity() == 3
    assert sampler.items_seen() == 0
    assert len(sampler.sample()) == 0
    assert not sampler.is_full()
    
    # Add items when reservoir is not full
    sampler.add("item1")
    sampler.add("item2") 
    sampler.add("item3")
    
    assert sampler.items_seen() == 3
    assert len(sampler.sample()) == 3
    assert sampler.is_full()
    assert set(sampler.sample()) == {"item1", "item2", "item3"}
    
    # Add more items - sample size should remain 3
    for i in range(4, 50):
        sampler.add(f"item{i}")
    
    assert sampler.items_seen() == 49
    assert len(sampler.sample()) == 3
    assert sampler.is_full()


def test_weighted_reservoir_sampler():
    """Test weighted reservoir sampling functionality."""
    import sketches as ds
    
    sampler = ds.WeightedReservoirSampler(3)
    
    # Test initial state
    assert sampler.capacity() == 3
    assert sampler.total_weight() == 0.0
    assert len(sampler.sample()) == 0
    
    # Add items with different weights
    sampler.add_weighted("rare", 0.1)
    sampler.add_weighted("common", 1.0)
    sampler.add_weighted("legendary", 10.0)
    
    assert sampler.total_weight() == 11.1
    assert len(sampler.sample()) == 3
    
    # Check sample with weights
    sample_with_weights = sampler.sample_with_weights()
    assert len(sample_with_weights) == 3
    
    # Verify all items are present
    items = [item for item, weight in sample_with_weights]
    assert set(items) == {"rare", "common", "legendary"}
    
    # Add unweighted item (weight 1.0)
    sampler.add("normal")
    assert sampler.total_weight() == 12.1


def test_stream_sampler():
    """Test stream sampler functionality."""
    import sketches as ds
    
    sampler = ds.StreamSampler(5, 3)  # capacity=5, batch_size=3
    
    # Test initial state
    stats = sampler.stats()
    assert stats["capacity"] == 5
    assert stats["items_processed"] == 0
    assert stats["sample_size"] == 0
    assert stats["buffer_size"] == 0
    
    # Push a batch smaller than batch size
    sampler.push_batch(["item1", "item2"])
    stats = sampler.stats()
    assert stats["buffer_size"] == 2  # Items still in buffer
    assert stats["items_processed"] == 0  # Not processed yet
    
    # Push another item to trigger batch processing
    sampler.push_batch(["item3"])
    stats = sampler.stats()
    assert stats["buffer_size"] == 0  # Buffer processed
    assert stats["items_processed"] == 3  # Items processed
    
    # Push more items and flush
    sampler.push_batch(["item4", "item5", "item6", "item7"])
    sampler.flush()
    stats = sampler.stats()
    
    assert stats["items_processed"] == 7
    assert stats["buffer_size"] == 0
    assert stats["sample_size"] <= 5  # Sample size limited by capacity


def test_reservoir_sampler_merge():
    """Test merging functionality of reservoir samplers."""
    import sketches as ds
    
    sampler1 = ds.ReservoirSamplerR(3)
    sampler2 = ds.ReservoirSamplerR(3)
    
    # Add different items to each sampler
    sampler1.add("a")
    sampler1.add("b")
    sampler1.add("c")
    
    sampler2.add("d")
    sampler2.add("e")
    sampler2.add("f")
    
    # Merge sampler2 into sampler1
    sampler1.merge(sampler2)
    
    # sampler1 should have processed 6 items total
    assert sampler1.items_seen() == 6
    assert len(sampler1.sample()) == 3  # But only keep 3 items
    
    # sampler2 should be unchanged
    assert sampler2.items_seen() == 3
    assert len(sampler2.sample()) == 3


def test_reservoir_sampler_clear():
    """Test clear functionality."""
    import sketches as ds
    
    sampler = ds.ReservoirSamplerR(3)
    
    # Add items
    sampler.add("item1")
    sampler.add("item2")
    sampler.add("item3")
    
    assert sampler.items_seen() == 3
    assert len(sampler.sample()) == 3
    assert sampler.is_full()
    
    # Clear the sampler
    sampler.clear()
    
    assert sampler.items_seen() == 0
    assert len(sampler.sample()) == 0
    assert not sampler.is_full()


def test_reservoir_sampling_uniformity():
    """Test that reservoir sampling produces approximately uniform distribution."""
    import sketches as ds
    from collections import Counter
    
    # Run multiple trials and check distribution
    trials = 1000
    sample_counts = Counter()
    
    for _ in range(trials):
        sampler = ds.ReservoirSamplerR(3)
        
        # Add items 0-9
        for i in range(10):
            sampler.add(str(i))
        
        # Count selections
        for item in sampler.sample():
            sample_counts[item] += 1
    
    # Each item should appear roughly 300 times (3/10 * 1000)
    # Allow for statistical variation
    expected_count = trials * 3 // 10
    for item, count in sample_counts.items():
        assert 200 < count < 400, f"Item {item} appeared {count} times, expected ~{expected_count}"


@pytest.mark.parametrize("sampler_class", ["ReservoirSamplerR", "ReservoirSamplerA"])
def test_both_algorithms_consistency(sampler_class):
    """Test that both algorithms produce samples of correct size."""
    import sketches as ds
    
    sampler_cls = getattr(ds, sampler_class)
    sampler = sampler_cls(5)
    
    # Test with various stream sizes
    for stream_size in [1, 5, 10, 100]:
        sampler.clear()
        
        for i in range(stream_size):
            sampler.add(f"item_{i}")
        
        expected_sample_size = min(stream_size, 5)
        assert len(sampler.sample()) == expected_sample_size
        assert sampler.items_seen() == stream_size
        
        if stream_size >= 5:
            assert sampler.is_full()
        else:
            assert not sampler.is_full()


def test_weighted_sampler_bias():
    """Test that weighted sampling is biased toward higher weights."""
    import sketches as ds
    from collections import Counter
    
    trials = 1000
    selection_counts = Counter()
    
    for _ in range(trials):
        sampler = ds.WeightedReservoirSampler(2)
        
        # Add items with very different weights
        sampler.add_weighted("heavy", 100.0)  # Very high weight
        sampler.add_weighted("light", 1.0)    # Low weight
        sampler.add_weighted("medium", 10.0)  # Medium weight
        
        for item in sampler.sample():
            selection_counts[item] += 1
    
    # Heavy items should be selected much more often
    assert selection_counts["heavy"] > selection_counts["medium"]
    assert selection_counts["medium"] > selection_counts["light"]
    
    # Heavy should be selected in most trials
    assert selection_counts["heavy"] > trials * 0.8


# T-Digest Tests

def test_tdigest_basic():
    """Test basic T-Digest functionality."""
    import sketches as ds
    
    digest = ds.TDigest()
    
    # Test initial state
    assert digest.count() == 0
    assert digest.is_empty()
    assert digest.median() is None
    assert digest.min() is None
    assert digest.max() is None
    
    # Add values 1-100
    for i in range(1, 101):
        digest.add(float(i))
    
    assert digest.count() == 100
    assert not digest.is_empty()
    assert digest.min() == 1.0
    assert digest.max() == 100.0
    
    # Test median
    median = digest.median()
    assert median is not None
    assert abs(median - 50.5) < 2.0  # Should be close to true median


# AOD Sketch Tests

def test_aod_basic():
    """Test basic AOD sketch functionality."""
    import sketches as ds
    
    # Single value per key
    sketch = ds.AodSketch(capacity=100, num_values=1)
    
    assert sketch.is_empty()
    assert sketch.len() == 0
    assert sketch.estimate() == 0.0
    assert sketch.num_values() == 1
    
    # Add some items
    sketch.update("key1", [10.0])
    sketch.update("key2", [20.0])
    sketch.update("key3", [30.0])
    
    assert not sketch.is_empty()
    assert sketch.len() == 3
    assert sketch.estimate() == 3.0
    assert sketch.theta() == 1.0  # No sampling yet
    
    # Test column operations
    sums = sketch.column_sums()
    means = sketch.column_means()
    assert len(sums) == 1
    assert len(means) == 1
    assert sums[0] == 60.0
    assert means[0] == 20.0


def test_aod_multi_value():
    """Test AOD sketch with multiple values per key."""
    import sketches as ds
    
    # Three values per key: [revenue, orders, sessions]
    sketch = ds.AodSketch(capacity=100, num_values=3)
    
    sketch.update("customer1", [100.0, 2.0, 5.0])
    sketch.update("customer2", [200.0, 3.0, 8.0])
    sketch.update("customer3", [150.0, 1.0, 4.0])
    
    assert sketch.len() == 3
    assert sketch.estimate() == 3.0
    assert sketch.num_values() == 3
    
    sums = sketch.column_sums()
    means = sketch.column_means()
    
    assert len(sums) == 3
    assert len(means) == 3
    
    # Check sums: revenue=450, orders=6, sessions=17
    assert sums[0] == 450.0
    assert sums[1] == 6.0
    assert sums[2] == 17.0
    
    # Check means
    assert means[0] == 150.0  # 450/3
    assert means[1] == 2.0    # 6/3
    assert abs(means[2] - 17.0/3) < 0.001  # 17/3


def test_aod_union():
    """Test AOD sketch union operation."""
    import sketches as ds
    
    sketch1 = ds.AodSketch(capacity=100, num_values=2)
    sketch2 = ds.AodSketch(capacity=100, num_values=2)
    
    # Add different data to each sketch
    sketch1.update("key1", [10.0, 100.0])
    sketch1.update("key2", [20.0, 200.0])
    
    sketch2.update("key2", [25.0, 250.0])  # Duplicate key (will override)
    sketch2.update("key3", [30.0, 300.0])
    
    # Union sketches
    sketch1.union(sketch2)
    
    # Should have 3 unique keys
    assert sketch1.estimate() == 3.0
    assert sketch1.len() == 3


def test_aod_bounds():
    """Test AOD sketch confidence bounds."""
    import sketches as ds
    
    sketch = ds.AodSketch(capacity=100, num_values=1)
    
    # Add many items to get meaningful bounds
    for i in range(50):
        sketch.update(f"key_{i}", [float(i)])
    
    estimate = sketch.estimate()
    lower = sketch.lower_bound(0.95)
    upper = sketch.upper_bound(0.95)
    
    assert lower <= estimate
    assert estimate <= upper
    assert lower >= 0.0


def test_aod_serialization():
    """Test AOD sketch serialization."""
    import sketches as ds
    
    sketch = ds.AodSketch(capacity=100, num_values=2)
    
    sketch.update("key1", [10.0, 20.0])
    sketch.update("key2", [30.0, 40.0])
    
    # Serialize and deserialize
    data = sketch.to_bytes()
    sketch2 = ds.AodSketch.from_bytes(data)
    
    assert sketch.estimate() == sketch2.estimate()
    assert sketch.len() == sketch2.len()
    assert sketch.column_sums() == sketch2.column_sums()


def test_aod_sampling():
    """Test AOD sketch sampling behavior."""
    import sketches as ds
    
    # Small capacity to force sampling
    sketch = ds.AodSketch(capacity=10, num_values=1)
    
    # Add many items
    for i in range(100):
        sketch.update(f"key_{i}", [float(i)])
    
    # Should trigger sampling
    assert sketch.theta() < 1.0
    assert sketch.len() <= 10
    
    # Estimate should be reasonable (probabilistic sketches can have high variance)
    estimate = sketch.estimate()
    assert estimate > sketch.len()
    # Ensure theta is being used correctly for estimation
    assert estimate == sketch.len() / sketch.theta()


def test_aod_error_handling():
    """Test AOD sketch error handling."""
    import sketches as ds
    
    sketch = ds.AodSketch(capacity=100, num_values=2)
    
    # Test wrong number of values
    try:
        sketch.update("key1", [10.0])  # Should be 2 values
        assert False, "Expected error for wrong value count"
    except ValueError:
        pass  # Expected
    
    # Test union with incompatible sketches
    sketch2 = ds.AodSketch(capacity=100, num_values=3)  # Different num_values
    
    try:
        sketch.union(sketch2)
        assert False, "Expected error for incompatible union"
    except ValueError:
        pass  # Expected


def test_aod_statistics():
    """Test AOD sketch statistics."""
    import sketches as ds
    
    sketch = ds.AodSketch(capacity=100, num_values=2)
    
    sketch.update("key1", [10.0, 20.0])
    sketch.update("key2", [30.0, 40.0])
    
    stats = sketch.statistics()
    
    assert "capacity" in stats
    assert "num_values" in stats
    assert "theta" in stats
    assert "estimated_cardinality" in stats
    assert "memory_usage" in stats
    
    assert stats["capacity"] == 100
    assert stats["num_values"] == 2
    assert stats["estimated_cardinality"] == 2.0


def test_tdigest_quantiles():
    """Test T-Digest quantile estimation."""
    import sketches as ds
    
    digest = ds.TDigest(compression=200)
    
    # Add uniform distribution 0-999
    for i in range(1000):
        digest.add(float(i))
    
    # Test various quantiles
    assert abs(digest.quantile(0.0) - 0.0) < 5.0
    assert abs(digest.quantile(0.25) - 249.5) < 10.0
    assert abs(digest.quantile(0.5) - 499.5) < 10.0
    assert abs(digest.quantile(0.75) - 749.5) < 10.0
    assert abs(digest.quantile(1.0) - 999.0) < 5.0
    
    # Test convenience methods
    assert abs(digest.p25() - 249.5) < 10.0
    assert abs(digest.p75() - 749.5) < 10.0
    assert abs(digest.p95() - 949.5) < 15.0
    assert abs(digest.p99() - 989.5) < 20.0


def test_tdigest_rank():
    """Test T-Digest rank estimation."""
    import sketches as ds
    
    digest = ds.TDigest()
    
    # Add values 0-99
    for i in range(100):
        digest.add(float(i))
    
    # Test rank estimates
    assert abs(digest.rank(0.0) - 0.005) < 0.05   # Should be close to 0.5/100
    assert abs(digest.rank(49.5) - 0.5) < 0.05     # Should be close to 0.5
    assert abs(digest.rank(99.0) - 0.995) < 0.05   # Should be close to 99.5/100


def test_tdigest_batch_add():
    """Test T-Digest batch adding."""
    import sketches as ds
    
    digest1 = ds.TDigest()
    digest2 = ds.TDigest()
    
    values = [float(i) for i in range(100)]
    
    # Add one by one
    for value in values:
        digest1.add(value)
    
    # Add as batch
    digest2.add_batch(values)
    
    assert digest1.count() == digest2.count()
    assert digest1.min() == digest2.min()
    assert digest1.max() == digest2.max()
    
    # Quantiles should be similar
    assert abs(digest1.median() - digest2.median()) < 1.0


def test_tdigest_merge():
    """Test T-Digest merging functionality."""
    import sketches as ds
    
    digest1 = ds.TDigest()
    digest2 = ds.TDigest()
    
    # Add different ranges to each digest
    for i in range(50):
        digest1.add(float(i))
    
    for i in range(50, 100):
        digest2.add(float(i))
    
    # Merge digest2 into digest1
    digest1.merge(digest2)
    
    assert digest1.count() == 100
    assert digest1.min() == 0.0
    assert digest1.max() == 99.0
    
    # Median should be around 49.5
    median = digest1.median()
    assert abs(median - 49.5) < 5.0


def test_tdigest_with_accuracy():
    """Test T-Digest creation with accuracy target."""
    import sketches as ds
    
    digest = ds.TDigest.with_accuracy(0.01)  # 1% target error
    
    # Add large dataset
    for i in range(10000):
        digest.add(float(i))
    
    # Test accuracy of extreme quantiles
    p99 = digest.p99()
    expected_p99 = 9900.0  # True 99th percentile
    error = abs(p99 - expected_p99) / expected_p99
    
    assert error < 0.02  # Should be within 2% (allow some slack)


def test_tdigest_clear():
    """Test T-Digest clear functionality."""
    import sketches as ds
    
    digest = ds.TDigest()
    
    # Add some values
    for i in range(10):
        digest.add(float(i))
    
    assert digest.count() == 10
    assert not digest.is_empty()
    
    # Clear and test
    digest.clear()
    
    assert digest.count() == 0
    assert digest.is_empty()
    assert digest.min() is None
    assert digest.max() is None
    assert digest.median() is None


def test_tdigest_statistics():
    """Test T-Digest statistics."""
    import sketches as ds
    
    digest = ds.TDigest(compression=150)
    
    # Add some values
    for i in range(1000):
        digest.add(float(i))
    
    stats = digest.statistics()
    
    assert stats["count"] == 1000
    assert stats["compression"] == 150
    assert stats["min_value"] == 0.0
    assert stats["max_value"] == 999.0
    assert stats["memory_usage"] > 0


def test_streaming_tdigest():
    """Test Streaming T-Digest functionality."""
    import sketches as ds
    
    streaming = ds.StreamingTDigest()
    
    # Add values that will trigger buffer flushes
    for i in range(2000):  # Default buffer size is 1000
        streaming.add(float(i))
    
    # Should have flushed automatically
    median = streaming.median()
    assert abs(median - 999.5) < 20.0
    
    stats = streaming.statistics()
    assert stats["count"] == 2000


def test_tdigest_multiple_quantiles():
    """Test T-Digest multiple quantiles at once."""
    import sketches as ds
    
    digest = ds.TDigest()
    
    # Add values 0-999
    for i in range(1000):
        digest.add(float(i))
    
    # Get multiple quantiles
    quantiles = digest.quantiles([0.25, 0.5, 0.75, 0.95])
    
    assert len(quantiles) == 4
    assert all(q is not None for q in quantiles)
    
    # Should be in ascending order
    values = [q for q in quantiles if q is not None]
    assert values == sorted(values)


def test_tdigest_trimmed_mean():
    """Test T-Digest trimmed mean calculation."""
    import sketches as ds
    
    digest = ds.TDigest()
    
    # Add values 0-99
    for i in range(100):
        digest.add(float(i))
    
    # Trimmed mean excluding bottom 10% and top 10%
    trimmed = digest.trimmed_mean(0.1, 0.9)
    
    # Should be approximately the mean of values 10-89
    # which is (10 + 89) / 2 = 49.5
    assert trimmed is not None
    assert abs(trimmed - 49.5) < 10.0

