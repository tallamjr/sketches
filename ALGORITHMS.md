# 🧮 Algorithm Deep Dive: Understanding Probabilistic Data Structures

This document provides detailed explanations of all probabilistic data structure algorithms implemented in this library, following the same clear approach as the Algorithm R vs Algorithm A comparison.

## 📊 **Cardinality Estimation Algorithms**

### **HyperLogLog (HLL) vs HyperLogLog++ (HLL++) vs Compressed Probabilistic Counting (CPC)**

#### **HyperLogLog (Basic)**
```rust
// For each new item:
let hash = hash64(item);
let bucket = hash & ((1 << p) - 1);  // First p bits select bucket
let leading_zeros = (hash >> p).leading_zeros() + 1;  // Count zeros in remaining bits
registers[bucket] = max(registers[bucket], leading_zeros);  // Update max rank

// Estimation using harmonic mean:
fn estimate() -> f64 {
    let raw_estimate = ALPHA * m² / sum(2^(-register[i]));
    bias_correction(raw_estimate)  // Apply small-range corrections
}
```

**Characteristics:**
- **Time Complexity**: O(1) per update
- **Space Complexity**: O(2^p) registers, each ~6 bits
- **Accuracy**: ~1.04/√m relative error (m = 2^p)
- **Memory**: Fixed size regardless of cardinality

#### **HyperLogLog++ (Advanced)**
```rust
// Enhanced with sparse representation and 64-bit hashing
if sparse_mode && num_registers_used < m/4 {
    // Sparse mode: store only non-zero registers
    sparse_map[bucket] = max(sparse_map[bucket], leading_zeros);
    if sparse_map.len() > m/2 { convert_to_dense(); }
} else {
    // Dense mode: like basic HLL but with bias correction
    registers[bucket] = max(registers[bucket], leading_zeros);
}

// Better estimation with empirical bias correction
fn estimate() -> f64 {
    if cardinality < 5*m/2 { return linear_counting(); }  // Small range correction
    let raw = harmonic_mean_estimate();
    empirical_bias_correction(raw)  // Large range correction
}
```

**Characteristics:**
- **Time Complexity**: O(1) amortized (sparse→dense conversion)
- **Space Complexity**: O(k) sparse, O(2^p) dense
- **Accuracy**: ~0.5% relative error with bias correction
- **Memory**: Adaptive - small for low cardinality, fixed for high

#### **Compressed Probabilistic Counting (CPC)**
```rust
// Uses compression and table modes for different ranges
enum CpcMode {
    Sparse { map: HashMap<u16, u8> },     // Very low cardinality
    Hybrid { compressed: BitVec },        // Medium cardinality  
    Pinned { table: Vec<u8> },           // High cardinality
}

// Automatic mode transitions based on efficiency
fn update(&mut self, item: &str) {
    let hash = hash64(item);
    let (index, rank) = extract_index_and_rank(hash);
    
    match &mut self.mode {
        Sparse(map) if map.len() < threshold => map.insert(index, rank),
        _ => {
            if should_compress() { self.compress(); }
            self.update_compressed(index, rank);
        }
    }
}
```

**Characteristics:**
- **Time Complexity**: O(1) average, O(n) for compression events
- **Space Complexity**: Best compression of all algorithms
- **Accuracy**: Similar to HLL but smaller serialized size
- **Memory**: Multi-mode adaptive compression

---

## 🎯 **Sampling Algorithms**

### **Reservoir Sampling: Algorithm R vs Algorithm A**

#### **Algorithm R (Basic Reservoir Sampling)**
```rust
// For each new item:
if reservoir.len() < capacity {
    reservoir.push(item)  // Always add if not full
} else {
    random_index = random(0..items_seen)  // Generate random number EVERY time
    if random_index < capacity {
        reservoir[random_index] = item    // Replace with probability k/n
    }
}
```

**Characteristics:**
- **Time Complexity**: O(1) per item, but generates random number for every item
- **Random Calls**: One random number generation per item after reservoir fills
- **Simple**: Straightforward implementation, easy to understand
- **Memory**: O(k) where k is reservoir size

#### **Algorithm A (Vitter's Optimized Algorithm)**
```rust
// Key insight: Skip items probabilistically instead of checking every item
if reservoir.len() < capacity {
    reservoir.push(item)
} else if skip_count > 0 {
    skip_count -= 1  // Skip this item (no random generation!)
} else {
    reservoir[random_index] = item
    compute_next_skip()  // Generate skip count using geometric distribution
}

fn compute_next_skip() {
    let u: f64 = random();
    let w = (-u.ln() / capacity as f64).exp();
    skip_count = ((items_seen + 1.0) * w).floor() - items_seen;
}
```

**Characteristics:**
- **Time Complexity**: O(1) amortized, much fewer random calls
- **Random Calls**: Uses geometric distribution to determine how many items to skip
- **Optimization**: Skips items without random generation, processes in batches
- **Performance**: ~19x faster for large datasets in our implementation

**The Key Insight**: Instead of asking "should I replace this item?" for every single item, Algorithm A asks "how many items should I skip before the next potential replacement?" This dramatically reduces expensive random number generations.

#### **Weighted Reservoir Sampling (A-Res)**
```rust
// Each item gets a key = uniform_random^(1/weight)
// Items with highest keys are kept in the reservoir
fn add_weighted(&mut self, item: T, weight: f64) {
    let u: f64 = random();
    let key = u.powf(1.0 / weight);  // Higher weight = higher expected key
    
    if reservoir.len() < capacity {
        reservoir.push((item, weight, key));
        if reservoir.len() == capacity { reservoir.sort_by_key(|&(_, _, k)| -k); }
    } else if key > reservoir.last().unwrap().2 {  // Better than worst key
        reservoir.pop();
        reservoir.insert_sorted((item, weight, key));
    }
}
```

**Characteristics:**
- **Time Complexity**: O(log k) per item due to sorted insertion
- **Space Complexity**: O(k) for reservoir
- **Weighted**: Probability of selection proportional to item weight
- **Distributed**: Can merge weighted samples correctly

---

## 📈 **Quantile Estimation Algorithms**

### **KLL Sketch vs T-Digest**

#### **KLL Sketch (K-Minimum Values)**
```rust
// Geometric levels structure for compaction
struct KllSketch<T> {
    levels: Vec<Vec<T>>,     // Level i can hold k * 2^i items
    k: usize,                // Base capacity parameter
    compaction_count: u64,   // Tracks compactions for accuracy
}

// Compaction maintains error bounds while reducing size
fn compact_level(&mut self, level: usize) {
    let items = &mut self.levels[level];
    items.sort();  // Sort for accuracy
    
    // Keep every other item (geometric sampling)
    let mut compacted = Vec::new();
    for i in (0..items.len()).step_by(2) {
        compacted.push(items[i]);
    }
    
    self.levels[level] = Vec::new();
    self.levels[level + 1].extend(compacted);  // Promote to next level
}
```

**Characteristics:**
- **Time Complexity**: O(log n) amortized per update
- **Space Complexity**: O(k log(n/k)) where k is accuracy parameter
- **Accuracy**: Provable error bounds ±ε with high probability
- **Merging**: Exact merge operations maintain error bounds

#### **T-Digest**
```rust
// Adaptive compression using centroids
struct TDigest {
    centroids: Vec<(f64, u64)>,  // (mean, weight) pairs
    compression: usize,          // Controls accuracy vs memory trade-off
}

// Merges values into centroids with adaptive compression
fn add(&mut self, value: f64) {
    // Find nearby centroids and merge if compression allows
    let nearby_centroid = self.find_nearest_centroid(value);
    
    if can_merge(nearby_centroid, self.compression) {
        // Merge into existing centroid
        update_centroid(nearby_centroid, value);
    } else {
        // Create new centroid
        self.centroids.push((value, 1));
        if self.centroids.len() > self.compression {
            self.compress();  // Reduce centroids while preserving distribution
        }
    }
}
```

**Characteristics:**
- **Time Complexity**: O(log C) per update where C is compression
- **Space Complexity**: O(C) centroids
- **Accuracy**: Better for extreme quantiles (p95, p99) than uniform quantiles
- **Streaming**: Excellent for streaming data with adaptive compression

**Key Difference**: KLL provides **provable error bounds** and exact merging, while T-Digest provides **better practical accuracy** for extreme quantiles with adaptive compression.

---

## 🏗️ **Set Operations & Membership Testing**

### **Bloom Filter vs Counting Bloom Filter**

#### **Standard Bloom Filter**
```rust
// Multiple hash functions map to bit array
fn add(&mut self, item: &str) {
    for i in 0..self.num_hash_functions {
        let hash = hash_i(item, i) % self.num_bits;
        self.bit_array.set(hash, true);  // Set bit to 1
    }
}

fn contains(&self, item: &str) -> bool {
    for i in 0..self.num_hash_functions {
        let hash = hash_i(item, i) % self.num_bits;
        if !self.bit_array.get(hash) { return false; }  // Any 0 bit = definitely not present
    }
    true  // All bits set = probably present
}
```

**Characteristics:**
- **Time Complexity**: O(k) where k is number of hash functions
- **Space Complexity**: O(m) bits where m = -n ln(p) / (ln(2))²
- **False Positives**: Possible (probability p)
- **False Negatives**: Impossible
- **Deletion**: Not supported

#### **Counting Bloom Filter**
```rust
// Counters instead of bits allow deletion
fn add(&mut self, item: &str) {
    for i in 0..self.num_hash_functions {
        let hash = hash_i(item, i) % self.num_counters;
        self.counters[hash] = self.counters[hash].saturating_add(1);  // Increment counter
    }
}

fn remove(&mut self, item: &str) -> bool {
    // First check if item is present
    if !self.contains(item) { return false; }
    
    // Decrement all counters
    for i in 0..self.num_hash_functions {
        let hash = hash_i(item, i) % self.num_counters;
        if self.counters[hash] > 0 {
            self.counters[hash] -= 1;
        }
    }
    true
}
```

**Characteristics:**
- **Time Complexity**: O(k) for add/remove/contains
- **Space Complexity**: O(m × counter_bits) - larger than standard Bloom
- **False Positives**: Possible but reduced over time with deletions
- **False Negatives**: Impossible
- **Deletion**: Supported with counter overflow protection

---

## 📊 **Frequency Estimation Algorithms**

### **Count-Min Sketch vs Count Sketch vs Frequent Items**

#### **Count-Min Sketch**
```rust
// Conservative frequency estimation using minimum
struct CountMinSketch {
    table: Vec<Vec<u64>>,    // depth × width matrix
    depth: usize,            // Number of hash functions
    width: usize,            // Counter array size per row
}

fn update(&mut self, item: &str, count: u64) {
    for row in 0..self.depth {
        let hash = hash_function(item, row) % self.width;
        self.table[row][hash] += count;  // Increment counter in each row
    }
}

fn estimate(&self, item: &str) -> u64 {
    let mut min_count = u64::MAX;
    for row in 0..self.depth {
        let hash = hash_function(item, row) % self.width;
        min_count = min_count.min(self.table[row][hash]);  // Take minimum
    }
    min_count
}
```

**Characteristics:**
- **Time Complexity**: O(d) per update/query where d is depth
- **Space Complexity**: O(d × w) counters
- **Accuracy**: ε-δ guarantees with probability 1-δ
- **Bias**: Always overestimates (due to hash collisions)
- **Heavy Hitters**: Good for finding frequent items

#### **Count Sketch** 
```rust
// Uses signed updates to cancel out noise
struct CountSketch {
    table: Vec<Vec<i64>>,    // Signed counters
    hash_functions: Vec<HashFunction>,
    sign_functions: Vec<SignFunction>,  // Random ±1 for each position
}

fn update(&mut self, item: &str, count: i64) {
    for row in 0..self.depth {
        let hash = self.hash_functions[row](item) % self.width;
        let sign = self.sign_functions[row](item);  // ±1
        self.table[row][hash] += sign * count;  // Signed update
    }
}

fn estimate(&self, item: &str) -> i64 {
    let mut estimates = Vec::new();
    for row in 0..self.depth {
        let hash = self.hash_functions[row](item) % self.width;
        let sign = self.sign_functions[row](item);
        estimates.push(sign * self.table[row][hash]);
    }
    median(estimates)  // Take median instead of minimum
}
```

**Characteristics:**
- **Time Complexity**: O(d) per update/query
- **Space Complexity**: O(d × w) signed counters
- **Accuracy**: Unbiased estimate (can over/under estimate)
- **Noise Cancellation**: Random signs help cancel hash collision noise
- **Point Queries**: Better for single-item frequency queries

#### **Frequent Items Sketch (Space-Saving)**
```rust
// Tracks top-k heavy hitters with error bounds
struct FrequentItemsSketch {
    counters: HashMap<String, (u64, u64)>,  // item -> (frequency, error)
    min_counter: u64,
    capacity: usize,
}

fn update(&mut self, item: &str) {
    if let Some((freq, error)) = self.counters.get_mut(item) {
        *freq += 1;  // Item already tracked
    } else if self.counters.len() < self.capacity {
        self.counters.insert(item.to_string(), (1, 0));  // Add new item
    } else {
        // Replace minimum item (Space-Saving guarantee)
        let (min_item, (min_freq, _)) = self.find_minimum();
        self.counters.remove(&min_item);
        self.counters.insert(item.to_string(), (min_freq + 1, min_freq));
    }
}
```

**Characteristics:**
- **Time Complexity**: O(1) average for updates, O(k) for minimum finding
- **Space Complexity**: O(k) for tracking k frequent items
- **Accuracy**: ε-approximation with error bounds
- **Heavy Hitters**: Specifically designed for top-k queries
- **Error Bounds**: Provides lower and upper bounds for frequencies

---

## 🎛️ **Algorithm Selection Guide**

| **Use Case** | **Recommended Algorithm** | **Why** |
|--------------|---------------------------|---------|
| **Cardinality Estimation** | HLL++ | Best accuracy-memory trade-off, industry standard |
| **Small Cardinalities** | Linear Counter → HLL | Better accuracy for n < 1000, auto-transition |
| **Set Operations** | Theta Sketch | Only algorithm supporting union/intersection |
| **Compressed Storage** | CPC | Smallest serialized size, good for network transfer |
| **Uniform Sampling** | Algorithm A | 19x faster than Algorithm R for large streams |
| **Weighted Sampling** | A-Res | Correct probability-proportional sampling |
| **Quantiles (Streaming)** | T-Digest | Better for extreme quantiles (p95, p99) |
| **Quantiles (Merging)** | KLL Sketch | Exact merge with provable error bounds |
| **Membership Testing** | Bloom Filter | Fastest, smallest for membership-only |
| **Membership + Deletion** | Counting Bloom | Supports removal operations |
| **Frequency Estimation** | Count-Min | Conservative estimates, good for heavy hitters |
| **Point Queries** | Count Sketch | Unbiased estimates with median |
| **Top-K Items** | Frequent Items | Specifically designed for heavy hitters |

## 🔬 **Performance Characteristics Summary**

| Algorithm | Time/Update | Space | Accuracy | Special Properties |
|-----------|-------------|-------|----------|-------------------|
| **HLL** | O(1) | O(2^p) | ~1% | Fixed memory |
| **HLL++** | O(1) amortized | O(k) → O(2^p) | ~0.5% | Adaptive memory |
| **CPC** | O(1) average | Minimal | ~1% | Best compression |
| **Linear Counter** | O(1) | O(m) | <1% for small n | Exact for small sets |
| **Algorithm R** | O(1) | O(k) | Exact | Simple uniform sampling |
| **Algorithm A** | O(1) | O(k) | Exact | 19x faster than R |
| **T-Digest** | O(log C) | O(C) | <1% | Great for extremes |
| **KLL** | O(log n) | O(k log n) | ±ε provable | Exact merging |
| **Bloom Filter** | O(h) | O(m) | p false positive | No false negatives |
| **Count-Min** | O(d) | O(d×w) | ε-δ bounds | Always overestimates |
| **Count Sketch** | O(d) | O(d×w) | Unbiased | Better point queries |

This comprehensive comparison should help users understand the trade-offs and choose the right algorithm for their specific use case!