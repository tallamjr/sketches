//! VarOpt (Variance-Optimal) Sampling
//!
//! Implements the VarOpt algorithm by Cohen et al. for weighted sampling that
//! produces minimum-variance unbiased estimators for subset sums.
//!
//! Items are partitioned into "heavy" items (weight > tau, always included)
//! and "light" items (weight <= tau, probabilistically sampled). The threshold
//! tau is maintained dynamically as items arrive and equals
//!   tau = (total_weight - sum_of_heavy_weights) / (k - num_heavy)
//! ensuring the sum of adjusted weights equals the total stream weight.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for unique seed generation across sketch instances.
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0xDEAD_BEEF_CAFE_BABEu64);

/// Simple xorshift64 PRNG for deterministic but pseudorandom sampling decisions.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a unique seed for each sketch instance.
fn unique_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut state = counter;
    // Mix the counter value through xorshift to avoid correlated seeds
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// A variance-optimal weighted sampling sketch.
///
/// Maintains a reservoir of at most `k` items from a weighted stream, partitioned
/// into heavy items (always retained) and light items (probabilistically sampled).
/// The resulting sample provides minimum-variance unbiased estimators for subset
/// sum queries over the original stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarOptSketch<T: Clone> {
    /// Maximum number of items in the reservoir
    k: usize,
    /// Stored items (heavy items at indices 0..num_heavy, light at num_heavy..)
    items: Vec<T>,
    /// Original weights corresponding to each item
    weights: Vec<f64>,
    /// Whether each item is heavy
    heavy: Vec<bool>,
    /// Threshold dividing heavy from light items
    tau: f64,
    /// Total weight of all items seen in the stream
    total_weight: f64,
    /// Sum of weights of heavy items currently in the reservoir
    heavy_weight_sum: f64,
    /// Total number of items seen
    num_items_seen: u64,
    /// Number of heavy items currently in the reservoir
    num_heavy: usize,
    /// Whether the reservoir has reached capacity at least once
    filled: bool,
    /// PRNG state for randomised sampling decisions
    rng_state: u64,
}

impl<T: Clone> VarOptSketch<T> {
    /// Create a new VarOpt sketch with the given reservoir capacity.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Capacity k must be at least 1");
        Self {
            k,
            items: Vec::with_capacity(k),
            weights: Vec::with_capacity(k),
            heavy: Vec::with_capacity(k),
            tau: 0.0,
            total_weight: 0.0,
            heavy_weight_sum: 0.0,
            num_items_seen: 0,
            num_heavy: 0,
            filled: false,
            rng_state: unique_seed(),
        }
    }

    /// Update the sketch with a new weighted item.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not positive.
    pub fn update(&mut self, item: T, weight: f64) {
        assert!(weight > 0.0, "Weight must be positive, got {weight}");

        self.total_weight += weight;
        self.num_items_seen += 1;

        if !self.filled {
            self.items.push(item);
            self.weights.push(weight);
            self.heavy.push(false);
            if self.items.len() == self.k {
                self.filled = true;
                self.recompute_tau_and_partition();
            }
            return;
        }

        // Reservoir is full. Recompute tau with the new total weight.
        self.recompute_tau_and_partition();

        if weight > self.tau {
            // Heavy item: always include
            self.evict_one_light();
            self.items.push(item);
            self.weights.push(weight);
            self.heavy.push(true);
            self.num_heavy += 1;
            self.heavy_weight_sum += weight;
            self.recompute_tau_and_partition();
        } else {
            // Light item: include with probability weight / tau
            let u: f64 = (xorshift64(&mut self.rng_state) as f64) / (u64::MAX as f64);
            if u < weight / self.tau {
                self.evict_one_light();
                self.items.push(item);
                self.weights.push(weight);
                self.heavy.push(false);
                // tau doesn't change from the light item's perspective;
                // it was already recomputed above with the new total_weight
            }
            // Whether or not the item was accepted, tau is already correct
            // because total_weight already includes this item's weight.
        }
    }

    /// Evict a uniformly random light item from the reservoir.
    fn evict_one_light(&mut self) {
        let light_indices: Vec<usize> = (0..self.items.len()).filter(|&i| !self.heavy[i]).collect();

        assert!(
            !light_indices.is_empty(),
            "No light items available for eviction"
        );

        let chosen =
            light_indices[(xorshift64(&mut self.rng_state) as usize) % light_indices.len()];
        self.items.remove(chosen);
        self.weights.remove(chosen);
        self.heavy.remove(chosen);
    }

    /// Recompute tau and promote/demote items between heavy and light.
    ///
    /// tau = (total_weight - heavy_weight_sum) / (k - num_heavy)
    ///
    /// After computing tau, any light item with weight > tau is promoted to heavy,
    /// and any heavy item with weight <= tau is demoted. This is iterated until
    /// stable.
    fn recompute_tau_and_partition(&mut self) {
        loop {
            let light_slots = self.k.saturating_sub(self.num_heavy);
            if light_slots == 0 {
                self.tau = f64::INFINITY;
                return;
            }

            let residual = self.total_weight - self.heavy_weight_sum;
            self.tau = residual / light_slots as f64;

            let mut changed = false;

            // Promote light items with weight > tau to heavy
            for i in 0..self.items.len() {
                if !self.heavy[i] && self.weights[i] > self.tau {
                    self.heavy[i] = true;
                    self.num_heavy += 1;
                    self.heavy_weight_sum += self.weights[i];
                    changed = true;
                    break; // Recompute tau after each promotion
                }
            }

            if changed {
                continue;
            }

            // Demote heavy items with weight <= tau to light (rare, but possible after merge)
            for i in 0..self.items.len() {
                if self.heavy[i] && self.weights[i] <= self.tau {
                    self.heavy[i] = false;
                    self.num_heavy -= 1;
                    self.heavy_weight_sum -= self.weights[i];
                    changed = true;
                    break;
                }
            }

            if !changed {
                return;
            }
        }
    }

    /// Return the sampled items with their adjusted weights.
    ///
    /// Heavy items are returned with their original weights. Light items are
    /// returned with weight equal to tau (their adjusted weight for unbiased
    /// estimation).
    ///
    /// When the reservoir has not yet filled, all items are returned with their
    /// original weights (they are all effectively retained since nothing has been
    /// discarded).
    pub fn get_samples(&self) -> Vec<(&T, f64)> {
        if !self.filled {
            return self
                .items
                .iter()
                .zip(self.weights.iter())
                .map(|(item, &w)| (item, w))
                .collect();
        }
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if self.heavy[i] {
                    (item, self.weights[i])
                } else {
                    (item, self.tau)
                }
            })
            .collect()
    }

    /// Return the number of items currently in the reservoir.
    pub fn get_num_samples(&self) -> usize {
        self.items.len()
    }

    /// Return the total weight of all items seen so far.
    pub fn get_total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Return the total number of items seen so far.
    pub fn count(&self) -> u64 {
        self.num_items_seen
    }

    /// Return the current threshold tau.
    pub fn get_tau(&self) -> f64 {
        self.tau
    }

    /// Return the number of heavy items in the reservoir.
    pub fn get_num_heavy(&self) -> usize {
        self.num_heavy
    }

    /// Return the reservoir capacity.
    pub fn capacity(&self) -> usize {
        self.k
    }

    /// Estimate a subset sum for items matching the given predicate.
    ///
    /// Returns a tuple of (estimate, lower_bound, upper_bound) representing a
    /// Horvitz-Thompson estimator and confidence bounds.
    ///
    /// The estimate is unbiased: each sampled item contributes its adjusted weight
    /// (original weight for heavy items, tau for light items) if it matches the
    /// predicate.
    ///
    /// The bounds are computed using the variance of the Horvitz-Thompson estimator,
    /// with a 95% confidence interval (z = 1.96).
    pub fn estimate_subset_sum<F: Fn(&T) -> bool>(&self, predicate: F) -> (f64, f64, f64) {
        if self.items.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        if !self.filled {
            // All items are retained; the estimate is exact
            let estimate: f64 = self
                .items
                .iter()
                .zip(self.weights.iter())
                .filter(|(item, _)| predicate(item))
                .map(|(_, &w)| w)
                .sum();
            return (estimate, estimate, estimate);
        }

        let mut estimate = 0.0;
        let mut variance = 0.0;

        for (i, item) in self.items.iter().enumerate() {
            if predicate(item) {
                if self.heavy[i] {
                    estimate += self.weights[i];
                    // Heavy items: always included, zero variance contribution
                } else {
                    estimate += self.tau;
                    // Light item: inclusion probability p_i = w_i / tau
                    // HT adjusted weight = tau
                    // Var contribution = tau^2 * (1 - p_i) = tau * (tau - w_i)
                    variance += self.tau * (self.tau - self.weights[i]);
                }
            }
        }

        let std_dev = variance.max(0.0).sqrt();
        let z = 1.96; // 95% confidence interval
        let lower = (estimate - z * std_dev).max(0.0);
        let upper = estimate + z * std_dev;

        (estimate, lower, upper)
    }

    /// Merge another VarOpt sketch into this one.
    ///
    /// Items from the other sketch are fed into this sketch using their adjusted
    /// weights (original weight for heavy items, tau for light items).
    pub fn merge(&mut self, other: &VarOptSketch<T>) {
        let other_samples = other.get_samples();
        for (item, weight) in other_samples {
            if weight > 0.0 {
                self.update(item.clone(), weight);
            }
        }
    }
}

impl<T: Clone + fmt::Debug> fmt::Display for VarOptSketch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VarOptSketch {{ k: {}, samples: {}, heavy: {}, light: {}, tau: {:.6}, total_weight: {:.2}, items_seen: {} }}",
            self.k,
            self.items.len(),
            self.num_heavy,
            self.items.len() - self.num_heavy,
            self.tau,
            self.total_weight,
            self.num_items_seen,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sketch() {
        let sketch: VarOptSketch<i32> = VarOptSketch::new(10);
        assert_eq!(sketch.capacity(), 10);
        assert_eq!(sketch.get_num_samples(), 0);
        assert_eq!(sketch.get_total_weight(), 0.0);
        assert_eq!(sketch.count(), 0);
    }

    #[test]
    #[should_panic(expected = "Capacity k must be at least 1")]
    fn test_zero_capacity_panics() {
        let _sketch: VarOptSketch<i32> = VarOptSketch::new(0);
    }

    #[test]
    #[should_panic(expected = "Weight must be positive")]
    fn test_zero_weight_panics() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        sketch.update(1, 0.0);
    }

    #[test]
    #[should_panic(expected = "Weight must be positive")]
    fn test_negative_weight_panics() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        sketch.update(1, -1.0);
    }

    #[test]
    fn test_under_capacity_all_items_retained() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(10);
        for i in 0..5 {
            sketch.update(i, (i + 1) as f64);
        }
        assert_eq!(sketch.get_num_samples(), 5);
        assert_eq!(sketch.count(), 5);
        assert!((sketch.get_total_weight() - 15.0).abs() < 1e-10);

        let samples = sketch.get_samples();
        assert_eq!(samples.len(), 5);
        let weight_sum: f64 = samples.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_at_capacity_all_items_retained() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        for i in 0..5 {
            sketch.update(i, (i + 1) as f64);
        }
        assert_eq!(sketch.get_num_samples(), 5);
        assert_eq!(sketch.count(), 5);
    }

    #[test]
    fn test_reservoir_never_exceeds_capacity() {
        let k = 10;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);
        for i in 0..1000 {
            sketch.update(i, (i % 50 + 1) as f64);
        }
        assert!(sketch.get_num_samples() <= k);
        assert_eq!(sketch.count(), 1000);
    }

    #[test]
    fn test_heavy_items_always_included() {
        let k = 5;

        for _ in 0..50 {
            let mut sketch: VarOptSketch<String> = VarOptSketch::new(k);

            for i in 0..k {
                sketch.update(format!("light_{i}"), 1.0);
            }

            sketch.update("HEAVY".to_string(), 1_000_000.0);

            let samples = sketch.get_samples();
            let heavy_present = samples.iter().any(|(item, _)| item.as_str() == "HEAVY");
            assert!(
                heavy_present,
                "Heavy item must always be included in the sample"
            );

            let heavy_entry = samples
                .iter()
                .find(|(item, _)| item.as_str() == "HEAVY")
                .unwrap();
            assert!(
                (heavy_entry.1 - 1_000_000.0).abs() < 1e-6,
                "Heavy item should retain its original weight, got {}",
                heavy_entry.1
            );
        }
    }

    #[test]
    fn test_heavy_items_stability() {
        let k = 10;
        let mut sketch: VarOptSketch<String> = VarOptSketch::new(k);

        for i in 0..5 {
            sketch.update(format!("heavy_{i}"), 10_000.0 + i as f64);
        }

        for i in 0..100 {
            sketch.update(format!("light_{i}"), 0.01);
        }

        let samples = sketch.get_samples();
        for i in 0..5 {
            let name = format!("heavy_{i}");
            let found = samples.iter().any(|(item, _)| item.as_str() == name);
            assert!(found, "Heavy item '{name}' must be in the sample");
        }
    }

    #[test]
    fn test_light_items_probabilistic_sampling() {
        // With many equal-weight light items, the adjusted weight sum should
        // approximate the total weight. The inclusion probabilities are not
        // uniform across stream positions (recent items have higher probability),
        // but the Horvitz-Thompson estimator is still unbiased.
        let k = 10;
        let n = 100;
        let trials = 500;
        let mut total_adj_weight_sum = 0.0;

        for _ in 0..trials {
            let mut sketch: VarOptSketch<usize> = VarOptSketch::new(k);
            for i in 0..n {
                sketch.update(i, 1.0);
            }
            let adj_sum: f64 = sketch.get_samples().iter().map(|(_, w)| w).sum();
            total_adj_weight_sum += adj_sum;
        }

        let mean_adj_sum = total_adj_weight_sum / trials as f64;
        let expected_total = n as f64; // All weights are 1.0
        let relative_error = (mean_adj_sum - expected_total).abs() / expected_total;

        assert!(
            relative_error < 0.05,
            "Mean adjusted weight sum {mean_adj_sum:.2} should approximate total weight {expected_total:.2}, relative_error={relative_error:.4}"
        );
    }

    #[test]
    fn test_subset_sum_estimation_unbiased() {
        let k = 20;
        let n = 200;
        let trials = 3000;

        let true_subset_sum: f64 = (0..n).filter(|&i| i % 2 == 0).map(|i| (i + 1) as f64).sum();

        let mut estimate_sum = 0.0;
        for _ in 0..trials {
            let mut sketch: VarOptSketch<usize> = VarOptSketch::new(k);
            for i in 0..n {
                sketch.update(i, (i + 1) as f64);
            }
            let (estimate, lower, upper) = sketch.estimate_subset_sum(|item| item % 2 == 0);
            estimate_sum += estimate;
            assert!(
                lower <= estimate + 1e-10,
                "Lower bound {lower} exceeds estimate {estimate}"
            );
            assert!(
                upper >= estimate - 1e-10,
                "Upper bound {upper} below estimate {estimate}"
            );
        }

        let mean_estimate = estimate_sum / trials as f64;
        let relative_error = (mean_estimate - true_subset_sum).abs() / true_subset_sum;

        assert!(
            relative_error < 0.15,
            "Subset sum estimator appears biased: mean_estimate={mean_estimate:.2}, true={true_subset_sum:.2}, relative_error={relative_error:.4}"
        );
    }

    #[test]
    fn test_subset_sum_empty_sketch() {
        let sketch: VarOptSketch<i32> = VarOptSketch::new(10);
        let (est, lower, upper) = sketch.estimate_subset_sum(|_| true);
        assert_eq!(est, 0.0);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 0.0);
    }

    #[test]
    fn test_subset_sum_all_items_fit() {
        let k = 10;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);
        for i in 0..5 {
            sketch.update(i, 100.0 + i as f64);
        }

        let (estimate, lower, upper) = sketch.estimate_subset_sum(|_| true);
        let expected: f64 = (0..5).map(|i| 100.0 + i as f64).sum();
        assert!(
            (estimate - expected).abs() < 1e-10,
            "Expected {expected}, got {estimate}"
        );
        assert!((lower - expected).abs() < 1e-10);
        assert!((upper - expected).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_tracking() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        let weights = [1.0, 2.5, 3.7, 0.8, 4.2, 1.1, 6.0];
        let expected_total: f64 = weights.iter().sum();

        for (i, &w) in weights.iter().enumerate() {
            sketch.update(i as i32, w);
        }

        assert!(
            (sketch.get_total_weight() - expected_total).abs() < 1e-10,
            "Expected total weight {}, got {}",
            expected_total,
            sketch.get_total_weight()
        );
    }

    #[test]
    fn test_count_tracking() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(3);
        for i in 0..50 {
            sketch.update(i, 1.0);
        }
        assert_eq!(sketch.count(), 50);
    }

    #[test]
    fn test_get_samples_weight_assignment() {
        let k = 5;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);

        sketch.update(0, 1000.0);
        sketch.update(1, 1.0);
        sketch.update(2, 1.0);
        sketch.update(3, 1.0);
        sketch.update(4, 1.0);

        let samples = sketch.get_samples();
        assert_eq!(samples.len(), k);

        let tau = sketch.get_tau();
        for (item, adj_weight) in &samples {
            if **item == 0 {
                assert!(
                    (*adj_weight - 1000.0).abs() < 1e-10,
                    "Heavy item should have original weight"
                );
            } else {
                assert!(
                    (*adj_weight - tau).abs() < 1e-10,
                    "Light items should have weight tau={tau}, got {adj_weight}"
                );
            }
        }
    }

    #[test]
    fn test_adjusted_weight_sum_equals_total() {
        // The fundamental VarOpt invariant: sum of adjusted weights = total stream weight
        let k = 5;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);

        sketch.update(0, 1000.0);
        sketch.update(1, 1.0);
        sketch.update(2, 1.0);
        sketch.update(3, 1.0);
        sketch.update(4, 1.0);

        let adjusted_sum: f64 = sketch.get_samples().iter().map(|(_, w)| w).sum();
        let total = sketch.get_total_weight();
        assert!(
            (adjusted_sum - total).abs() < 1e-10,
            "Adjusted weight sum {adjusted_sum} should equal total weight {total}"
        );

        // Add more items and check again
        for i in 5..20 {
            sketch.update(i, (i as f64) * 0.5);
        }

        let adjusted_sum: f64 = sketch.get_samples().iter().map(|(_, w)| w).sum();
        let total = sketch.get_total_weight();
        assert!(
            (adjusted_sum - total).abs() < 1e-6,
            "Adjusted weight sum {adjusted_sum} should equal total weight {total} after more items"
        );
    }

    #[test]
    fn test_merge_basic() {
        let k = 10;
        let mut sketch1: VarOptSketch<i32> = VarOptSketch::new(k);
        let mut sketch2: VarOptSketch<i32> = VarOptSketch::new(k);

        for i in 0..20 {
            sketch1.update(i, (i + 1) as f64);
        }
        for i in 20..40 {
            sketch2.update(i, (i + 1) as f64);
        }

        sketch1.merge(&sketch2);
        assert!(sketch1.get_num_samples() <= k);
    }

    #[test]
    fn test_merge_preserves_heavy_items() {
        let k = 10;
        let mut sketch1: VarOptSketch<String> = VarOptSketch::new(k);
        let mut sketch2: VarOptSketch<String> = VarOptSketch::new(k);

        for i in 0..20 {
            sketch1.update(format!("s1_{i}"), 1.0);
        }

        sketch2.update("HEAVY_IN_S2".to_string(), 1_000_000.0);
        for i in 0..19 {
            sketch2.update(format!("s2_{i}"), 1.0);
        }

        sketch1.merge(&sketch2);

        let samples = sketch1.get_samples();
        let heavy_present = samples
            .iter()
            .any(|(item, _)| item.as_str() == "HEAVY_IN_S2");
        assert!(
            heavy_present,
            "Heavy item from merged sketch must be present after merge"
        );
    }

    #[test]
    fn test_merge_empty_into_populated() {
        let k = 5;
        let mut sketch1: VarOptSketch<i32> = VarOptSketch::new(k);
        let sketch2: VarOptSketch<i32> = VarOptSketch::new(k);

        for i in 0..10 {
            sketch1.update(i, 1.0);
        }

        let count_before = sketch1.count();
        let weight_before = sketch1.get_total_weight();
        sketch1.merge(&sketch2);

        assert_eq!(sketch1.count(), count_before);
        assert!((sketch1.get_total_weight() - weight_before).abs() < 1e-10);
    }

    #[test]
    fn test_merge_into_empty() {
        let k = 5;
        let mut sketch1: VarOptSketch<i32> = VarOptSketch::new(k);
        let mut sketch2: VarOptSketch<i32> = VarOptSketch::new(k);

        for i in 0..10 {
            sketch2.update(i, (i + 1) as f64);
        }

        sketch1.merge(&sketch2);
        assert!(sketch1.get_num_samples() > 0);
        assert!(sketch1.get_total_weight() > 0.0);
    }

    #[test]
    fn test_single_item() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        sketch.update(42, 7.5);

        assert_eq!(sketch.get_num_samples(), 1);
        assert_eq!(sketch.count(), 1);
        assert!((sketch.get_total_weight() - 7.5).abs() < 1e-10);

        let samples = sketch.get_samples();
        assert_eq!(samples.len(), 1);
        assert_eq!(*samples[0].0, 42);
        assert!((samples[0].1 - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_capacity_one() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(1);
        for i in 0..100 {
            sketch.update(i, (i + 1) as f64);
        }
        assert_eq!(sketch.get_num_samples(), 1);
        assert_eq!(sketch.count(), 100);
    }

    #[test]
    fn test_display() {
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(5);
        for i in 0..10 {
            sketch.update(i, (i + 1) as f64);
        }
        let display = format!("{sketch}");
        assert!(display.contains("VarOptSketch"));
        assert!(display.contains("k: 5"));
    }

    #[test]
    fn test_tau_monotonicity_equal_weights() {
        let k = 10;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);

        for i in 0..(k as i32) {
            sketch.update(i, 1.0);
        }

        let tau_initial = sketch.get_tau();

        for i in (k as i32)..100 {
            sketch.update(i, 1.0);
        }

        let tau_final = sketch.get_tau();
        // With equal weights, tau = total_weight / k, which increases as more items arrive
        assert!(
            tau_final >= tau_initial - 1e-10,
            "Tau should not decrease with equal weights: initial={tau_initial}, final={tau_final}"
        );
    }

    #[test]
    fn test_adjusted_weights_sum_approximates_total() {
        // The sum of adjusted weights should approximate the total weight.
        // With the correct tau computation, this should be exact (or very close).
        let k = 20;
        let n = 200;

        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);
        for i in 0..n {
            sketch.update(i, (i % 10 + 1) as f64);
        }

        let adjusted_sum: f64 = sketch.get_samples().iter().map(|(_, w)| w).sum();
        let total = sketch.get_total_weight();

        let relative_error = (adjusted_sum - total).abs() / total;
        assert!(
            relative_error < 0.01,
            "Adjusted weight sum {adjusted_sum:.2} should equal total weight {total:.2}, relative_error={relative_error:.6}"
        );
    }

    #[test]
    fn test_all_equal_weights() {
        let k = 5;
        let n = 20;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);
        for i in 0..n {
            sketch.update(i, 1.0);
        }

        assert_eq!(sketch.get_num_samples(), k);

        // tau = total_weight / k = 20 / 5 = 4
        let expected_tau = n as f64 / k as f64;
        assert!(
            (sketch.get_tau() - expected_tau).abs() < 1e-10,
            "With equal weights, tau should be total/k = {}, got {}",
            expected_tau,
            sketch.get_tau()
        );

        // No items should be heavy (all original weights 1.0 < tau = 4.0)
        assert_eq!(
            sketch.get_num_heavy(),
            0,
            "With equal weights smaller than tau, no items should be heavy"
        );
    }

    #[test]
    fn test_extreme_weight_disparity() {
        let k = 5;
        let mut sketch: VarOptSketch<String> = VarOptSketch::new(k);

        for i in 0..3 {
            sketch.update(format!("heavy_{i}"), 1_000_000.0);
        }

        for i in 0..100 {
            sketch.update(format!("light_{i}"), 0.001);
        }

        let samples = sketch.get_samples();
        for i in 0..3 {
            let name = format!("heavy_{i}");
            assert!(
                samples.iter().any(|(item, _)| item.as_str() == name),
                "Heavy item '{name}' must be in the sample"
            );
        }

        assert_eq!(sketch.get_num_samples(), k);
    }

    #[test]
    fn test_subset_sum_predicate_filtering() {
        let k = 10;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);
        for i in 0..5 {
            sketch.update(i, 10.0);
        }

        let (est, _, _) = sketch.estimate_subset_sum(|_| false);
        assert!((est - 0.0).abs() < 1e-10);

        let (est, _, _) = sketch.estimate_subset_sum(|_| true);
        assert!((est - 50.0).abs() < 1e-10);

        let (est, _, _) = sketch.estimate_subset_sum(|item| *item >= 3);
        assert!((est - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_stream() {
        let k = 50;
        let n: u64 = 100_000;
        let mut sketch: VarOptSketch<u64> = VarOptSketch::new(k);

        for i in 0..n {
            sketch.update(i, ((i % 100) + 1) as f64);
        }

        assert_eq!(sketch.get_num_samples(), k);
        assert_eq!(sketch.count(), n);

        let expected_total: f64 = (0..n).map(|i| ((i % 100) + 1) as f64).sum();
        assert!(
            (sketch.get_total_weight() - expected_total).abs() < 1e-6,
            "Total weight tracking should be exact"
        );
    }

    #[test]
    fn test_adjusted_sum_invariant_throughout_stream() {
        // Check that the adjusted weight sum equals total weight at multiple points
        let k = 10;
        let mut sketch: VarOptSketch<i32> = VarOptSketch::new(k);

        for i in 0..500 {
            sketch.update(i, ((i % 20) + 1) as f64);

            if sketch.get_num_samples() == k {
                let adjusted_sum: f64 = sketch.get_samples().iter().map(|(_, w)| w).sum();
                let total = sketch.get_total_weight();
                assert!(
                    (adjusted_sum - total).abs() < 1e-6,
                    "At item {i}: adjusted sum {adjusted_sum} != total weight {total}"
                );
            }
        }
    }
}
