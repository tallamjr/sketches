use sketches::{bloom::BloomFilter, cpc::CpcSketch, hll::HllSketch, theta::ThetaSketch};

fn main() {
    println!("Sketches Library Demo");
    println!("====================");

    // HyperLogLog Example
    println!("\n1. HyperLogLog Sketch:");
    let mut hll = HllSketch::new(12); // precision 12

    for i in 0..10000 {
        hll.update(&format!("item_{}", i));
    }

    println!("   Added 10,000 unique items");
    println!("   HLL estimate: {:.0}", hll.estimate());

    // Theta Sketch Example
    println!("\n2. Theta Sketch:");
    let mut theta1 = ThetaSketch::new(4096); // k = 4096
    let mut theta2 = ThetaSketch::new(4096);

    // Add some overlapping data
    for i in 0..5000 {
        theta1.update(&format!("common_{}", i));
        theta2.update(&format!("common_{}", i));
    }

    // Add unique data to each
    for i in 0..3000 {
        theta1.update(&format!("unique1_{}", i));
        theta2.update(&format!("unique2_{}", i));
    }

    println!("   Theta1 estimate: {:.0}", theta1.estimate());
    println!("   Theta2 estimate: {:.0}", theta2.estimate());

    let union = ThetaSketch::union_many(&[&theta1, &theta2], 4096);
    println!("   Union estimate: {:.0}", union.estimate());

    let intersection = ThetaSketch::intersect_many(&theta1, &theta2, 4096);
    println!("   Intersection estimate: {:.0}", intersection.estimate());

    // CPC Sketch Example
    println!("\n3. CPC Sketch:");
    let mut cpc = CpcSketch::new(11); // lg_k = 11

    for i in 0..8000 {
        cpc.update(&format!("cpc_item_{}", i));
    }

    println!("   Added 8,000 unique items");
    println!("   CPC estimate: {:.0}", cpc.estimate());

    // Bloom Filter Example
    println!("\n4. Bloom Filter:");
    let mut bloom = BloomFilter::new(10000, 0.01, false); // capacity, error_rate, use_simd

    // Add items
    for i in 0..5000 {
        bloom.add(&format!("bloom_item_{}", i));
    }

    // Test membership
    let test_item = "bloom_item_1234";
    println!("   Added 5,000 items to Bloom filter");
    println!(
        "   Testing membership for '{}': {}",
        test_item,
        bloom.contains(&test_item)
    );

    let non_existent = "non_existent_item";
    println!(
        "   Testing membership for '{}': {}",
        non_existent,
        bloom.contains(&non_existent)
    );

    println!("\nDemo completed successfully!");
}
