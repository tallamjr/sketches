use sketches::cpc::CpcSketch;
use sketches::hll::HllSketch;
use sketches::theta::ThetaSketch;

fn main() {
    println!("=== Sketches Basic Usage Example ===\n");

    // HyperLogLog example
    println!("1. HyperLogLog (HLL) - Cardinality Estimation");
    let mut hll = HllSketch::new(12);

    // Add some elements
    for i in 0..10000 {
        hll.update(&i.to_string());
    }

    println!("   Added 10,000 unique elements");
    println!("   Estimated cardinality: {:.0}", hll.estimate());
    println!("   Memory usage: {} bytes\n", std::mem::size_of_val(&hll));

    // Theta Sketch example
    println!("2. Theta Sketch - Set Operations");
    let mut theta1 = ThetaSketch::new(4096);  // k parameter instead of precision
    let mut theta2 = ThetaSketch::new(4096);

    // Add elements to first sketch
    for i in 0..5000 {
        theta1.update(&format!("user_{}", i));
    }

    // Add elements to second sketch (with overlap)
    for i in 2500..7500 {
        theta2.update(&format!("user_{}", i));
    }

    println!("   Sketch 1: ~5,000 users (0-4999)");
    println!("   Sketch 2: ~5,000 users (2500-7499)");

    let union = theta1.union(&theta2);
    let intersection = theta1.intersect(&theta2);

    println!("   Union estimate: {:.0}", union.estimate());
    println!("   Intersection estimate: {:.0}", intersection.estimate());

    // CPC Sketch example (when properly implemented)
    println!("\n3. CPC Sketch - Compressed Counting");
    let mut cpc = CpcSketch::new(11);  // lg_k parameter

    for i in 0..1000 {
        cpc.update(&format!("item_{}", i));
    }

    println!("   Added 1,000 unique items");
    println!("   Estimated cardinality: {:.0}", cpc.estimate());
}
