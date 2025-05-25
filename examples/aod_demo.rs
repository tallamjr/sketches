//! Array of Doubles (AOD) Sketch demonstration
//!
//! This example showcases the Array of Doubles sketch capabilities:
//! - Cardinality estimation with associated double values  
//! - Statistical analysis of multi-dimensional data
//! - Set operations and sketch merging
//! - Performance characteristics and memory usage

use sketches::aod::{AodConfig, AodSketch};
use std::time::Instant;

fn main() {
    println!("🔢 Array of Doubles Sketch Demonstration");
    println!("========================================\n");

    // 1. Basic Usage - Single Value per Key
    basic_usage_demo();

    // 2. Multi-Value Analytics - Multiple Values per Key
    multi_value_demo();

    // 3. Set Operations
    set_operations_demo();

    // 4. Performance and Sampling
    performance_demo();

    // 5. Real-world Analytics Simulation
    analytics_simulation();
}

fn basic_usage_demo() {
    println!("1️⃣ Basic Usage - Single Value per Key");
    println!("-------------------------------------");

    let mut sketch = AodSketch::with_capacity_and_values(1000, 1);

    // Add users with their scores
    let users = [
        ("user1", 85.5),
        ("user2", 92.1),
        ("user3", 78.9),
        ("user1", 87.2), // Duplicate user - will update
        ("user4", 95.5),
        ("user5", 69.8),
    ];

    for (user, score) in users {
        sketch.update(&user, &[score]).unwrap();
    }

    println!("Unique users: {:.1}", sketch.estimate());
    println!("Total scores sum: {:.2}", sketch.column_sums()[0]);
    println!("Average score: {:.2}", sketch.column_means()[0]);
    println!("Storage entries: {}", sketch.len());
    println!();
}

fn multi_value_demo() {
    println!("2️⃣ Multi-Value Analytics - Customer Metrics");
    println!("-------------------------------------------");

    // Track multiple metrics per customer: [revenue, orders, sessions]
    let mut sketch = AodSketch::with_capacity_and_values(5000, 3);

    let customers = [
        ("customer_001", [1250.50, 5.0, 12.0]), // revenue, orders, sessions
        ("customer_002", [890.25, 3.0, 8.0]),
        ("customer_003", [2100.75, 8.0, 15.0]),
        ("customer_004", [456.80, 2.0, 6.0]),
        ("customer_005", [3200.00, 12.0, 20.0]),
        ("customer_001", [1350.75, 6.0, 14.0]), // Updated metrics
    ];

    for (customer, metrics) in customers {
        sketch.update(&customer, &metrics).unwrap();
    }

    let sums = sketch.column_sums();
    let means = sketch.column_means();

    println!("📊 Customer Analytics Summary:");
    println!("  Unique customers: {:.0}", sketch.estimate());
    println!("  Total revenue: ${:.2}", sums[0]);
    println!("  Total orders: {:.0}", sums[1]);
    println!("  Total sessions: {:.0}", sums[2]);
    println!("  Average revenue per customer: ${:.2}", means[0]);
    println!("  Average orders per customer: {:.1}", means[1]);
    println!("  Average sessions per customer: {:.1}", means[2]);
    println!();
}

fn set_operations_demo() {
    println!("3️⃣ Set Operations - Department Analytics");
    println!("---------------------------------------");

    // Sales department metrics
    let mut sales_sketch = AodSketch::with_capacity_and_values(1000, 2);
    let sales_employees = [
        ("alice", [50000.0, 12.0]), // salary, months_experience
        ("bob", [65000.0, 18.0]),
        ("charlie", [55000.0, 15.0]),
    ];

    for (employee, metrics) in sales_employees {
        sales_sketch.update(&employee, &metrics).unwrap();
    }

    // Engineering department metrics
    let mut eng_sketch = AodSketch::with_capacity_and_values(1000, 2);
    let eng_employees = [
        ("bob", [75000.0, 24.0]), // Bob works in both departments
        ("diana", [85000.0, 36.0]),
        ("eve", [70000.0, 20.0]),
    ];

    for (employee, metrics) in eng_employees {
        eng_sketch.update(&employee, &metrics).unwrap();
    }

    println!("Sales department:");
    println!("  Employees: {:.0}", sales_sketch.estimate());
    println!("  Avg salary: ${:.0}", sales_sketch.column_means()[0]);

    println!("Engineering department:");
    println!("  Employees: {:.0}", eng_sketch.estimate());
    println!("  Avg salary: ${:.0}", eng_sketch.column_means()[0]);

    // Union to get all employees across departments
    let mut combined_sketch = sales_sketch.clone();
    combined_sketch.union(&eng_sketch).unwrap();

    println!("Combined (union):");
    println!(
        "  Total unique employees: {:.0}",
        combined_sketch.estimate()
    );
    println!(
        "  Overall avg salary: ${:.0}",
        combined_sketch.column_means()[0]
    );
    println!();
}

fn performance_demo() {
    println!("4️⃣ Performance and Sampling Behavior");
    println!("------------------------------------");

    let mut sketch = AodSketch::with_capacity_and_values(100, 2); // Small capacity

    let start = Instant::now();

    // Add many items to trigger sampling
    for i in 0..10000 {
        let key = format!("item_{}", i);
        let metrics = [i as f64, (i * 2) as f64];
        sketch.update(&key, &metrics).unwrap();
    }

    let duration = start.elapsed();

    println!("📈 Performance Results:");
    println!("  Items processed: 10,000");
    println!("  Processing time: {:.2?}", duration);
    println!(
        "  Items per second: {:.0}",
        10000.0 / duration.as_secs_f64()
    );
    println!();

    println!("🎯 Sampling Behavior:");
    println!("  Sketch capacity: {}", sketch.config.capacity);
    println!("  Current size: {}", sketch.len());
    println!("  Theta (sampling prob): {:.6}", sketch.theta());
    println!("  Estimated cardinality: {:.0}", sketch.estimate());
    println!(
        "  Estimation error: {:.2}%",
        ((sketch.estimate() - 10000.0) / 10000.0 * 100.0).abs()
    );

    // Confidence bounds
    println!(
        "  95% confidence bounds: [{:.0}, {:.0}]",
        sketch.lower_bound(0.95),
        sketch.upper_bound(0.95)
    );
    println!();
}

fn analytics_simulation() {
    println!("5️⃣ Real-world Analytics Simulation - E-commerce");
    println!("----------------------------------------------");

    // Simulate e-commerce event tracking: [revenue, clicks, conversions, time_spent]
    let mut sketch = AodSketch::with_capacity_and_values(2000, 4);

    let start = Instant::now();

    // Simulate diverse user behavior patterns
    for user_id in 0..5000 {
        let revenue = if user_id % 10 == 0 {
            rand::random::<f64>() * 500.0 // 10% are buyers 
        } else {
            0.0
        };
        let clicks = 1.0 + rand::random::<f64>() * 20.0;
        let conversions = if revenue > 0.0 { 1.0 } else { 0.0 };
        let time_spent = 30.0 + rand::random::<f64>() * 300.0; // 30-330 seconds

        let metrics = [revenue, clicks, conversions, time_spent];
        sketch
            .update(&format!("user_{}", user_id), &metrics)
            .unwrap();
    }

    let processing_time = start.elapsed();
    let sums = sketch.column_sums();
    let means = sketch.column_means();

    println!("🛒 E-commerce Analytics Dashboard:");
    println!("  Processing time: {:.2?}", processing_time);
    println!("  Unique users: {:.0}", sketch.estimate());
    println!("  Total revenue: ${:.2}", sums[0]);
    println!("  Total clicks: {:.0}", sums[1]);
    println!("  Total conversions: {:.0}", sums[2]);
    println!("  Total time spent: {:.1} hours", sums[3] / 3600.0);
    println!();

    println!("📊 Average Metrics per User:");
    println!("  Revenue: ${:.2}", means[0]);
    println!("  Clicks: {:.1}", means[1]);
    println!("  Conversion rate: {:.2}%", means[2] * 100.0);
    println!("  Time spent: {:.1} minutes", means[3] / 60.0);
    println!();

    // Business insights
    let conversion_rate = sums[2] / sketch.estimate() * 100.0;
    let revenue_per_user = sums[0] / sketch.estimate();
    let click_to_conversion = if sums[1] > 0.0 {
        sums[2] / sums[1] * 100.0
    } else {
        0.0
    };

    println!("💡 Business Insights:");
    println!("  Overall conversion rate: {:.2}%", conversion_rate);
    println!("  Revenue per user: ${:.2}", revenue_per_user);
    println!("  Click-to-conversion rate: {:.3}%", click_to_conversion);

    // Memory usage estimation
    let memory_usage = sketch.len() * (8 + 8 * sketch.num_values()); // hash + values
    println!(
        "  Memory usage: {} bytes ({:.1} KB)",
        memory_usage,
        memory_usage as f64 / 1024.0
    );

    println!("\n✨ AOD Sketch enables real-time analytics on massive datasets");
    println!("   with bounded memory and probabilistic accuracy guarantees!");
}
