use sketches::quantiles::KllSketch;

fn main() {
    println!("=== Quantile Estimation with KLL Sketch ===\n");

    // Basic KLL Sketch Demo
    println!("1. Basic Quantile Estimation");
    println!("{}", "-".repeat(30));

    let mut kll = KllSketch::new(200);
    println!("Created KLL sketch with k=200");

    // Simulate server response times (in milliseconds)
    let response_times = vec![
        10, 12, 15, 8, 20, 25, 30, 18, 22, 35, 40, 45, 28, 32, 50, 60, 55, 42, 38, 65, 70, 75, 80,
        85, 90, 95, 100, 110, 120, 150, 200, 180, 160, 140, 130, 125, 135, 145, 155, 175,
    ];

    for &time in &response_times {
        kll.update(time);
    }

    println!(
        "Processed {} response time measurements",
        response_times.len()
    );

    // Calculate key percentiles
    println!("\nResponse Time Percentiles:");
    println!(
        "  50th percentile (median): {:.1}ms",
        kll.quantile(0.5).unwrap_or(0)
    );
    println!("  90th percentile: {:.1}ms", kll.quantile(0.9).unwrap_or(0));
    println!(
        "  95th percentile: {:.1}ms",
        kll.quantile(0.95).unwrap_or(0)
    );
    println!(
        "  99th percentile: {:.1}ms",
        kll.quantile(0.99).unwrap_or(0)
    );

    // Test specific SLA thresholds
    let sla_threshold = 100;
    let rank_under_sla = kll.rank(&sla_threshold);
    println!("\nSLA Analysis:");
    println!(
        "  Percentage under {}ms: {:.1}%",
        sla_threshold,
        rank_under_sla * 100.0
    );
    println!("  Min response time: {}ms", kll.min().unwrap_or(&0));
    println!("  Max response time: {}ms", kll.max().unwrap_or(&0));

    let stats = kll.statistics();
    println!("\nSketch Statistics:");
    println!("  Memory levels: {}", stats.levels);
    println!("  Total items stored: {}", stats.total_items);
    println!("  Memory usage: {} bytes", stats.memory_usage);

    println!("\n{}", "=".repeat(50));

    // Large Dataset Demo
    println!("\n2. Large Dataset Performance");
    println!("{}", "-".repeat(30));

    let mut large_kll = KllSketch::with_accuracy(0.01, 0.99);
    println!("Created KLL sketch with 1% accuracy, 99% confidence");

    // Simulate a large dataset
    let n_samples = 100000;
    println!("Processing {} samples from normal distribution", n_samples);

    let start = std::time::Instant::now();

    // Simulate normal distribution (mean=50, std=15)
    for i in 0..n_samples {
        // Simple Box-Muller transform for normal distribution
        if i % 2 == 0 {
            let u1 = (i as f64 + 1.0) / (n_samples as f64 + 1.0);
            let u2 = ((i / 2) as f64 + 1.0) / ((n_samples / 2) as f64 + 1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let value = (50.0 + 15.0 * z) as i32;
            large_kll.update(value);
        }
    }

    let duration = start.elapsed();
    println!("Processing completed in {:?}", duration);

    // Analyze the distribution
    println!("\nDistribution Analysis:");
    println!("  Count: {}", large_kll.count());
    println!("  Min: {}", large_kll.min().unwrap_or(&0));
    println!("  Max: {}", large_kll.max().unwrap_or(&0));

    // Compare with theoretical normal distribution percentiles
    let percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    println!("\nQuantile Analysis:");
    println!(
        "{:<10} {:<12} {:<12}",
        "Percentile", "Estimated", "Theoretical"
    );
    println!("{}", "-".repeat(35));

    for &p in &percentiles {
        let estimated = large_kll.quantile(p).unwrap_or(0);
        // Approximate theoretical values for normal(50, 15)
        let z_score = match p {
            0.01 => -2.33,
            0.05 => -1.64,
            0.1 => -1.28,
            0.25 => -0.67,
            0.5 => 0.0,
            0.75 => 0.67,
            0.9 => 1.28,
            0.95 => 1.64,
            0.99 => 2.33,
            _ => 0.0,
        };
        let theoretical = (50.0 + 15.0 * z_score) as i32;

        println!(
            "{:<10} {:<12} {:<12}",
            format!("{:.0}%", p * 100.0),
            estimated,
            theoretical
        );
    }

    let large_stats = large_kll.statistics();
    println!("\nLarge Dataset Sketch Statistics:");
    println!("  Memory levels: {}", large_stats.levels);
    println!(
        "  Items stored: {} (compression ratio: {:.1}x)",
        large_stats.total_items,
        large_stats.total_count as f64 / large_stats.total_items as f64
    );
    println!(
        "  Memory usage: {:.1} KB",
        large_stats.memory_usage as f64 / 1024.0
    );

    println!("\n{}", "=".repeat(50));

    // Sketch Merging Demo
    println!("\n3. Distributed Computing: Sketch Merging");
    println!("{}", "-".repeat(40));

    let mut sketch_east = KllSketch::new(100);
    let mut sketch_west = KllSketch::new(100);

    // Simulate data from different regions
    println!("Simulating data from East and West regions");

    // East region: slightly faster responses
    for i in 0..1000 {
        let response_time = 20 + (i % 80); // 20-100ms range
        sketch_east.update(response_time);
    }

    // West region: slightly slower responses
    for i in 0..1000 {
        let response_time = 30 + (i % 100); // 30-130ms range
        sketch_west.update(response_time);
    }

    println!(
        "East region - Median: {:.1}ms",
        sketch_east.quantile(0.5).unwrap_or(0)
    );
    println!(
        "West region - Median: {:.1}ms",
        sketch_west.quantile(0.5).unwrap_or(0)
    );

    // Merge the sketches for global view
    sketch_east.merge(&mut sketch_west);

    println!("\nAfter merging:");
    println!(
        "Global median: {:.1}ms",
        sketch_east.quantile(0.5).unwrap_or(0)
    );
    println!(
        "Global 95th percentile: {:.1}ms",
        sketch_east.quantile(0.95).unwrap_or(0)
    );
    println!("Total samples: {}", sketch_east.count());

    let merged_stats = sketch_east.statistics();
    println!("Merged sketch levels: {}", merged_stats.levels);

    println!("\n{}", "=".repeat(50));

    // Accuracy Demonstration
    println!("\n4. Accuracy vs Memory Trade-off");
    println!("{}", "-".repeat(35));

    let k_values = [50, 100, 200, 400];
    let test_data: Vec<i32> = (0..10000).collect();

    println!(
        "{:<8} {:<12} {:<12} {:<10}",
        "k value", "Memory (KB)", "Error %", "Items"
    );
    println!("{}", "-".repeat(45));

    for &k in &k_values {
        let mut sketch = KllSketch::new(k);

        // Add all test data
        for &value in &test_data {
            sketch.update(value);
        }

        // Calculate error for median
        let estimated_median = sketch.quantile(0.5).unwrap_or(0);
        let true_median = 4999; // Middle of 0..9999
        let error = ((estimated_median - true_median).abs() as f64 / true_median as f64) * 100.0;

        let stats = sketch.statistics();
        println!(
            "{:<8} {:<12.1} {:<12.2} {:<10}",
            k,
            stats.memory_usage as f64 / 1024.0,
            error,
            stats.total_items
        );
    }

    println!("\n{}", "=".repeat(50));

    // Real-world Use Cases
    println!("\n5. Real-world Applications");
    println!("{}", "-".repeat(25));

    println!("KLL Sketches are ideal for:");
    println!("• Monitoring system latencies and response times");
    println!("• Database query performance analysis");
    println!("• Network monitoring and SLA compliance");
    println!("• IoT sensor data analysis");
    println!("• Financial risk assessment (VaR calculations)");
    println!("• A/B testing statistical analysis");
    println!("• Stream processing with limited memory");

    println!("\nAdvantages:");
    println!("• Constant memory usage regardless of data size");
    println!("• Mergeable for distributed computing");
    println!("• Configurable accuracy vs memory trade-off");
    println!("• Efficient quantile and rank queries");

    println!("\nDemo completed successfully!");
}
