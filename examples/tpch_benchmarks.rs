use sketches::hll::HllSketch;
use sketches::theta::ThetaSketch;
use sketches::cpc::CpcSketch;
use std::time::Instant;
use tpchgen::generators::{Customer, LineItem, Nation, Order, Part, PartSupp, Region, Supplier};

fn main() {
    println!("=== TPC-H Benchmarks with Sketches ===\n");
    
    // Generate TPC-H data at scale factor 0.1 (100MB)
    let scale_factor = 0.1;
    println!("Generating TPC-H data at scale factor {}...", scale_factor);
    
    // Benchmark distinct customer counting
    benchmark_customers(scale_factor);
    
    // Benchmark distinct order counting by date
    benchmark_orders(scale_factor);
    
    // Benchmark supplier-part relationships
    benchmark_partsupp(scale_factor);
}

fn benchmark_customers(scale_factor: f64) {
    println!("\n--- Distinct Customer Analysis ---");
    
    let mut total_customers = HllSketch::new(14);
    let mut customers_by_nation = std::collections::HashMap::new();
    
    let start = Instant::now();
    
    for customer in Customer::generator(scale_factor, 1, 1) {
        // Count total unique customers
        total_customers.update(&customer.custkey.to_string());
        
        // Count customers by nation
        let nation_sketch = customers_by_nation
            .entry(customer.nationkey)
            .or_insert_with(|| HllSketch::new(12));
        nation_sketch.update(&customer.custkey.to_string());
    }
    
    let duration = start.elapsed();
    
    println!("Processing time: {:.2?}", duration);
    println!("Total unique customers: ~{:.0}", total_customers.estimate());
    println!("Nations with customers: {}", customers_by_nation.len());
    
    // Show top nations
    let mut nation_counts: Vec<_> = customers_by_nation
        .iter()
        .map(|(nation, sketch)| (*nation, sketch.estimate()))
        .collect();
    nation_counts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 nations by customer count:");
    for (nation, count) in nation_counts.iter().take(5) {
        println!("  Nation {}: ~{:.0} customers", nation, count);
    }
}

fn benchmark_orders(scale_factor: f64) {
    println!("\n--- Order Analysis by Year ---");
    
    let mut orders_by_year = std::collections::HashMap::new();
    let mut high_priority_orders = ThetaSketch::new(12);
    let mut all_orders = ThetaSketch::new(12);
    
    let start = Instant::now();
    
    for order in Order::generator(scale_factor, 1, 1) {
        let order_key = order.orderkey.to_string();
        
        // Extract year from orderdate
        let year = order.orderdate / 10000; // YYYYMMDD format
        let year_sketch = orders_by_year
            .entry(year)
            .or_insert_with(|| HllSketch::new(12));
        year_sketch.update(&order_key);
        
        // Track high priority orders
        all_orders.update(&order_key);
        if order.orderpriority.starts_with("1-URGENT") || order.orderpriority.starts_with("2-HIGH") {
            high_priority_orders.update(&order_key);
        }
    }
    
    let duration = start.elapsed();
    
    println!("Processing time: {:.2?}", duration);
    println!("Total orders: ~{:.0}", all_orders.estimate());
    println!("High priority orders: ~{:.0}", high_priority_orders.estimate());
    println!("High priority percentage: ~{:.1}%", 
             high_priority_orders.estimate() / all_orders.estimate() * 100.0);
    
    // Show orders by year
    let mut year_counts: Vec<_> = orders_by_year
        .iter()
        .map(|(year, sketch)| (*year, sketch.estimate()))
        .collect();
    year_counts.sort_by_key(|&(year, _)| year);
    
    println!("\nOrders by year:");
    for (year, count) in year_counts {
        println!("  {}: ~{:.0} orders", year, count);
    }
}

fn benchmark_partsupp(scale_factor: f64) {
    println!("\n--- Supplier-Part Relationship Analysis ---");
    
    let mut unique_parts = HllSketch::new(14);
    let mut unique_suppliers = HllSketch::new(14);
    let mut parts_per_supplier = std::collections::HashMap::new();
    
    let start = Instant::now();
    
    for partsupp in PartSupp::generator(scale_factor, 1, 1) {
        unique_parts.update(&partsupp.partkey.to_string());
        unique_suppliers.update(&partsupp.suppkey.to_string());
        
        let supplier_parts = parts_per_supplier
            .entry(partsupp.suppkey)
            .or_insert_with(|| HllSketch::new(10));
        supplier_parts.update(&partsupp.partkey.to_string());
    }
    
    let duration = start.elapsed();
    
    println!("Processing time: {:.2?}", duration);
    println!("Unique parts: ~{:.0}", unique_parts.estimate());
    println!("Unique suppliers: ~{:.0}", unique_suppliers.estimate());
    
    // Calculate average parts per supplier
    let total_part_estimates: f64 = parts_per_supplier
        .values()
        .map(|sketch| sketch.estimate())
        .sum();
    let avg_parts_per_supplier = total_part_estimates / parts_per_supplier.len() as f64;
    
    println!("Average parts per supplier: ~{:.1}", avg_parts_per_supplier);
    
    // Find suppliers with most parts
    let mut supplier_parts: Vec<_> = parts_per_supplier
        .iter()
        .map(|(supplier, sketch)| (*supplier, sketch.estimate()))
        .collect();
    supplier_parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 suppliers by part count:");
    for (supplier, count) in supplier_parts.iter().take(5) {
        println!("  Supplier {}: ~{:.0} parts", supplier, count);
    }
}