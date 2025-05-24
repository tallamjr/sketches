use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tpchgen::{Customer, LineItem, Nation, Order, Part, PartSupp, Region, Supplier};

fn main() {
    let scale_factor = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01);
    
    let output_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "tpch_data".to_string());
    
    println!("Generating TPC-H data");
    println!("Scale factor: {}", scale_factor);
    println!("Output directory: {}", output_dir);
    
    // Create output directory
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    
    // Generate each table
    generate_region(&output_dir);
    generate_nation(&output_dir);
    generate_supplier(&output_dir, scale_factor);
    generate_customer(&output_dir, scale_factor);
    generate_part(&output_dir, scale_factor);
    generate_partsupp(&output_dir, scale_factor);
    generate_orders(&output_dir, scale_factor);
    generate_lineitem(&output_dir, scale_factor);
    
    println!("\nGeneration complete!");
    println!("Data files written to: {}/", output_dir);
}

fn generate_region(output_dir: &str) {
    println!("Generating region.csv...");
    let path = Path::new(output_dir).join("region.csv");
    let mut file = File::create(path).expect("Failed to create region.csv");
    
    writeln!(file, "r_regionkey,r_name,r_comment").unwrap();
    
    for region in Region::generator() {
        writeln!(
            file,
            "{},{},\"{}\"",
            region.regionkey,
            region.name,
            region.comment
        ).unwrap();
    }
}

fn generate_nation(output_dir: &str) {
    println!("Generating nation.csv...");
    let path = Path::new(output_dir).join("nation.csv");
    let mut file = File::create(path).expect("Failed to create nation.csv");
    
    writeln!(file, "n_nationkey,n_name,n_regionkey,n_comment").unwrap();
    
    for nation in Nation::generator() {
        writeln!(
            file,
            "{},{},{},\"{}\"",
            nation.nationkey,
            nation.name,
            nation.regionkey,
            nation.comment
        ).unwrap();
    }
}

fn generate_supplier(output_dir: &str, scale_factor: f64) {
    println!("Generating supplier.csv...");
    let path = Path::new(output_dir).join("supplier.csv");
    let mut file = File::create(path).expect("Failed to create supplier.csv");
    
    writeln!(
        file,
        "s_suppkey,s_name,s_address,s_nationkey,s_phone,s_acctbal,s_comment"
    ).unwrap();
    
    let mut count = 0;
    for supplier in Supplier::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},\"{}\",{},{},{},\"{}\"",
            supplier.suppkey,
            supplier.name,
            supplier.address,
            supplier.nationkey,
            supplier.phone,
            supplier.acctbal,
            supplier.comment
        ).unwrap();
        
        count += 1;
        if count % 1000 == 0 {
            print!("  {} suppliers generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} suppliers generated", count);
}

fn generate_customer(output_dir: &str, scale_factor: f64) {
    println!("Generating customer.csv...");
    let path = Path::new(output_dir).join("customer.csv");
    let mut file = File::create(path).expect("Failed to create customer.csv");
    
    writeln!(
        file,
        "c_custkey,c_name,c_address,c_nationkey,c_phone,c_acctbal,c_mktsegment,c_comment"
    ).unwrap();
    
    let mut count = 0;
    for customer in Customer::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},\"{}\",{},{},{},{},\"{}\"",
            customer.custkey,
            customer.name,
            customer.address,
            customer.nationkey,
            customer.phone,
            customer.acctbal,
            customer.mktsegment,
            customer.comment
        ).unwrap();
        
        count += 1;
        if count % 10000 == 0 {
            print!("  {} customers generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} customers generated", count);
}

fn generate_part(output_dir: &str, scale_factor: f64) {
    println!("Generating part.csv...");
    let path = Path::new(output_dir).join("part.csv");
    let mut file = File::create(path).expect("Failed to create part.csv");
    
    writeln!(
        file,
        "p_partkey,p_name,p_mfgr,p_brand,p_type,p_size,p_container,p_retailprice,p_comment"
    ).unwrap();
    
    let mut count = 0;
    for part in Part::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},\"{}\"",
            part.partkey,
            part.name,
            part.mfgr,
            part.brand,
            part.typ,
            part.size,
            part.container,
            part.retailprice,
            part.comment
        ).unwrap();
        
        count += 1;
        if count % 10000 == 0 {
            print!("  {} parts generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} parts generated", count);
}

fn generate_partsupp(output_dir: &str, scale_factor: f64) {
    println!("Generating partsupp.csv...");
    let path = Path::new(output_dir).join("partsupp.csv");
    let mut file = File::create(path).expect("Failed to create partsupp.csv");
    
    writeln!(
        file,
        "ps_partkey,ps_suppkey,ps_availqty,ps_supplycost,ps_comment"
    ).unwrap();
    
    let mut count = 0;
    for partsupp in PartSupp::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},{},{},\"{}\"",
            partsupp.partkey,
            partsupp.suppkey,
            partsupp.availqty,
            partsupp.supplycost,
            partsupp.comment
        ).unwrap();
        
        count += 1;
        if count % 10000 == 0 {
            print!("  {} partsupp records generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} partsupp records generated", count);
}

fn generate_orders(output_dir: &str, scale_factor: f64) {
    println!("Generating orders.csv...");
    let path = Path::new(output_dir).join("orders.csv");
    let mut file = File::create(path).expect("Failed to create orders.csv");
    
    writeln!(
        file,
        "o_orderkey,o_custkey,o_orderstatus,o_totalprice,o_orderdate,o_orderpriority,o_clerk,o_shippriority,o_comment"
    ).unwrap();
    
    let mut count = 0;
    for order in Order::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},\"{}\"",
            order.orderkey,
            order.custkey,
            order.orderstatus,
            order.totalprice,
            order.orderdate,
            order.orderpriority,
            order.clerk,
            order.shippriority,
            order.comment
        ).unwrap();
        
        count += 1;
        if count % 10000 == 0 {
            print!("  {} orders generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} orders generated", count);
}

fn generate_lineitem(output_dir: &str, scale_factor: f64) {
    println!("Generating lineitem.csv...");
    let path = Path::new(output_dir).join("lineitem.csv");
    let mut file = File::create(path).expect("Failed to create lineitem.csv");
    
    writeln!(
        file,
        "l_orderkey,l_partkey,l_suppkey,l_linenumber,l_quantity,l_extendedprice,l_discount,l_tax,l_returnflag,l_linestatus,l_shipdate,l_commitdate,l_receiptdate,l_shipinstruct,l_shipmode,l_comment"
    ).unwrap();
    
    let mut count = 0;
    for lineitem in LineItem::generator(scale_factor, 1, 1) {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\"{}\"",
            lineitem.orderkey,
            lineitem.partkey,
            lineitem.suppkey,
            lineitem.linenumber,
            lineitem.quantity,
            lineitem.extendedprice,
            lineitem.discount,
            lineitem.tax,
            lineitem.returnflag,
            lineitem.linestatus,
            lineitem.shipdate,
            lineitem.commitdate,
            lineitem.receiptdate,
            lineitem.shipinstruct,
            lineitem.shipmode,
            lineitem.comment
        ).unwrap();
        
        count += 1;
        if count % 100000 == 0 {
            print!("  {} line items generated\r", count);
            std::io::stdout().flush().unwrap();
        }
    }
    println!("  {} line items generated", count);
}