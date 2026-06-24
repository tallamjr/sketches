//! Shared dataset contract for the benchmark runners.
//!
//! Every benchmark runner (ours, Apache-Rust, Apache-C++) feeds from the same
//! data sources so that throughput and accuracy numbers are comparable. This
//! crate provides synthetic streams with known exact cardinality, an exact
//! distinct-count ground truth helper, and a TPC-H CSV column reader.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Synthetic stream of `n` distinct u64 values (0..n). Exact cardinality = n.
pub fn synthetic_distinct(n: u64) -> impl Iterator<Item = u64> {
    0..n
}

/// Exact distinct count of an iterator of owned strings (ground truth).
pub fn exact_distinct<I: IntoIterator<Item = String>>(it: I) -> u64 {
    it.into_iter().collect::<HashSet<String>>().len() as u64
}

/// Inspect the first line of a file to choose the field delimiter.
///
/// TPC-H data is commonly pipe-delimited even when stored with a `.csv`
/// extension, whereas plain CSV exports use commas. We pick whichever of the
/// two characters appears more often on the first line, defaulting to comma
/// when neither is present.
fn detect_delimiter(path: &Path) -> Result<u8, csv::Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;

    let pipes = first_line.bytes().filter(|&b| b == b'|').count();
    let commas = first_line.bytes().filter(|&b| b == b',').count();

    if pipes > commas { Ok(b'|') } else { Ok(b',') }
}

/// Read one column (0-based) from a TPC-H style CSV file as owned strings.
/// Detects the delimiter (TPC-H files are often pipe-delimited; plain CSV is comma).
pub fn tpch_column(path: &Path, col: usize) -> Result<Vec<String>, csv::Error> {
    let delimiter = detect_delimiter(path)?;

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .flexible(true)
        .from_path(path)?;

    let mut values = Vec::new();
    for record in reader.records() {
        let record = record?;
        if let Some(field) = record.get(col) {
            values.push(field.to_string());
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_has_exact_cardinality() {
        let v: Vec<u64> = synthetic_distinct(1000).collect();
        assert_eq!(v.len(), 1000);
        assert_eq!(exact_distinct(v.iter().map(|x| x.to_string())), 1000);
    }
    #[test]
    fn tpch_column_reads_real_file() {
        // small real file
        let rows = tpch_column(std::path::Path::new("../../tests/data/customer.csv"), 0).unwrap();
        assert!(!rows.is_empty());
    }
}
