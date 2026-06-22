//! `runner-apache-rust`: run the upstream Apache `datasketches` crate over the
//! shared datasets and emit results in the shared CSV schema.
//!
//! Usage:
//!   runner-apache-rust --n <N> [--tpch <csv_path> --col <COL>] --out <results.csv>
//!
//! Always runs the synthetic dataset. If `--tpch` is given, also runs that
//! column. The TPC-H dataset label is derived from the CSV file stem (for
//! example `customer.csv` becomes `customer`).

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

struct Args {
    n: u64,
    tpch: Option<PathBuf>,
    col: Option<usize>,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut n: Option<u64> = None;
    let mut tpch: Option<PathBuf> = None;
    let mut col: Option<usize> = None;
    let mut out: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--n" => {
                let v = args.next().ok_or("--n requires a value")?;
                n = Some(v.parse().map_err(|_| format!("invalid --n value: {v}"))?);
            }
            "--tpch" => {
                let v = args.next().ok_or("--tpch requires a path")?;
                tpch = Some(PathBuf::from(v));
            }
            "--col" => {
                let v = args.next().ok_or("--col requires a value")?;
                col = Some(v.parse().map_err(|_| format!("invalid --col value: {v}"))?);
            }
            "--out" => {
                let v = args.next().ok_or("--out requires a path")?;
                out = Some(PathBuf::from(v));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let n = n.ok_or("--n is required")?;
    let out = out.ok_or("--out is required")?;
    if tpch.is_some() != col.is_some() {
        return Err("--tpch and --col must be given together".to_string());
    }

    Ok(Args { n, tpch, col, out })
}

/// Derive the dataset label from a TPC-H CSV path (file stem, lowercased).
fn dataset_label(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "tpch".to_string())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let mut lines = runner_apache_rust::run(args.n);

    if let (Some(path), Some(col)) = (&args.tpch, args.col) {
        let label = dataset_label(path);
        let tpch_lines = runner_apache_rust::run_tpch(path, col, &label)
            .map_err(|e| format!("failed to read TPC-H column from {path:?}: {e}"))?;
        lines.extend(tpch_lines);
    }

    write_lines(&args.out, &lines).map_err(|e| format!("failed to write {:?}: {e}", args.out))?;

    Ok(())
}

fn write_lines(out: &Path, lines: &[String]) -> io::Result<()> {
    let mut file = File::create(out)?;
    for line in lines {
        writeln!(file, "{line}")?;
    }
    file.flush()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("runner-apache-rust: {e}");
        process::exit(1);
    }
}
