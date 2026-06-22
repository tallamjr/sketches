// Apache datasketches-cpp benchmark runner.
//
// Exercises the four shared sketches (HLL, Theta, Bloom, Count-Min) over the
// same synthetic and TPC-H datasets as the Rust runners, emitting rows in the
// shared CSV schema so the reporter can join C++ reference numbers directly.
//
// Schema (must match benchmarks/runner-ours exactly):
//   implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error
//
// Usage:
//   runner_cpp --n <N> [--tpch <csv_path> --col <COL>] --out <results.csv>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "hll.hpp"
#include "theta_sketch.hpp"
#include "cpc_sketch.hpp"
#include "bloom_filter.hpp"
#include "count_min.hpp"

namespace {

const char* const HEADER =
    "implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error";

const char* const IMPLEMENTATION = "apache-cpp";
const std::string HOT_KEY = "__hot__";
const uint64_t HOT_KEY_COUNT = 1000;

// HLL configuration matching runner-ours (lg_k = 12).
const uint8_t HLL_LG_K = 12;
// Theta configuration matching runner-ours nominal entries (4096 = 2^12).
const uint8_t THETA_LG_K = 12;
// CPC configuration matching runner-ours (lg_k = 12).
const uint8_t CPC_LG_K = 12;
// Count-Min configuration matching runner-ours (5 hashes, 2048 buckets).
const uint8_t COUNTMIN_NUM_HASHES = 5;
const uint32_t COUNTMIN_NUM_BUCKETS = 2048;
// Bloom target false positive probability matching runner-ours (1%).
const double BLOOM_TARGET_FPP = 0.01;

// Format a double with the same precision the Rust runner uses ("{:.6}").
std::string format_f64(double value) {
  std::ostringstream out;
  out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(6);
  out << value;
  return out.str();
}

double throughput(uint64_t items, double elapsed_secs) {
  if (elapsed_secs > 0.0) {
    return static_cast<double>(items) / elapsed_secs;
  }
  return 0.0;
}

// Build one CSV row. Optional fields are passed as empty strings.
std::string row(const std::string& sketch, const std::string& dataset, const std::string& op,
                uint64_t n, double throughput_ops, const std::string& bytes,
                const std::string& estimate, const std::string& exact,
                const std::string& rel_error) {
  std::ostringstream out;
  out << IMPLEMENTATION << ',' << sketch << ',' << dataset << ',' << op << ',' << n << ','
      << format_f64(throughput_ops) << ',' << bytes << ',' << estimate << ',' << exact << ','
      << rel_error;
  return out.str();
}

using Clock = std::chrono::steady_clock;

double elapsed_secs(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// HLL distinct-count row. T is the item type fed to the sketch.
template <typename T>
std::string hll_row(const std::string& dataset, const std::vector<T>& items, double exact) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  datasketches::hll_sketch sketch(HLL_LG_K, datasketches::HLL_8);
  const auto start = Clock::now();
  for (const T& item : items) {
    sketch.update(item);
  }
  const double secs = elapsed_secs(start);
  const double estimate = sketch.get_estimate();
  const double rel_error = std::abs(estimate - exact) / exact;
  const uint32_t bytes = sketch.get_compact_serialization_bytes();
  return row("hll", dataset, "distinct_count", n, throughput(n, secs),
             std::to_string(bytes), format_f64(estimate), format_f64(exact),
             format_f64(rel_error));
}

// CPC distinct-count row. T is the item type fed to the sketch.
template <typename T>
std::string cpc_row(const std::string& dataset, const std::vector<T>& items, double exact) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  datasketches::cpc_sketch sketch(CPC_LG_K);
  const auto start = Clock::now();
  for (const T& item : items) {
    sketch.update(item);
  }
  const double secs = elapsed_secs(start);
  const double estimate = sketch.get_estimate();
  const double rel_error = std::abs(estimate - exact) / exact;
  const size_t bytes = sketch.serialize().size();
  return row("cpc", dataset, "distinct_count", n, throughput(n, secs),
             std::to_string(bytes), format_f64(estimate), format_f64(exact),
             format_f64(rel_error));
}

// Theta distinct-count row.
template <typename T>
std::string theta_row(const std::string& dataset, const std::vector<T>& items, double exact) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  auto sketch = datasketches::update_theta_sketch::builder().set_lg_k(THETA_LG_K).build();
  const auto start = Clock::now();
  for (const T& item : items) {
    sketch.update(item);
  }
  const double secs = elapsed_secs(start);
  const double estimate = sketch.get_estimate();
  const double rel_error = std::abs(estimate - exact) / exact;
  // Bytes is the serialized size of the compact form, the comparable
  // persisted footprint of the sketch.
  const datasketches::compact_theta_sketch compact = sketch.compact();
  const size_t bytes = compact.get_serialized_size_bytes();
  return row("theta", dataset, "distinct_count", n, throughput(n, secs),
             std::to_string(bytes), format_f64(estimate), format_f64(exact),
             format_f64(rel_error));
}

// Bloom build row. A membership filter has no cardinality estimate, so the
// estimate/exact/rel_error fields are left empty.
template <typename T>
std::string bloom_row(const std::string& dataset, const std::vector<T>& items) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  // Size the filter for n distinct items at the target false positive rate.
  const uint64_t max_items = n > 0 ? n : 1;
  datasketches::bloom_filter filter =
      datasketches::bloom_filter::builder::create_by_accuracy(max_items, BLOOM_TARGET_FPP);
  const auto start = Clock::now();
  for (const T& item : items) {
    filter.update(item);
  }
  const double secs = elapsed_secs(start);
  const size_t bytes = filter.get_serialized_size_bytes();
  return row("bloom", dataset, "build", n, throughput(n, secs), std::to_string(bytes), "",
             "", "");
}

// Count-Min point-query row. Each item is incremented once, then a designated
// hot key is incremented HOT_KEY_COUNT times; the query is for that key.
template <typename T>
std::string countmin_row(const std::string& dataset, const std::vector<T>& items) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  datasketches::count_min_sketch<uint64_t> sketch(COUNTMIN_NUM_HASHES, COUNTMIN_NUM_BUCKETS);
  const auto start = Clock::now();
  for (const T& item : items) {
    sketch.update(item, 1);
  }
  for (uint64_t i = 0; i < HOT_KEY_COUNT; ++i) {
    sketch.update(HOT_KEY, 1);
  }
  const double secs = elapsed_secs(start);
  const double estimate = static_cast<double>(sketch.get_estimate(HOT_KEY));
  const double exact = static_cast<double>(HOT_KEY_COUNT);
  const double rel_error = std::abs(estimate - exact) / exact;
  const size_t bytes = sketch.get_serialized_size_bytes();
  const uint64_t total_ops = n + HOT_KEY_COUNT;
  return row("countmin", dataset, "point_query", total_ops, throughput(total_ops, secs),
             std::to_string(bytes), format_f64(estimate), format_f64(exact),
             format_f64(rel_error));
}

// The exact CSV header line for the multi-trial RMSE mode (--trials). This
// schema is separate from HEADER; the two are never mixed in one file. Its
// columns match runner-ours and runner-apache-rust so the implementations can
// be compared row-for-row.
const char* const RMSE_HEADER =
    "implementation,sketch,lg_k,trials,n_per_trial,rmse,mean_rel_error,max_rel_error";

// Format a single RMSE summary row from a vector of per-trial relative errors.
// rmse = sqrt(mean(rel_error^2)), mean_rel_error = mean(rel_error),
// max_rel_error = max(rel_error). All three doubles are printed at %.6f.
std::string rmse_row(const std::string& sketch, uint8_t lg_k, uint64_t trials,
                     uint64_t n, const std::vector<double>& errors) {
  const double count = static_cast<double>(errors.size());
  double sum_sq = 0.0;
  double sum = 0.0;
  double max = 0.0;
  for (double e : errors) {
    sum_sq += e * e;
    sum += e;
    if (e > max) {
      max = e;
    }
  }
  const double rmse = std::sqrt(sum_sq / count);
  const double mean = sum / count;
  std::ostringstream out;
  out << IMPLEMENTATION << ',' << sketch << ',' << static_cast<unsigned>(lg_k) << ','
      << trials << ',' << n << ',' << format_f64(rmse) << ',' << format_f64(mean) << ','
      << format_f64(max);
  return out.str();
}

// Run `trials` independent trials of `n` distinct uint64 items each (trial t
// over the disjoint range [t*n, (t+1)*n)) and emit one RMSE summary row per
// distinct-count sketch (theta/hll/cpc) at lg_k = 12. A fresh sketch is built
// per trial. Mirrors run_rmse in runner-apache-rust.
std::vector<std::string> run_rmse(uint64_t trials, uint64_t n) {
  std::vector<double> theta_errs;
  std::vector<double> hll_errs;
  std::vector<double> cpc_errs;
  theta_errs.reserve(trials);
  hll_errs.reserve(trials);
  cpc_errs.reserve(trials);
  const double truth = static_cast<double>(n);
  for (uint64_t t = 0; t < trials; ++t) {
    const uint64_t start = t * n;
    auto theta = datasketches::update_theta_sketch::builder().set_lg_k(THETA_LG_K).build();
    datasketches::hll_sketch hll(HLL_LG_K, datasketches::HLL_8);
    datasketches::cpc_sketch cpc(CPC_LG_K);
    for (uint64_t i = start; i < start + n; ++i) {
      theta.update(i);
      hll.update(i);
      cpc.update(i);
    }
    theta_errs.push_back(std::fabs(theta.get_estimate() - truth) / truth);
    hll_errs.push_back(std::fabs(hll.get_estimate() - truth) / truth);
    cpc_errs.push_back(std::fabs(cpc.get_estimate() - truth) / truth);
  }
  return {
      RMSE_HEADER,
      rmse_row("theta", THETA_LG_K, trials, n, theta_errs),
      rmse_row("hll", HLL_LG_K, trials, n, hll_errs),
      rmse_row("cpc", CPC_LG_K, trials, n, cpc_errs),
  };
}

// Detect the field delimiter by inspecting the first line of the file. TPC-H
// data is commonly pipe-delimited; plain CSV uses commas. Pick whichever
// appears more often, defaulting to comma. Mirrors datasets::detect_delimiter.
char detect_delimiter(const std::string& first_line) {
  size_t pipes = 0;
  size_t commas = 0;
  for (char c : first_line) {
    if (c == '|') {
      ++pipes;
    } else if (c == ',') {
      ++commas;
    }
  }
  return pipes > commas ? '|' : ',';
}

// Parse a single delimited line into fields. Honours double-quoted fields so
// that quoted values containing the delimiter are not split, matching the CSV
// reader the Rust runner uses.
std::vector<std::string> parse_line(const std::string& line, char delimiter) {
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        field.push_back(c);
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == delimiter) {
      fields.push_back(field);
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  fields.push_back(field);
  return fields;
}

// Read one column (0-based) from a TPC-H style CSV file as strings. The reader
// is header-agnostic (has_headers=false in the Rust runner), so the header row
// is included as a data row to keep the row counts identical.
bool read_tpch_column(const std::string& path, size_t col, std::vector<std::string>& out) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "runner_cpp: failed to open TPC-H file: " << path << std::endl;
    return false;
  }
  std::string line;
  char delimiter = ',';
  bool first = true;
  while (std::getline(file, line)) {
    // Strip a trailing carriage return for CRLF files.
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (first) {
      delimiter = detect_delimiter(line);
      first = false;
    }
    const std::vector<std::string> fields = parse_line(line, delimiter);
    if (col < fields.size()) {
      out.push_back(fields[col]);
    }
  }
  if (file.bad()) {
    std::cerr << "runner_cpp: error while reading TPC-H file: " << path << std::endl;
    return false;
  }
  return true;
}

// Derive the dataset label from a TPC-H CSV path: the file stem, lowercased
// (for example "/data/nation.csv" becomes "nation"). Mirrors runner-ours.
std::string dataset_label(const std::string& path) {
  size_t slash = path.find_last_of("/\\");
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  if (dot != std::string::npos && dot != 0) {
    name = name.substr(0, dot);
  }
  if (name.empty()) {
    name = "tpch";
  }
  for (char& c : name) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return name;
}

void print_usage() {
  std::cerr << "usage: runner_cpp --n <N> [--tpch <csv_path> --col <COL>] --out <results.csv>"
            << std::endl;
  std::cerr << "       runner_cpp --trials <T> --n <N> --out <results.csv>" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t n = 0;
  bool have_n = false;
  std::string tpch_path;
  bool have_tpch = false;
  size_t col = 0;
  bool have_col = false;
  std::string out_path;
  bool have_out = false;
  uint64_t trials = 0;
  bool have_trials = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      n = std::strtoull(argv[++i], nullptr, 10);
      have_n = true;
    } else if (arg == "--trials" && i + 1 < argc) {
      trials = std::strtoull(argv[++i], nullptr, 10);
      have_trials = true;
    } else if (arg == "--tpch" && i + 1 < argc) {
      tpch_path = argv[++i];
      have_tpch = true;
    } else if (arg == "--col" && i + 1 < argc) {
      col = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
      have_col = true;
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
      have_out = true;
    } else {
      std::cerr << "runner_cpp: unrecognised argument: " << arg << std::endl;
      print_usage();
      return 1;
    }
  }

  if (!have_n || !have_out) {
    print_usage();
    return 1;
  }
  if (have_tpch != have_col) {
    std::cerr << "runner_cpp: --tpch and --col must be supplied together" << std::endl;
    print_usage();
    return 1;
  }

  // Multi-trial RMSE mode. When --trials is supplied the runner emits the RMSE
  // summary schema (theta/hll/cpc over disjoint uint64 ranges) instead of the
  // single-run benchmark schema; the two are never mixed in one file.
  if (have_trials) {
    if (trials == 0) {
      std::cerr << "runner_cpp: --trials must be a positive integer" << std::endl;
      print_usage();
      return 1;
    }
    if (n == 0) {
      std::cerr << "runner_cpp: --n must be a positive integer in --trials mode" << std::endl;
      print_usage();
      return 1;
    }
    const std::vector<std::string> rmse_lines = run_rmse(trials, n);
    std::ofstream out(out_path);
    if (!out.is_open()) {
      std::cerr << "runner_cpp: failed to open output file: " << out_path << std::endl;
      return 1;
    }
    for (const std::string& line : rmse_lines) {
      out << line << '\n';
    }
    out.flush();
    if (!out.good()) {
      std::cerr << "runner_cpp: error while writing output file: " << out_path << std::endl;
      return 1;
    }
    return 0;
  }

  std::vector<std::string> lines;
  lines.push_back(HEADER);

  // Synthetic dataset: values 0..n-1, exact cardinality = n.
  {
    std::vector<uint64_t> synthetic;
    synthetic.reserve(n);
    std::unordered_set<uint64_t> distinct;
    for (uint64_t i = 0; i < n; ++i) {
      synthetic.push_back(i);
      distinct.insert(i);
    }
    const double exact = static_cast<double>(distinct.size());
    lines.push_back(hll_row("synthetic", synthetic, exact));
    lines.push_back(cpc_row("synthetic", synthetic, exact));
    lines.push_back(theta_row("synthetic", synthetic, exact));
    lines.push_back(bloom_row("synthetic", synthetic));
    lines.push_back(countmin_row("synthetic", synthetic));
  }

  // TPC-H column dataset, if requested.
  if (have_tpch) {
    std::vector<std::string> values;
    if (!read_tpch_column(tpch_path, col, values)) {
      return 1;
    }
    std::unordered_set<std::string> distinct(values.begin(), values.end());
    const double exact = static_cast<double>(distinct.size());
    const std::string dataset = dataset_label(tpch_path);
    lines.push_back(hll_row(dataset, values, exact));
    lines.push_back(cpc_row(dataset, values, exact));
    lines.push_back(theta_row(dataset, values, exact));
    lines.push_back(bloom_row(dataset, values));
    lines.push_back(countmin_row(dataset, values));
  }

  std::ofstream out(out_path);
  if (!out.is_open()) {
    std::cerr << "runner_cpp: failed to open output file: " << out_path << std::endl;
    return 1;
  }
  for (const std::string& line : lines) {
    out << line << '\n';
  }
  out.flush();
  if (!out.good()) {
    std::cerr << "runner_cpp: error while writing output file: " << out_path << std::endl;
    return 1;
  }

  return 0;
}
