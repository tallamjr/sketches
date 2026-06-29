// Apache datasketches-cpp benchmark runner.
//
// Exercises the four shared sketches (HLL, Theta, Bloom, Count-Min) over the
// same synthetic and TPC-H datasets as the Rust runners, emitting rows in the
// shared CSV schema so the reporter can join C++ reference numbers directly.
//
// Schema (must match benchmarks/runner-ours exactly):
//   implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,estimate,exact,rel_error
//
// Throughput follows the shared rounds protocol: `reps` independent rounds, each
// with one untimed warmup pass then REPS_PER_ROUND timed reps rebuilding a fresh
// sketch per rep; the round-sample is the median of its per-rep rates. The row
// reports the median over round-samples, their population stddev, and a
// deterministic 95% bootstrap CI (fixed-seed SplitMix64) over them.
//
// Usage:
//   runner_cpp --n <N> [--reps <R>] [--tpch <csv_path> --col <COL>] --out <results.csv>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <new>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__)
// Declare malloc_size directly rather than including <malloc/malloc.h>: under
// GNU g++ on macOS that header transitively pulls in <mach/message.h>, whose
// xnu_static_assert_struct_size macros fail to compile. The signature is stable
// libSystem ABI, so a local extern "C" prototype is sufficient and portable
// across both Apple clang and Homebrew g++.
extern "C" size_t malloc_size(const void* ptr);
#elif defined(__GLIBC__)
#include <malloc.h>  // malloc_usable_size on glibc
#endif

#include "hll.hpp"
#include "theta_sketch.hpp"
#include "cpc_sketch.hpp"
#include "bloom_filter.hpp"
#include "count_min.hpp"

// ---------------------------------------------------------------------------
// Global allocation counter for per-sketch live-heap measurement.
//
// We override the global operator new/delete to track the number of live heap
// bytes the process currently holds. To get a TRUE delta we must know the size
// at delete time; rather than thread a manual prefix header through every
// allocation we ask the allocator for the real block size:
//   - macOS: malloc_size(ptr)
//   - glibc: malloc_usable_size(ptr)
//   - otherwise: fall back to the requested size stashed in a small header.
//
// The counter is process-wide (it sees datasketches-cpp internals and every
// std container too); that is exactly what we want, because the per-sketch
// delta is read OUTSIDE the timed reps via measure_live. Single-threaded
// measurement, but the atomic keeps it sound regardless.
// ---------------------------------------------------------------------------
namespace bench_alloc {

std::atomic<size_t> g_live_bytes{0};

inline size_t live_bytes() { return g_live_bytes.load(std::memory_order_relaxed); }

#if defined(__APPLE__) || defined(__GLIBC__)

inline size_t block_size(void* ptr) {
#if defined(__APPLE__)
  return malloc_size(ptr);
#else
  return malloc_usable_size(ptr);
#endif
}

inline void* counted_alloc(size_t size) {
  // operator new must never return nullptr for a non-zero request; round 0 up
  // to 1 so the allocation (and its bookkeeping) is well defined.
  void* ptr = std::malloc(size == 0 ? 1 : size);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  g_live_bytes.fetch_add(block_size(ptr), std::memory_order_relaxed);
  return ptr;
}

inline void counted_free(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
  g_live_bytes.fetch_sub(block_size(ptr), std::memory_order_relaxed);
  std::free(ptr);
}

#else  // portable prefix-header fallback (no malloc introspection available)

inline void* counted_alloc(size_t size) {
  const size_t header = sizeof(size_t);
  void* base = std::malloc(header + (size == 0 ? 1 : size));
  if (base == nullptr) {
    throw std::bad_alloc();
  }
  *static_cast<size_t*>(base) = size;
  g_live_bytes.fetch_add(size, std::memory_order_relaxed);
  return static_cast<char*>(base) + header;
}

inline void counted_free(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
  const size_t header = sizeof(size_t);
  void* base = static_cast<char*>(ptr) - header;
  g_live_bytes.fetch_sub(*static_cast<size_t*>(base), std::memory_order_relaxed);
  std::free(base);
}

#endif

}  // namespace bench_alloc

void* operator new(std::size_t size) { return bench_alloc::counted_alloc(size); }
void* operator new[](std::size_t size) { return bench_alloc::counted_alloc(size); }
void operator delete(void* ptr) noexcept { bench_alloc::counted_free(ptr); }
void operator delete[](void* ptr) noexcept { bench_alloc::counted_free(ptr); }
// Sized deletes g++ may emit; the size is redundant since the allocator knows
// the block size, so forward to the same counted_free.
void operator delete(void* ptr, std::size_t) noexcept { bench_alloc::counted_free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { bench_alloc::counted_free(ptr); }

namespace {

const char* const HEADER =
    "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,"
    "throughput_ci_low,throughput_ci_high,"
    "bytes,live_bytes,estimate,exact,rel_error";

const char* const IMPLEMENTATION = "apache-cpp";
const std::string HOT_KEY = "__hot__";
const uint64_t HOT_KEY_COUNT = 1000;
// Default number of rounds for the rounds throughput protocol.
const int DEFAULT_REPS = 30;
// Timed reps inside each independent round; the round-sample is their median.
const int REPS_PER_ROUND = 5;
// Number of bootstrap resamples for the throughput confidence interval.
const int BOOTSTRAP_RESAMPLES = 2000;
// Fixed SplitMix64 seed so the bootstrap CI is reproducible run to run.
const uint64_t BOOTSTRAP_SEED = 0x9E3779B97F4A7C15ULL;

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

// A stabilised throughput measurement: the median over independent rounds, the
// population stddev of the round-samples, and a nonparametric 95% bootstrap CI.
struct Throughput {
  double median;
  double stddev;
  double ci_low;
  double ci_high;
};

// Build one CSV row. Optional fields are passed as empty strings. `live_bytes`
// is the net heap delta to build and hold one populated sketch, measured by
// measure_live outside the timed reps.
std::string row(const std::string& sketch, const std::string& dataset, const std::string& op,
                uint64_t n, int reps, const Throughput& t,
                const std::string& bytes, const std::string& live_bytes,
                const std::string& estimate, const std::string& exact,
                const std::string& rel_error) {
  std::ostringstream out;
  out << IMPLEMENTATION << ',' << sketch << ',' << dataset << ',' << op << ',' << n << ',' << reps
      << ',' << format_f64(t.median) << ',' << format_f64(t.stddev) << ','
      << format_f64(t.ci_low) << ',' << format_f64(t.ci_high) << ','
      << bytes << ',' << live_bytes << ',' << estimate << ',' << exact << ','
      << rel_error;
  return out.str();
}

using Clock = std::chrono::steady_clock;

// Black-box sink: force the compiler to treat `value` as observed so the build
// of the sketch inside the timed lambda cannot be optimised away. Mirrors the
// role of core::hint::black_box in the Rust runners.
template <typename T>
void do_not_optimize(const T& value) {
  asm volatile("" : : "g"(value) : "memory");
}

// Median of a vector (sorts in place). Returns 0 for an empty vector.
double median(std::vector<double>& xs) {
  if (xs.empty()) {
    return 0.0;
  }
  std::sort(xs.begin(), xs.end());
  const size_t n = xs.size();
  if (n % 2 == 1) {
    return xs[n / 2];
  }
  return (xs[n / 2 - 1] + xs[n / 2]) / 2.0;
}

// Median over a copy, leaving the caller's vector untouched. median() sorts its
// argument in place, so the bootstrap resample loop (which reuses one draw
// buffer) must go through this pass-by-value helper to avoid corrupting the
// buffer-to-sample correspondence between iterations.
double median_copy(std::vector<double> xs) { return median(xs); }

// Population standard deviation (divides by n). Returns 0 for an empty vector.
double population_stddev(const std::vector<double>& xs) {
  if (xs.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double x : xs) {
    sum += x;
  }
  const double mean = sum / static_cast<double>(xs.size());
  double var = 0.0;
  for (double x : xs) {
    const double d = x - mean;
    var += d * d;
  }
  var /= static_cast<double>(xs.size());
  return std::sqrt(var);
}

// SplitMix64 step. Deterministic, identical across the Rust and Python runners.
uint64_t splitmix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Nonparametric 95% bootstrap CI of the median over `samples`. Deterministic:
// fixed-seed SplitMix64, B=2000 resamples, 2.5/97.5 nearest-rank percentiles.
// Returns (median, median) when fewer than two samples are given. Mirrors the
// Rust runners' bootstrap_ci exactly so the CI columns are cross-language.
std::pair<double, double> bootstrap_ci(const std::vector<double>& samples) {
  const size_t r = samples.size();
  if (r <= 1) {
    const double m = r == 1 ? samples[0] : 0.0;
    return {m, m};
  }
  uint64_t state = BOOTSTRAP_SEED;
  std::vector<double> resample_medians;
  resample_medians.reserve(static_cast<size_t>(BOOTSTRAP_RESAMPLES));
  std::vector<double> draw(r);
  for (int b = 0; b < BOOTSTRAP_RESAMPLES; ++b) {
    for (size_t i = 0; i < r; ++i) {
      draw[i] = samples[splitmix64(state) % r];
    }
    resample_medians.push_back(median_copy(draw));
  }
  std::sort(resample_medians.begin(), resample_medians.end());
  const int B = BOOTSTRAP_RESAMPLES;
  auto idx = [B](double p) {
    long i = static_cast<long>(std::floor(p * B));
    if (i < 0) i = 0;
    if (i > B - 1) i = B - 1;
    return static_cast<size_t>(i);
  };
  return {resample_medians[idx(0.025)], resample_medians[idx(0.975)]};
}

// Run `rounds` independent rounds. Each round does one untimed warmup body()
// then `reps_per_round` timed body() calls; the round-sample is the median of
// its per-rep rates (ops/s). Reports the median over round-samples, their
// population stddev, and the bootstrap CI over them. Matches the runner-ours
// timing helper semantics exactly.
Throughput timed_throughput_rounds(int rounds, int reps_per_round, uint64_t ops_per_rep,
                                   const std::function<void()>& body) {
  std::vector<double> round_samples;
  round_samples.reserve(static_cast<size_t>(rounds));
  for (int r = 0; r < rounds; ++r) {
    body();  // untimed warmup
    std::vector<double> rates;
    rates.reserve(static_cast<size_t>(reps_per_round));
    for (int i = 0; i < reps_per_round; ++i) {
      const auto start = Clock::now();
      body();
      const double secs = std::chrono::duration<double>(Clock::now() - start).count();
      rates.push_back(secs > 0.0 ? static_cast<double>(ops_per_rep) / secs : 0.0);
    }
    round_samples.push_back(median(rates));
  }
  const double sd = population_stddev(round_samples);
  const std::pair<double, double> ci = bootstrap_ci(round_samples);
  const double lo = ci.first;
  const double hi = ci.second;
  const double med = median_copy(round_samples);
  return {med, sd, lo, hi};
}

// Measure the net heap bytes attributable to building and holding ONE populated
// sketch, the C++ analogue of the Rust runners' measure_live. `build` returns a
// freshly populated sketch by value; we snapshot the live-byte counter before
// and after its construction, then hand the still-alive sketch to `consume`
// (which reads estimate/serialised size) before it goes out of scope. The delta
// is saturating (returns 0 rather than underflowing) so transient allocator
// noise can never produce a bogus huge number. Read OUTSIDE the timed reps.
template <typename Build, typename Consume>
size_t measure_live(const Build& build, const Consume& consume) {
  const size_t before = bench_alloc::live_bytes();
  auto sketch = build();
  const size_t after = bench_alloc::live_bytes();
  const size_t live = after > before ? after - before : 0;
  consume(sketch);
  return live;
}

// HLL distinct-count row. T is the item type fed to the sketch.
template <typename T>
std::string hll_row(const std::string& dataset, const std::vector<T>& items, double exact,
                    int reps) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  const Throughput tput = timed_throughput_rounds(reps, REPS_PER_ROUND, n, [&] {
    datasketches::hll_sketch sketch(HLL_LG_K, datasketches::HLL_8);
    for (const T& item : items) {
      sketch.update(item);
    }
    do_not_optimize(sketch.get_estimate());
  });
  // Build one more populated sketch outside the timing loop, inside the
  // measure_live scope, to read the row's estimate, serialised size and the
  // net live-heap delta.
  double estimate = 0.0;
  uint32_t bytes = 0;
  const size_t live = measure_live(
      [&] {
        datasketches::hll_sketch s(HLL_LG_K, datasketches::HLL_8);
        for (const T& item : items) {
          s.update(item);
        }
        return s;
      },
      [&](const datasketches::hll_sketch& s) {
        estimate = s.get_estimate();
        bytes = s.get_compact_serialization_bytes();
      });
  const double rel_error = std::abs(estimate - exact) / exact;
  return row("hll", dataset, "distinct_count", n, reps, tput,
             std::to_string(bytes), std::to_string(live), format_f64(estimate),
             format_f64(exact), format_f64(rel_error));
}

// CPC distinct-count row. T is the item type fed to the sketch.
template <typename T>
std::string cpc_row(const std::string& dataset, const std::vector<T>& items, double exact,
                    int reps) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  const Throughput tput = timed_throughput_rounds(reps, REPS_PER_ROUND, n, [&] {
    datasketches::cpc_sketch sketch(CPC_LG_K);
    for (const T& item : items) {
      sketch.update(item);
    }
    do_not_optimize(sketch.get_estimate());
  });
  double estimate = 0.0;
  size_t bytes = 0;
  const size_t live = measure_live(
      [&] {
        datasketches::cpc_sketch s(CPC_LG_K);
        for (const T& item : items) {
          s.update(item);
        }
        return s;
      },
      [&](const datasketches::cpc_sketch& s) {
        estimate = s.get_estimate();
        bytes = s.serialize().size();
      });
  const double rel_error = std::abs(estimate - exact) / exact;
  return row("cpc", dataset, "distinct_count", n, reps, tput,
             std::to_string(bytes), std::to_string(live), format_f64(estimate),
             format_f64(exact), format_f64(rel_error));
}

// Theta distinct-count row.
template <typename T>
std::string theta_row(const std::string& dataset, const std::vector<T>& items, double exact,
                      int reps) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  const Throughput tput = timed_throughput_rounds(reps, REPS_PER_ROUND, n, [&] {
    auto sketch = datasketches::update_theta_sketch::builder().set_lg_k(THETA_LG_K).build();
    for (const T& item : items) {
      sketch.update(item);
    }
    do_not_optimize(sketch.get_estimate());
  });
  double estimate = 0.0;
  size_t bytes = 0;
  const size_t live = measure_live(
      [&] {
        auto s = datasketches::update_theta_sketch::builder().set_lg_k(THETA_LG_K).build();
        for (const T& item : items) {
          s.update(item);
        }
        return s;
      },
      [&](const datasketches::update_theta_sketch& s) {
        estimate = s.get_estimate();
        // Bytes is the serialized size of the compact form, the comparable
        // persisted footprint of the sketch.
        const datasketches::compact_theta_sketch compact = s.compact();
        bytes = compact.get_serialized_size_bytes();
      });
  const double rel_error = std::abs(estimate - exact) / exact;
  return row("theta", dataset, "distinct_count", n, reps, tput,
             std::to_string(bytes), std::to_string(live), format_f64(estimate),
             format_f64(exact), format_f64(rel_error));
}

// Bloom build row. A membership filter has no cardinality estimate, so the
// estimate/exact/rel_error fields are left empty.
template <typename T>
std::string bloom_row(const std::string& dataset, const std::vector<T>& items, int reps) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  // Size the filter for n distinct items at the target false positive rate.
  const uint64_t max_items = n > 0 ? n : 1;
  const Throughput tput = timed_throughput_rounds(reps, REPS_PER_ROUND, n, [&] {
    datasketches::bloom_filter filter =
        datasketches::bloom_filter::builder::create_by_accuracy(max_items, BLOOM_TARGET_FPP);
    for (const T& item : items) {
      filter.update(item);
    }
    do_not_optimize(filter.get_bits_used());
  });
  size_t bytes = 0;
  const size_t live = measure_live(
      [&] {
        datasketches::bloom_filter f =
            datasketches::bloom_filter::builder::create_by_accuracy(max_items, BLOOM_TARGET_FPP);
        for (const T& item : items) {
          f.update(item);
        }
        return f;
      },
      [&](const datasketches::bloom_filter& f) { bytes = f.get_serialized_size_bytes(); });
  return row("bloom", dataset, "build", n, reps, tput, std::to_string(bytes),
             std::to_string(live), "", "", "");
}

// Count-Min point-query row. Each item is incremented once, then a designated
// hot key is incremented HOT_KEY_COUNT times; the query is for that key.
template <typename T>
std::string countmin_row(const std::string& dataset, const std::vector<T>& items, int reps) {
  const uint64_t n = static_cast<uint64_t>(items.size());
  const uint64_t total_ops = n + HOT_KEY_COUNT;
  const Throughput tput = timed_throughput_rounds(reps, REPS_PER_ROUND, total_ops, [&] {
    datasketches::count_min_sketch<uint64_t> sketch(COUNTMIN_NUM_HASHES, COUNTMIN_NUM_BUCKETS);
    for (const T& item : items) {
      sketch.update(item, 1);
    }
    for (uint64_t i = 0; i < HOT_KEY_COUNT; ++i) {
      sketch.update(HOT_KEY, 1);
    }
    do_not_optimize(sketch.get_estimate(HOT_KEY));
  });
  double estimate = 0.0;
  size_t bytes = 0;
  const size_t live = measure_live(
      [&] {
        datasketches::count_min_sketch<uint64_t> s(COUNTMIN_NUM_HASHES, COUNTMIN_NUM_BUCKETS);
        for (const T& item : items) {
          s.update(item, 1);
        }
        for (uint64_t i = 0; i < HOT_KEY_COUNT; ++i) {
          s.update(HOT_KEY, 1);
        }
        return s;
      },
      [&](const datasketches::count_min_sketch<uint64_t>& s) {
        estimate = static_cast<double>(s.get_estimate(HOT_KEY));
        bytes = s.get_serialized_size_bytes();
      });
  const double exact = static_cast<double>(HOT_KEY_COUNT);
  const double rel_error = std::abs(estimate - exact) / exact;
  return row("countmin", dataset, "point_query", total_ops, reps, tput,
             std::to_string(bytes), std::to_string(live), format_f64(estimate),
             format_f64(exact), format_f64(rel_error));
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
  std::cerr << "usage: runner_cpp --n <N> [--reps <R>] [--tpch <csv_path> --col <COL>] "
               "--out <results.csv>"
            << std::endl;
  std::cerr << "       runner_cpp --trials <T> --n <N> --out <results.csv>" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  // Self-check: build one populated sketch through the measured path and assert
  // the live-heap delta is non-zero. This is the C++ analogue of a failing test
  // (before the allocation counter was wired, this reported 0). A broken counter
  // must abort the run so CI/the reviewer catches a misleading metric.
  {
    std::vector<uint64_t> probe;
    probe.reserve(1000);
    for (uint64_t i = 0; i < 1000; ++i) {
      probe.push_back(i);
    }
    const size_t live = measure_live(
        [&] {
          datasketches::hll_sketch s(HLL_LG_K, datasketches::HLL_8);
          for (uint64_t v : probe) {
            s.update(v);
          }
          return s;
        },
        [](const datasketches::hll_sketch& s) { do_not_optimize(s.get_estimate()); });
    if (live == 0) {
      std::cerr << "runner_cpp: live_bytes self-check failed: measured 0 heap bytes for a "
                   "populated HLL sketch; the allocation counter is broken"
                << std::endl;
      return 2;
    }
  }

  // Cross-language parity self-check: bootstrap_ci of the pinned input vector
  // must reproduce the values the Rust runners pinned, byte for byte, or the
  // SplitMix64/bootstrap port has diverged and the CI columns are meaningless.
  {
    std::vector<double> samples = {100.0, 110.0, 90.0, 105.0, 95.0, 120.0,
                                   80.0, 115.0, 85.0, 100.0};
    const std::pair<double, double> ci = bootstrap_ci(samples);
    const double lo = ci.first;
    const double hi = ci.second;
    const double want_lo = 90.0;
    const double want_hi = 110.0;
    if (std::abs(lo - want_lo) > 1e-6 || std::abs(hi - want_hi) > 1e-6) {
      std::cerr << "runner_cpp: bootstrap_ci parity self-check failed: got ["
                << lo << ", " << hi << "], want [" << want_lo << ", " << want_hi
                << "]; the SplitMix64/bootstrap port diverges from Rust"
                << std::endl;
      return 3;
    }
  }

  uint64_t n = 0;
  bool have_n = false;
  int reps = DEFAULT_REPS;
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
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
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
  if (reps <= 0) {
    std::cerr << "runner_cpp: --reps must be a positive integer" << std::endl;
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
    lines.push_back(hll_row("synthetic", synthetic, exact, reps));
    lines.push_back(cpc_row("synthetic", synthetic, exact, reps));
    lines.push_back(theta_row("synthetic", synthetic, exact, reps));
    lines.push_back(bloom_row("synthetic", synthetic, reps));
    lines.push_back(countmin_row("synthetic", synthetic, reps));
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
    lines.push_back(hll_row(dataset, values, exact, reps));
    lines.push_back(cpc_row(dataset, values, exact, reps));
    lines.push_back(theta_row(dataset, values, exact, reps));
    lines.push_back(bloom_row(dataset, values, reps));
    lines.push_back(countmin_row(dataset, values, reps));
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
