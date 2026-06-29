"""Python benchmark plane: our maturin wheel vs the pip `datasketches` package.

Mirrors the Rust runners (runner-ours, runner-apache-rust): a fresh sketch is
built and fully populated each timed rep, one untimed warmup pass precedes the
timed reps, and per-object heap is measured via tracemalloc. Each perf-core
sketch is emitted as one CSV row using the shared 15-column schema.

Run with `--impl ours` to measure the wheel (module `sketches`) or
`--impl apache` to measure the pip `datasketches` package. A missing whole
library fails loudly; an individual sketch type absent from the chosen library
is skipped with a note to stderr (a legitimate breadth difference).
"""
import argparse
import sys

import bench_common as bc

# Parameters chosen to match the Rust runners (runner-ours/src/lib.rs):
#   HLL lg_k = 12, Theta k = 4096, CPC lg_k = 12, Bloom error_rate = 0.01,
#   Count-Min width = 2048 / depth = 5, KLL k = 200.
HLL_LG_K = 12
THETA_K = 4096
CPC_LG_K = 12
BLOOM_ERROR_RATE = 0.01
COUNTMIN_WIDTH = 2048
COUNTMIN_DEPTH = 5
KLL_K = 200

# Count-Min point-query convention from the Rust runner: every distinct item is
# incremented once, then a hot key is incremented HOT_KEY_COUNT times and that
# key is queried. exact = HOT_KEY_COUNT.
HOT_KEY = "__hot__"
HOT_KEY_COUNT = 1000


def fmt(value):
    """Render an optional float field: empty string for None, else %.6f."""
    if value is None:
        return ""
    return f"{value:.6f}"


def make_row(impl, sketch, dataset, op, n, reps, median, stddev, ci_low, ci_high,
             nbytes, live_bytes, estimate, exact, rel_error):
    """Assemble one CSV line in the 15-column HEADER order."""
    return ",".join([
        impl,
        sketch,
        dataset,
        op,
        str(n),
        str(reps),
        f"{median:.6f}",
        f"{stddev:.6f}",
        f"{ci_low:.6f}",
        f"{ci_high:.6f}",
        "" if nbytes is None else str(nbytes),
        "" if live_bytes is None else str(live_bytes),
        fmt(estimate),
        fmt(exact),
        fmt(rel_error),
    ])


# --------------------------------------------------------------------------
# Per-sketch row builders. Each takes the impl label, the prepared dataset
# (list of string keys), N, exact cardinality, reps, and a builder-bundle that
# closes over the library-specific constructors/methods. They return a CSV row
# string (or raise to be caught and skipped-with-note by the caller).
# --------------------------------------------------------------------------

def distinct_count_row(impl, sketch_name, keys, n, exact, reps, build, estimate_of, bytes_of):
    """Generic distinct-counter row (HLL, Theta, CPC). op = distinct_count."""
    median, stddev, ci_low, ci_high = bc.timed_throughput_rounds(
        reps, bc.REPS_PER_ROUND, n, build)
    obj, live = bc.measure_live(build)
    estimate = estimate_of(obj)
    rel_error = abs(estimate - exact) / exact
    nbytes = bytes_of(obj)
    return make_row(impl, sketch_name, "synthetic", "distinct_count", n, reps,
                    median, stddev, ci_low, ci_high, nbytes, live, estimate,
                    exact, rel_error)


def bloom_row(impl, keys, n, reps, build, bytes_of):
    """Bloom build row. A membership filter has no cardinality estimate, so
    estimate/exact/rel_error are left empty. op = build."""
    median, stddev, ci_low, ci_high = bc.timed_throughput_rounds(
        reps, bc.REPS_PER_ROUND, n, build)
    obj, live = bc.measure_live(build)
    nbytes = bytes_of(obj)
    return make_row(impl, "bloom", "synthetic", "build", n, reps,
                    median, stddev, ci_low, ci_high, nbytes, live, None, None, None)


def countmin_row(impl, keys, n, reps, build, estimate_hot, bytes_of):
    """Count-Min point-query row. Each key is incremented once, the hot key
    HOT_KEY_COUNT times; the hot key is queried. op = point_query,
    n = N + HOT_KEY_COUNT, exact = HOT_KEY_COUNT."""
    total_ops = n + HOT_KEY_COUNT
    median, stddev, ci_low, ci_high = bc.timed_throughput_rounds(
        reps, bc.REPS_PER_ROUND, total_ops, build)
    obj, live = bc.measure_live(build)
    estimate = float(estimate_hot(obj))
    exact = float(HOT_KEY_COUNT)
    rel_error = abs(estimate - exact) / exact
    nbytes = bytes_of(obj)
    return make_row(impl, "countmin", "synthetic", "point_query", total_ops,
                    reps, median, stddev, ci_low, ci_high, nbytes, live,
                    estimate, exact, rel_error)


def kll_row(impl, n, reps, build, median_of, bytes_of):
    """KLL median row over the numeric range 0..n. The exact median is n/2.
    op = quantile_median."""
    median, stddev, ci_low, ci_high = bc.timed_throughput_rounds(
        reps, bc.REPS_PER_ROUND, n, build)
    obj, live = bc.measure_live(build)
    estimate = median_of(obj)
    exact = n / 2.0
    rel_error = abs(estimate - exact) / exact
    nbytes = bytes_of(obj)
    return make_row(impl, "kll", "synthetic", "quantile_median", n, reps,
                    median, stddev, ci_low, ci_high, nbytes, live, estimate,
                    exact, rel_error)


# --------------------------------------------------------------------------
# Implementation drivers. Each returns a list of CSV row strings; an absent
# sketch type is skipped with a stderr note.
# --------------------------------------------------------------------------

def note_skip(sketch_name, reason):
    print(f"note: skipping {sketch_name} ({reason})", file=sys.stderr)


def run_ours(keys, n, reps):
    try:
        import sketches
    except ImportError:
        print(
            "error: cannot import the `sketches` wheel for --impl ours. "
            "Build it first with:\n"
            "    .venv/bin/maturin develop --release --features extension-module",
            file=sys.stderr,
        )
        sys.exit(1)

    exact = float(n)
    rows = []

    def hll_build():
        s = sketches.HllSketch(HLL_LG_K)
        for k in keys:
            s.update(k)
        return s

    rows.append(distinct_count_row(
        "ours", "hll", keys, n, exact, reps, hll_build,
        lambda s: s.estimate(), lambda s: len(s.to_bytes())))

    def cpc_build():
        s = sketches.CpcSketch(CPC_LG_K)
        for k in keys:
            s.update(k)
        return s

    rows.append(distinct_count_row(
        "ours", "cpc", keys, n, exact, reps, cpc_build,
        lambda s: s.estimate(), lambda s: len(s.to_bytes())))

    def theta_build():
        s = sketches.ThetaSketch(THETA_K)
        for k in keys:
            s.update(k)
        return s

    # ThetaSketch exposes no serialiser in the wheel: emit empty bytes.
    rows.append(distinct_count_row(
        "ours", "theta", keys, n, exact, reps, theta_build,
        lambda s: s.estimate(), lambda s: None))

    def kll_build():
        s = sketches.KllSketch(KLL_K)
        for i in range(n):
            s.update(float(i))
        return s

    rows.append(kll_row(
        "ours", n, reps, kll_build,
        lambda s: s.median(), lambda s: None))

    def bloom_build():
        f = sketches.BloomFilter(n, BLOOM_ERROR_RATE)
        for k in keys:
            f.add(k)
        return f

    rows.append(bloom_row(
        "ours", keys, n, reps, bloom_build,
        lambda f: f.statistics()["num_bits"] // 8))

    def countmin_build():
        s = sketches.CountMinSketch(COUNTMIN_WIDTH, COUNTMIN_DEPTH, False)
        for k in keys:
            s.increment(k)
        for _ in range(HOT_KEY_COUNT):
            s.increment(HOT_KEY)
        return s

    rows.append(countmin_row(
        "ours", keys, n, reps, countmin_build,
        lambda s: s.estimate(HOT_KEY),
        lambda s: s.statistics()["total_cells"] * 8))

    return rows


def run_apache(keys, n, reps):
    try:
        import datasketches
    except ImportError:
        print(
            "error: cannot import `datasketches` for --impl apache. "
            "Install it first with:\n"
            "    .venv/bin/pip install datasketches",
            file=sys.stderr,
        )
        sys.exit(1)

    exact = float(n)
    rows = []

    # HLL: hll_sketch(lg_k, tgt_type=HLL_8). serialize_compact() for bytes.
    if hasattr(datasketches, "hll_sketch"):
        def hll_build():
            s = datasketches.hll_sketch(HLL_LG_K)
            for k in keys:
                s.update(k)
            return s

        rows.append(distinct_count_row(
            "apache", "hll", keys, n, exact, reps, hll_build,
            lambda s: s.get_estimate(), lambda s: len(s.serialize_compact())))
    else:
        note_skip("hll", "not in datasketches")

    # CPC: cpc_sketch(lg_k). serialize() for bytes.
    if hasattr(datasketches, "cpc_sketch"):
        def cpc_build():
            s = datasketches.cpc_sketch(CPC_LG_K)
            for k in keys:
                s.update(k)
            return s

        rows.append(distinct_count_row(
            "apache", "cpc", keys, n, exact, reps, cpc_build,
            lambda s: s.get_estimate(), lambda s: len(s.serialize())))
    else:
        note_skip("cpc", "not in datasketches")

    # Theta: update_theta_sketch(lg_k). lg_k=12 gives k=4096 to match ours.
    if hasattr(datasketches, "update_theta_sketch"):
        theta_lg_k = THETA_K.bit_length() - 1  # 4096 -> 12

        def theta_build():
            s = datasketches.update_theta_sketch(theta_lg_k)
            for k in keys:
                s.update(k)
            return s

        # update_theta_sketch has no direct serializer; the compact form does,
        # but to keep the build path comparable we emit the compacted size.
        rows.append(distinct_count_row(
            "apache", "theta", keys, n, exact, reps, theta_build,
            lambda s: s.get_estimate(),
            lambda s: len(s.compact().serialize())))
    else:
        note_skip("theta", "not in datasketches")

    # KLL: kll_floats_sketch(k). get_quantile(0.5) for the median.
    if hasattr(datasketches, "kll_floats_sketch"):
        def kll_build():
            s = datasketches.kll_floats_sketch(KLL_K)
            for i in range(n):
                s.update(float(i))
            return s

        rows.append(kll_row(
            "apache", n, reps, kll_build,
            lambda s: s.get_quantile(0.5),
            lambda s: len(s.serialize())))
    else:
        note_skip("kll", "not in datasketches")

    # Count-Min: count_min_sketch(num_hashes=depth, num_buckets=width).
    if hasattr(datasketches, "count_min_sketch"):
        def countmin_build():
            s = datasketches.count_min_sketch(COUNTMIN_DEPTH, COUNTMIN_WIDTH)
            for k in keys:
                s.update(k, 1)
            s.update(HOT_KEY, HOT_KEY_COUNT)
            return s

        rows.append(countmin_row(
            "apache", keys, n, reps, countmin_build,
            lambda s: s.get_estimate(HOT_KEY),
            lambda s: s.get_serialized_size_bytes()))
    else:
        note_skip("countmin", "not in datasketches")

    # Bloom: this datasketches version exposes no bloom_filter type.
    if hasattr(datasketches, "bloom_filter"):
        def bloom_build():
            f = datasketches.bloom_filter(n, BLOOM_ERROR_RATE)
            for k in keys:
                f.update(k)
            return f

        rows.append(bloom_row(
            "apache", keys, n, reps, bloom_build,
            lambda f: len(f.serialize())))
    else:
        note_skip("bloom", "not in datasketches")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Python benchmark plane: our wheel vs pip datasketches.")
    parser.add_argument("--impl", choices=["ours", "apache"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Build the synthetic distinct dataset once, outside any timing: N distinct
    # string keys so the exact cardinality equals N (matches the Rust runners).
    keys = [str(i) for i in range(args.n)]

    if args.impl == "ours":
        rows = run_ours(keys, args.n, args.reps)
    else:
        rows = run_apache(keys, args.n, args.reps)

    with open(args.out, "w") as f:
        f.write(bc.HEADER + "\n")
        for row in rows:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
