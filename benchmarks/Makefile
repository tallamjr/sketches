# Benchmark orchestration: run the four standalone runners, then compare and
# gate their results with the Python reporter.
#
# Each runner is its own standalone cargo project (or, for C++, a cmake build;
# for Python, a runner.py driving the maturin wheel and pip datasketches).
# The reporter and gate targets tolerate a missing plane CSV: they compare
# whatever result CSVs actually exist.

N ?= 1000000
REPS ?= 30

PYTHON := ../.venv/bin/python
REPORTER := reporter
RESULTS := results
PLOTS := $(RESULTS)/plots

RMSE_TRIALS ?= 100
RMSE_N ?= 100000

.PHONY: ours apache-rust cpp python report memory gate rmse baseline compare clean

ours:
	cd runner-ours && cargo run --release -- --n $(N) --reps $(REPS) --out ../$(RESULTS)/ours.csv

apache-rust:
	cd runner-apache-rust && cargo run --release -- --n $(N) --reps $(REPS) --out ../$(RESULTS)/apache-rust.csv

cpp:
	cmake -S runner-cpp -B runner-cpp/build
	cmake --build runner-cpp/build
	runner-cpp/build/runner_cpp --n $(N) --reps $(REPS) --out $(RESULTS)/cpp.csv

# Python plane: build the wheel first with
#   ../.venv/bin/maturin develop --release --features extension-module
# then run both impls (our wheel vs the pip datasketches package). The runner
# is launched from runner-python/, so PYTHON (relative to benchmarks/) gets an
# extra `../` to stay anchored at the repo .venv.
python:
	cd runner-python && ../$(PYTHON) runner.py --impl ours --n $(N) --reps $(REPS) --out ../$(RESULTS)/python-ours.csv
	cd runner-python && ../$(PYTHON) runner.py --impl apache --n $(N) --reps $(REPS) --out ../$(RESULTS)/python-apache.csv

# Compare whatever result CSVs exist, print the table, and render plots.
report:
	@csvs=""; \
	for f in $(RESULTS)/ours.csv $(RESULTS)/apache-rust.csv $(RESULTS)/cpp.csv $(RESULTS)/python-ours.csv $(RESULTS)/python-apache.csv; do \
		if [ -f "$$f" ]; then csvs="$$csvs $$f"; fi; \
	done; \
	if [ -z "$$csvs" ]; then echo "no result CSVs found in $(RESULTS)/" >&2; exit 1; fi; \
	$(PYTHON) $(REPORTER)/report.py $$csvs; \
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(REPORTER)'); import plots, report; \
rows=report.load_rows('$$csvs'.split()); print(plots.render_plots(rows,'$(PLOTS)'))"

# Memory is emitted inline by every runner as the live_bytes column and appears
# in the standard report table and memory.png. This is an alias for `report`.
memory: report

# Run our runner, then gate its accuracy against the per-sketch thresholds.
gate: ours
	$(PYTHON) $(REPORTER)/report.py --check-accuracy $(REPORTER)/thresholds.json $(RESULTS)/ours.csv

# Run each runner in multi-trial RMSE mode, then render the RMSE parity table
# and rmse.png comparing each implementation against the 1/sqrt(k) floor.
# RMSE mode does not use the warmup+reps protocol, so --reps is not passed here.
rmse:
	cd runner-ours && cargo run --release -- --trials $(RMSE_TRIALS) --n $(RMSE_N) --out ../$(RESULTS)/rmse_ours.csv
	cd runner-apache-rust && cargo run --release -- --trials $(RMSE_TRIALS) --n $(RMSE_N) --out ../$(RESULTS)/rmse_apache-rust.csv
	cmake -S runner-cpp -B runner-cpp/build
	cmake --build runner-cpp/build
	runner-cpp/build/runner_cpp --trials $(RMSE_TRIALS) --n $(RMSE_N) --out $(RESULTS)/rmse_apache-cpp.csv
	$(PYTHON) $(REPORTER)/report.py --rmse $(RESULTS)/rmse_*.csv

# Snapshot the current `ours` result CSV as the committed baseline that future
# `make compare` runs are checked against.
baseline: ours
	cp $(RESULTS)/ours.csv $(RESULTS)/baseline-ours.csv
	@echo "baseline snapshot written to $(RESULTS)/baseline-ours.csv"

# Re-run `ours`, then print the per-plane separation verdict against the
# committed baseline. Errors if no baseline has been snapshotted yet.
compare: ours
	@if [ ! -f $(RESULTS)/baseline-ours.csv ]; then \
		echo "no baseline; run 'make baseline' first" >&2; exit 1; fi
	$(PYTHON) $(REPORTER)/report.py --compare $(RESULTS)/baseline-ours.csv $(RESULTS)/ours.csv

clean:
	rm -f $(RESULTS)/*.csv
	rm -rf $(PLOTS)
