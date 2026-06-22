# Benchmark orchestration: run the three standalone runners, then compare and
# gate their results with the Python reporter.
#
# Each runner is its own standalone cargo project (or, for C++, a cmake build).
# The reporter and gate targets tolerate a missing apache/cpp CSV: they compare
# whatever result CSVs actually exist.

N ?= 1000000

PYTHON := ../.venv/bin/python
REPORTER := reporter
RESULTS := results
PLOTS := $(RESULTS)/plots

.PHONY: ours apache-rust cpp report gate clean

ours:
	cd runner-ours && cargo run --release -- --n $(N) --out ../$(RESULTS)/ours.csv

apache-rust:
	cd runner-apache-rust && cargo run --release -- --n $(N) --out ../$(RESULTS)/apache-rust.csv

cpp:
	cmake -S runner-cpp -B runner-cpp/build
	cmake --build runner-cpp/build
	runner-cpp/build/runner_cpp --n $(N) --out $(RESULTS)/cpp.csv

# Compare whatever result CSVs exist, print the table, and render plots.
report:
	@csvs=""; \
	for f in $(RESULTS)/ours.csv $(RESULTS)/apache-rust.csv $(RESULTS)/cpp.csv; do \
		if [ -f "$$f" ]; then csvs="$$csvs $$f"; fi; \
	done; \
	if [ -z "$$csvs" ]; then echo "no result CSVs found in $(RESULTS)/" >&2; exit 1; fi; \
	$(PYTHON) $(REPORTER)/report.py $$csvs; \
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(REPORTER)'); import plots, report; \
rows=report.load_rows('$$csvs'.split()); print(plots.render_plots(rows,'$(PLOTS)'))"

# Run our runner, then gate its accuracy against the per-sketch thresholds.
gate: ours
	$(PYTHON) $(REPORTER)/report.py --check-accuracy $(REPORTER)/thresholds.json $(RESULTS)/ours.csv

clean:
	rm -f $(RESULTS)/*.csv
	rm -rf $(PLOTS)
