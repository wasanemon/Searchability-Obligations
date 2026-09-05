SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
REPORT_INPUT ?= results/runs
REPORT_OUTPUT ?= $(REPORT_INPUT)/analysis

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: setup test test-full smoke data evaluate report clean-generated

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
	$(PIP) install --requirement requirements-lock.txt
	$(PIP) install --no-build-isolation --no-deps --editable .
	$(PYTHON) scripts/check_environment.py

test:
	$(PYTHON) -m pytest -m "not evaluation"

test-full:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/run_experiment.py --config configs/smoke.json
	$(PYTHON) scripts/analyze_results.py --input results/smoke --evidence-role calibration

data:
	$(PYTHON) scripts/download_datasets.py --config configs/data.json

evaluate:
	$(PYTHON) scripts/run_experiment.py --config configs/evaluate.json --resume
	$(PYTHON) scripts/run_experiment.py --config configs/evaluate_delta100k.json --resume

report:
	$(PYTHON) scripts/analyze_results.py --input "$(REPORT_INPUT)" \
		--output "$(REPORT_OUTPUT)" --evidence-role final \
		--immutable-evidence results/final_evidence/summary.json \
		--evidence-manifest results/final_evidence/manifest.json

clean-generated:
	@echo "Generated runs are intentionally retained; remove an explicit run directory manually."
