SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/python -m pip
REPORT_INPUT ?= results/runs
REPORT_OUTPUT ?= $(REPORT_INPUT)/analysis
NATIVE_VALIDATION_INPUT ?= results/native_recheck_runs/validation
NATIVE_SUMMARY ?= results/native_recheck_evidence/validation_summary.json
NATIVE_GATE ?= results/native_recheck_evidence/validation_gate.json
NATIVE_CORRECTNESS ?= results/native_recheck_evidence/native_correctness.json
NATIVE_JUNIT ?= results/native_recheck_evidence/native_correctness_junit.xml
NATIVE_REPRESENTATIVE_RAW ?= results/native_recheck_evidence/validation_representative_raw.json
NATIVE_FINAL_POLICY ?= configs/native_recheck_final_policy.json
NATIVE_FINAL_LOCK ?= results/native_recheck_evidence/final_lock.json
NATIVE_FINAL_CONFIG ?= results/native_recheck_evidence/final_config.locked.json
NATIVE_FINAL_DECISION ?= results/native_recheck_evidence/final_decision.json
NATIVE_FINAL_SUMMARY ?= results/native_recheck_evidence/final_summary.json
NATIVE_FINAL_GATE ?= results/native_recheck_evidence/final_gate.json
NATIVE_PRE_HNSW_AUTHORIZATION ?= results/native_recheck_evidence/pre_hnsw_authorization.json
NATIVE_FINAL_HNSW ?= results/native_recheck_runs/final_hnsw
NATIVE_TEST_COMMAND := $(PYTHON) -m pytest -m "not evaluation or native_fixed_seed" --junitxml=$(NATIVE_JUNIT)

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: setup test test-full smoke data evaluate report native-test native-smoke native-validate native-evaluate native-report clean-generated

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

native-test:
	$(PYTHON) scripts/write_native_correctness_evidence.py \
		--output "$(NATIVE_CORRECTNESS)" --command '$(NATIVE_TEST_COMMAND)' \
		--mark-running
	@set +e; $(NATIVE_TEST_COMMAND); test_status=$$?; set -e; \
		$(PYTHON) scripts/write_native_correctness_evidence.py \
			--junit "$(NATIVE_JUNIT)" --output "$(NATIVE_CORRECTNESS)" \
			--command '$(NATIVE_TEST_COMMAND)' --pytest-exit-code $$test_status; \
		exit $$test_status

native-smoke:
	$(PYTHON) scripts/run_native_recheck.py --config configs/native_recheck_smoke.json
	$(PYTHON) scripts/analyze_native_recheck.py \
		--input results/native_recheck_runs/smoke-v3 \
		--output results/native_recheck_runs/smoke-v3/analysis/summary.json \
		--gate-output results/native_recheck_runs/smoke-v3/analysis/gate.json

native-validate:
	$(PYTHON) scripts/run_native_recheck.py \
		--config configs/native_recheck_validation.json --resume
	$(PYTHON) scripts/analyze_native_recheck.py \
		--input "$(NATIVE_VALIDATION_INPUT)" --output "$(NATIVE_SUMMARY)" \
		--gate-output "$(NATIVE_GATE)" --bootstrap-resamples 5000 \
		--bootstrap-seed 6202052 \
		--representative-raw-output "$(NATIVE_REPRESENTATIVE_RAW)"

native-evaluate:
	$(PYTHON) scripts/decide_native_final.py \
		--gate "$(NATIVE_GATE)" --summary "$(NATIVE_SUMMARY)" \
		--validation-input "$(NATIVE_VALIDATION_INPUT)" \
		--validation-config configs/native_recheck_validation.json \
		--correctness "$(NATIVE_CORRECTNESS)" \
		--holdout results/native_recheck_evidence/holdout_manifest.json \
		--policy "$(NATIVE_FINAL_POLICY)" \
		--decision-output "$(NATIVE_FINAL_DECISION)" \
		--lock-output "$(NATIVE_FINAL_LOCK)" \
		--final-config-output "$(NATIVE_FINAL_CONFIG)" \
		--final-summary-output "$(NATIVE_FINAL_SUMMARY)" \
		--final-gate-output "$(NATIVE_FINAL_GATE)" \
		--pre-hnsw-authorization-output "$(NATIVE_PRE_HNSW_AUTHORIZATION)" \
		--hnsw-output "$(NATIVE_FINAL_HNSW)"
	$(PYTHON) scripts/run_native_final.py \
		--decision "$(NATIVE_FINAL_DECISION)" \
		--lock "$(NATIVE_FINAL_LOCK)" --config "$(NATIVE_FINAL_CONFIG)" \
		--hnsw-script scripts/run_native_hnsw_references.py \
		--hnsw-output "$(NATIVE_FINAL_HNSW)"
	$(PYTHON) scripts/analyze_native_final.py \
		--decision "$(NATIVE_FINAL_DECISION)" \
		--lock "$(NATIVE_FINAL_LOCK)" --config "$(NATIVE_FINAL_CONFIG)" \
		--hnsw-input "$(NATIVE_FINAL_HNSW)" \
		--summary-output "$(NATIVE_FINAL_SUMMARY)" \
		--gate-output "$(NATIVE_FINAL_GATE)"

native-report:
	$(PYTHON) scripts/report_native_recheck.py \
		--input "$(NATIVE_VALIDATION_INPUT)" --summary "$(NATIVE_SUMMARY)" \
		--gate "$(NATIVE_GATE)" --correctness "$(NATIVE_CORRECTNESS)" \
		--profile results/native_recheck_evidence/development_profile_old.json \
		--representative-raw "$(NATIVE_REPRESENTATIVE_RAW)" \
		--final-decision "$(NATIVE_FINAL_DECISION)" \
		--final-summary "$(NATIVE_FINAL_SUMMARY)" \
		--final-gate "$(NATIVE_FINAL_GATE)" \
		--final-lock "$(NATIVE_FINAL_LOCK)" \
		--final-config "$(NATIVE_FINAL_CONFIG)" \
		--pre-hnsw-authorization "$(NATIVE_PRE_HNSW_AUTHORIZATION)" \
		--final-hnsw "$(NATIVE_FINAL_HNSW)" \
		--output reports/NATIVE_KERNEL_RECHECK_ja.md

clean-generated:
	@echo "Generated runs are intentionally retained; remove an explicit run directory manually."
