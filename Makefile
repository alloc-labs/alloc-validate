.PHONY: all check-env setup pytorch huggingface scan-only lightning ray distributed case-study diagnose-targets validate-free validate-full validate-fleet validate-upload validate-topology docker-build docker-test matrix matrix-quick matrix-multi matrix-pytorch matrix-huggingface matrix-scan matrix-distributed eval-recommendations eval-recommendations-full baseline clean setup-gcp test test-diagnose test-callbacks test-quick dual-l4-stress

VENV_BIN ?= $(CURDIR)/.venv/bin
ALLOC_BIN ?= $(VENV_BIN)/alloc
PYTHON_BIN ?= $(VENV_BIN)/python
BOOTSTRAP_PYTHON ?= python3

check-env:
	@test -x "$(ALLOC_BIN)" || (echo "ERROR: missing $(ALLOC_BIN). Run: python3 -m venv .venv && .venv/bin/pip install -e '.[all]'" && exit 1)
	@test -x "$(PYTHON_BIN)" || (echo "ERROR: missing $(PYTHON_BIN). Run: python3 -m venv .venv && .venv/bin/pip install -e '.[all]'" && exit 1)

# Active workloads (have real tests)
all: pytorch huggingface scan-only lightning ray distributed

pytorch: check-env
	cd pytorch && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh

huggingface: check-env
	cd huggingface && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh

scan-only: check-env
	cd scan-only && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh

lightning: check-env
	cd lightning && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh

ray: check-env
	cd ray && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh

distributed: check-env
	cd distributed && ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash validate.sh ddp

case-study: check-env
	$(PYTHON_BIN) -m pytest tests/test_case_study.py -v

diagnose-targets: check-env
	$(PYTHON_BIN) -m pytest tests/test_diagnose.py -v

validate-topology: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/validate_topology.sh

validate-free:
	@unset ALLOC_TOKEN && $(MAKE) all

validate-full:
	@test -n "$(ALLOC_TOKEN)" || (echo "ERROR: ALLOC_TOKEN must be set for full validation" && exit 1)
	$(MAKE) all

docker-build:
	docker build -t alloc-validate .

docker-test:
	docker run --rm alloc-validate

matrix: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py

matrix-quick: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --quick

matrix-multi: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --include-multi-gpu

matrix-pytorch: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --framework pytorch

matrix-huggingface: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --framework huggingface

matrix-scan: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --framework scan-only

matrix-distributed: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/run_matrix.py --framework distributed

validate-fleet: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/validate_fleet.sh

validate-upload: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/validate_upload.sh

eval-recommendations: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/eval_recommendations.py

eval-recommendations-full: check-env
	@test -n "$(ALLOC_TOKEN)" || (echo "ERROR: ALLOC_TOKEN must be set for full eval" && exit 1)
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" "$(PYTHON_BIN)" scripts/eval_recommendations.py --full

baseline: check-env
	ALLOC_BIN="$(ALLOC_BIN)" PYTHON_BIN="$(PYTHON_BIN)" bash scripts/record_baseline.sh

test: check-env
	$(PYTHON_BIN) -m pytest tests/ -v

test-diagnose: check-env
	$(PYTHON_BIN) -m pytest tests/test_diagnose.py -v

test-callbacks: check-env
	$(PYTHON_BIN) -m pytest tests/test_callbacks.py -v

test-quick: check-env
	$(PYTHON_BIN) -m pytest tests/test_cli_smoke.py tests/test_ghost.py tests/test_diagnose.py tests/test_case_study.py tests/test_artifact_contract.py tests/test_config.py tests/test_scan.py tests/test_repo_hygiene.py -v

setup:
	$(BOOTSTRAP_PYTHON) bootstrap.py

setup-gcp:
	bash scripts/gpu/setup-gcp.sh


dual-l4-stress: check-env
	bash scripts/local_dual_l4_stress.sh

clean:
	rm -rf */data */hf_output */alloc_artifact.json* */lightning_logs */.alloc_callback.json
	rm -rf scan-only/ghost_output*.json scan-only/scan_*.json scan-only/*.stderr.log
	rm -rf distributed/ghost_*.json distributed/scan_topo_*.json distributed/scan_topo_*.stderr.log
