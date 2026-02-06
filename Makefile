# SIT: Spectator Interference Tomography
# Makefile for reproducible builds

PYTHON = python3
CONFIG_DIR = sit/experiments/configs
QUICK_CONFIG = $(CONFIG_DIR)/quick.yaml
FULL_CONFIG = $(CONFIG_DIR)/full.yaml

.PHONY: quick full analyze workbook report package clean help

help:
	@echo "SIT: Spectator Interference Tomography"
	@echo ""
	@echo "Commands:"
	@echo "  make quick     - Run quick pipeline (validation, ~2-5 min)"
	@echo "  make full      - Run full pipeline (publication quality)"
	@echo "  make analyze   - Re-run analysis on existing data"
	@echo "  make workbook  - Regenerate Excel workbook"
	@echo "  make report    - Regenerate report"
	@echo "  make qa        - Run QA tests"
	@echo "  make package   - Create zip artifacts"
	@echo "  make clean     - Remove generated files"
	@echo ""
	@echo "One-command runs:"
	@echo "  $(PYTHON) -m sit.experiments.run_all --config $(QUICK_CONFIG)"
	@echo "  $(PYTHON) -m sit.experiments.run_all --config $(FULL_CONFIG)"

quick:
	$(PYTHON) -m sit.experiments.run_all --config $(QUICK_CONFIG)

full:
	$(PYTHON) -m sit.experiments.run_all --config $(FULL_CONFIG)

analyze:
	$(PYTHON) -m sit.analysis.metrics --input data/raw --output results/tables

workbook:
	$(PYTHON) -c "from sit.artifacts.make_workbook import create_workbook; import yaml; c=yaml.safe_load(open('$(FULL_CONFIG)')); create_workbook(c, {})"

report:
	$(PYTHON) -c "from sit.artifacts.make_report import create_report; import yaml; c=yaml.safe_load(open('$(FULL_CONFIG)')); create_report(c, {})"

qa:
	$(PYTHON) -m sit.qa.test_fixed_ratios
	$(PYTHON) -m sit.qa.test_monotonicity
	$(PYTHON) -m sit.qa.test_recompute
	$(PYTHON) -m sit.qa.test_seed_reproducibility

package:
	@echo "Creating SIT_artifact.zip (full)..."
	zip -r SIT_artifact.zip sit/ data/ results/ Makefile requirements.txt README.md -x "*.pyc" "__pycache__/*"
	@echo "Creating SIT_artifact_light.zip (no raw samples)..."
	zip -r SIT_artifact_light.zip sit/ data/derived/ results/ Makefile requirements.txt README.md -x "*.pyc" "__pycache__/*"
	@echo "Done. Artifacts:"
	@ls -lh SIT_artifact*.zip

clean:
	rm -rf data/raw/* data/derived/*
	rm -rf results/figures/* results/tables/* results/workbook/* results/report/* results/manifest/*
	rm -f SIT_artifact*.zip
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
