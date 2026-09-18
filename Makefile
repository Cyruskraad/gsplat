# Everything here runs on CPU unless the target says GPU.
PYTHON ?= python

.PHONY: help test lint format smoke-cpu imports check

help:
	@echo "test       - full CPU test suite (no GPU, no gsplat needed)"
	@echo "lint       - black --check, the repo's only formatter"
	@echo "format     - apply black"
	@echo "imports    - import every module that has tyro annotations"
	@echo "smoke-cpu  - whole trainer on a toy scene via the reference renderer"
	@echo "check      - lint + imports + test, what CI would run if there were CI"

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m black --check --required-version 22.3.0 atlas/ tests/

format:
	$(PYTHON) -m black --required-version 22.3.0 atlas/ tests/

# py_compile does not evaluate a tyro dataclass's annotations, so a missing
# import there compiles cleanly and breaks on first run. This catches it.
imports:
	$(PYTHON) -c "import atlas, atlas.functional, atlas.ply, atlas.model, atlas.tools.inspect_capture"

smoke-cpu:
	@echo "not yet implemented -- lands with atlas/train.py (T4)"; exit 1

check: lint imports test
