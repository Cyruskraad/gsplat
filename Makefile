# Everything here runs on CPU unless the target says GPU.
PYTHON ?= python

.PHONY: help test lint format smoke-cpu imports bench check

help:
	@echo "test       - full CPU test suite (no GPU, no gsplat needed)"
	@echo "lint       - black --check, the repo's only formatter"
	@echo "format     - apply black"
	@echo "imports    - import every module that has tyro annotations"
	@echo "smoke-cpu  - whole trainer on a toy scene via the reference renderer"
	@echo "gpu        - rasteriser parity tests; needs CUDA + gsplat"
	@echo "bench      - contraction throughput and memory, into the ledger"
	@echo "check      - lint + imports + test, the same set CI runs"

test:
	$(PYTHON) -m pytest tests/ -q

# Needs CUDA and gsplat. Skips cleanly without them, which is why `test` above
# can safely include the same directory.
gpu:
	$(PYTHON) -m pytest tests/gpu -q

# Throughput and peak memory of the contraction, into the ledger. Small sizes
# here; the sizes that decide B need the GPU runner.
bench:
	$(PYTHON) -m atlas.bench --report runs/bench-ledger.jsonl

lint:
	$(PYTHON) -m black --check --required-version 22.3.0 atlas/ tests/

format:
	$(PYTHON) -m black --required-version 22.3.0 atlas/ tests/

# py_compile does not evaluate a tyro dataclass's annotations, so a missing
# import there compiles cleanly and breaks on first run. This catches it.
imports:
	$(PYTHON) -c "import atlas, atlas.functional, atlas.ply, atlas.model, atlas.config, atlas.run, atlas.eval, atlas.imageio, atlas.bench, atlas.reference, atlas.tools.inspect_capture"

smoke-cpu:
	@echo "not yet implemented -- lands with atlas/train.py (T4)"; exit 1

check: lint imports test
