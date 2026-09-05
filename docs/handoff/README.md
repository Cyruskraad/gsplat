# Photogrammetry pipeline — handoff index

**Read this file first. It is written so that a new session can read only the
five files in this directory and be productive immediately.**

| File | What it answers |
|---|---|
| `README.md` (this) | Where things stand right now; how to pick up in five minutes |
| [`SCOPE.md`](SCOPE.md) | What this project is, what it deliberately is *not*, the conventions it follows |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The scaffolding: modules, data flow, public API, file inventory |
| [`PROGRESS.md`](PROGRESS.md) | What is built, what was *executed* vs only *reviewed*, the commit history, the bug list |
| [`ISSUES.md`](ISSUES.md) | Blockers, known limitations, and the traps that will waste your time if you rediscover them |

Deeper background, kept for detail rather than orientation:
[`../photogrammetry.md`](../photogrammetry.md) (how to *use* the pipeline),
[`../photogrammetry_status.md`](../photogrammetry_status.md) (the long-form
running log this directory summarises),
[`../photogrammetry_texturing_plan.md`](../photogrammetry_texturing_plan.md)
(the texturing work's design record).

---

## Current state, in one paragraph

A `gsplat.photogrammetry` subpackage adding the classic photogrammetry loop
around gsplat's existing Gaussian-splat training: **SfM → bundle adjustment →
dense MVS → training → mesh extraction → cull → decimate → texture → normal/AO
maps → OBJ**, orchestrated by one command, with automatic quality metrics at
every stage. It lives on one branch, as one draft PR, and has never run on a
GPU or a real capture — everything is verified against analytic ground truth on
CPU.

| | |
|---|---|
| Branch | `claude/photogrammetry-techniques-plan-jb0pod` |
| Head | `b342d5d` (33 commits, 34 files, +15024 / −10 against `main`) |
| PR | [#3](https://github.com/Cyruskraad/gsplat/pull/3) — **open draft**, mergeable clean, **no reviews, no comments, no CI** |
| Tests | **196 passing** across 12 files |
| Bugs found and fixed | 15, each mutation-checked |
| Blocking | GitHub Actions disabled at repo level; no GPU/CUDA/`colmap`; no model weights |

---

## Pick up in five minutes

**1. Confirm the tree is still green** before changing anything:

```bash
python -m pytest tests/test_bundle_adjustment.py tests/test_mesh_extraction.py \
    tests/test_neural_sfm.py tests/test_colmap_dataset.py \
    tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py \
    tests/test_texturing.py tests/test_extract_mesh_io.py \
    tests/test_extract_mesh_cli.py tests/test_photometric_alignment.py \
    tests/test_mesh_refinement.py tests/test_level_set.py -q
```

Expect **196 passed** (~2m). Needs `pycolmap`, `open3d`, `scikit-learn`,
`opencv-python-headless`, `imageio`, `piexif`, `pytest-check`.

**2. Check whether any blocker has lifted** — this decides what is worth doing:

```bash
nvidia-smi                                    # GPU?
python -c "import torch; print(torch.cuda.is_available())"
command -v colmap                             # CUDA colmap build?
# and via the GitHub tools: list_workflows on the repo (0 == Actions still off)
```

**3. Read [`ISSUES.md`](ISSUES.md).** It is the highest-value file here. Several
of its entries are measurements that *invert the obvious intuition*; rediscovering
them costs hours.

**4. Then pick work from [`ISSUES.md`](ISSUES.md) § "What to do next".**

---

## Ground rules for changing anything here

These are not stylistic preferences — each one exists because ignoring it
already caused a real defect on this branch.

- **Develop on the existing branch.** Never push elsewhere without being asked.
- **Validate by hand; there is no CI.** For anything you touch:
  ```bash
  python -m black --check --required-version 22.3.0 <files>
  python -m py_compile <files>
  cd examples && PYTHONPATH=<repo> python -c "import extract_mesh, run_pipeline"
  ```
  The last one is not optional: `py_compile` does **not** evaluate a `tyro`
  dataclass's annotations, so a missing import there compiles fine and breaks
  the script on import. That happened once.
- **Mutation-check every guard.** Revert the fix, confirm a test genuinely
  fails. A test that passes with the fix reverted is not a test. Four times on
  this branch a mechanism-level test passed while the *call site* went unpinned
  — see [`ISSUES.md`](ISSUES.md) § "The recurring testing failure".
- **Never bundle model-running code.** gsplat consumes precomputed output from
  external models via files. See [`SCOPE.md`](SCOPE.md).
- **Be explicit in commits and the PR about what you actually executed versus
  what you only reviewed.** [`PROGRESS.md`](PROGRESS.md) keeps that split, and
  it must stay honest — a large fraction of this work cannot be executed here.
