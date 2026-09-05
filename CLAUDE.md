# CLAUDE.md

## Active work: photogrammetry pipeline (PR #3)

Ongoing work on this fork adds a `gsplat.photogrammetry` subpackage — SfM ->
bundle adjustment -> dense MVS -> Gaussian-splat training -> mesh extraction ->
photometric refinement -> cull -> decimate -> texture -> normal/AO maps, with
AI-assisted priors, GOF-style level-set extraction, and per-stage automatic
quality metrics.

**Before continuing that work, read
[`docs/handoff/CURRENT_PLAN.md`](docs/handoff/CURRENT_PLAN.md)** — the live
plan, with what to do next and how to verify it — and
[`docs/handoff/README.md`](docs/handoff/README.md), which indexes the six short
documents that together are the complete picture: scope, scaffolding, progress
(including what was *executed* versus only *reviewed*), and the current issues.
A new session should be able to read only those and continue immediately.

Note that `docs/photogrammetry_texturing_plan.md` is a **finished** plan kept
as a record; `docs/handoff/CURRENT_PLAN.md` is the live one.

**[`docs/handoff/ISSUES.md`](docs/handoff/ISSUES.md) is the one to read
carefully.** Several of its entries are measurements that invert the obvious
intuition — the success metric for per-face view selection is contrast, not
error against ground truth, and the naive test fails. It also records a
recurring testing failure worth knowing before you write a test here: **six
times**, a test proved a mechanism worked while its call site went unpinned —
twice while building the most recent work, and both times the mechanism looked
thoroughly tested. §5b-§5d also record three methods evaluated on measurement
rather than expectation, one of which is deliberately *not* shipped.

Longer-form background, for depth rather than orientation:
[`docs/photogrammetry.md`](docs/photogrammetry.md) (how to *use* the pipeline),
[`docs/photogrammetry_status.md`](docs/photogrammetry_status.md) (the running
log), [`docs/photogrammetry_texturing_plan.md`](docs/photogrammetry_texturing_plan.md)
(the texturing design record).

## Conventions this repo already follows

- **Formatting:** `black` pinned at `22.3.0` — run
  `python -m black --check --required-version 22.3.0 <files>` on anything you
  touch. `lint/format-code.sh` wraps the repo-wide pass.
- **Tests:** `pytest` from the repo root (`pytest.ini` sets `pythonpath = .`
  and requires the `pytest-check` plugin). There is **no CI on this fork** —
  Actions is disabled at the repo level — so validate by hand.
- **`py_compile` is not enough for `examples/*.py`:** it compiles without
  executing, so it misses a `NameError` in a `tyro` dataclass's annotations
  that breaks the script on import. Also run
  `cd examples && PYTHONPATH=<repo> python -c "import <script>"` for each
  example script you touch.
- **No bundled model-running code.** gsplat consumes precomputed output from
  external AI models (depth estimators, segmenters, neural SfM) via files
  rather than shipping code that runs them — see
  `docs/source/proposals/gsharp_v0_2_port.rst` and
  `docs/source/examples/dynamic_surgical.rst`. Follow this when adding
  AI-assisted features.
- **External tools stay external.** `dense_mvs.py` shells out to the real
  `colmap` CLI rather than reimplementing patch-match stereo.
