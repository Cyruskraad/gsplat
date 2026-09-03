# CLAUDE.md

## Active work: photogrammetry pipeline (PR #3)

Ongoing work on this fork adds a `gsplat.photogrammetry` subpackage — SfM ->
bundle adjustment -> dense MVS -> Gaussian-splat training -> mesh extraction,
with AI-assisted priors and per-stage automatic quality metrics.

**Before continuing that work, read [`docs/photogrammetry_status.md`](docs/photogrammetry_status.md).**
It is the handoff document: what has been built, what was actually executed
versus verified by code review only, what's blocked, and what to do next. Its
"START HERE" section says exactly how to pick it up.

Feature documentation (how to *use* the pipeline) lives separately in
[`docs/photogrammetry.md`](docs/photogrammetry.md).

## Conventions this repo already follows

- **Formatting:** `black` pinned at `22.3.0` — run
  `python -m black --check --required-version 22.3.0 <files>` on anything you
  touch. `lint/format-code.sh` wraps the repo-wide pass.
- **Tests:** `pytest` from the repo root (`pytest.ini` sets `pythonpath = .`
  and requires the `pytest-check` plugin).
- **No bundled model-running code.** gsplat consumes precomputed output from
  external AI models (depth estimators, segmenters, neural SfM) via files
  rather than shipping code that runs them — see
  `docs/source/proposals/gsharp_v0_2_port.rst` and
  `docs/source/examples/dynamic_surgical.rst`. Follow this when adding
  AI-assisted features.
- **External tools stay external.** `dense_mvs.py` shells out to the real
  `colmap` CLI rather than reimplementing patch-match stereo.
