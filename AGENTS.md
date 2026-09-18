# AGENTS.md — ground rules for anyone, human or agent, working in this repo

Read this and `HANDOVER.md` before changing anything. `docs/relighting-atlas.md`
is the design; `docs/module-contract.md` is the contract for `atlas/functional/`.

## What this is

ATLAS makes a relightable Gaussian representation whose renderer is an **exact
linear operator in the illumination**. That single property is the whole point:
it makes relighting cost independent of how many lights the illumination
contains, makes superposition structural rather than penalised, and turns
inverse lighting into a linear least-squares problem.

Published methods trained on the same kind of data (GS³, OLAT Gaussians) are
fast *per light*, not per illumination. That is the gap.

## The invariants

These are not style preferences. Breaking any of them breaks the method, and
each is pinned by a named test in `tests/`.

1. **Outgoing radiance is linear in the illumination.** Every term is
   `sum_k M[c,k] * ell[c,k]`. **Do not add a term that is not.** If you find
   yourself wanting a non-linear shading term, stop and read
   `docs/relighting-atlas.md` first — it is probably expressible as another
   atom, and if it genuinely is not, that is a design change and not a patch.
2. **Compositing weights never depend on the light.**
3. Therefore `composite(contract(M, ell)) == contract_screen(composite(M), ell)`
   to rounding. Gate: `< 1e-5` float32, `< 1e-12` float64.
4. **Superposition is structural**, never a loss term.
5. `project_point_light` and `project_environment` are the same inner product.
6. Prefiltering commutes with the atom expansion.
7. `solve_flash_offset` reports a condition number, because a degenerate capture
   yields a wrong answer with a *zero* residual.
8. `compress_transport` is the Eckart–Young truncation.

## Ground rules

Each of these exists because ignoring it caused a real defect in this project or
its predecessors.

- **CI runs on every push.** `.github/workflows/cpu.yml` does lint, imports and
  the CPU suite on a hosted runner; `.github/workflows/gpu.yml` does the
  rasteriser parity tests on the workstation. Run `make check` locally anyway —
  finding it yourself is faster than finding it in a log.
- **Run it, don't just read it.** Both defects found so far — the compositing
  weights collapsing at `alpha = 1`, and the light solver never meeting its
  stopping rule — were invisible to inspection and obvious on the first run.
- **Mutation-check every guard.** Revert the fix, confirm a named test genuinely
  fails. A test that passes with the fix reverted is not a test.
- **`make imports` is not optional.** `py_compile` does not evaluate a `tyro`
  dataclass's annotations, so a missing import there compiles cleanly and breaks
  on first run.
- **Assert each test's premise with a measured number**, and do not shrink a
  fixture past the point where the method under test still works — that is
  tuning the test to pass.
- **Be explicit about executed versus reviewed**, in commits and in
  `HANDOVER.md`. Nothing in this repo has run on a GPU yet; say so until it has.
- **Withdraw a gate that measurement breaks; do not loosen it.** This has
  already happened once: the 2 mm chrome-sphere position gate was geometrically
  unreachable, so it was removed and replaced, not relaxed. The reasoning is in
  `docs/relighting-atlas.md`.
- **No new required dependencies.** `atlas.functional` is pure PyTorch and must
  stay that way — it is what lets the correctness argument be checked without a
  GPU. RAW, EXR, viewer and gsplat all live behind optional extras. The one
  addition outside that layer is PyYAML, for configs people hand-edit.
- **Every run is reproducible from a config hash, a commit and a data hash.**
  Adding a knob means adding a field to a config dataclass — never reading an
  environment variable or a global, both of which are invisible to the hash and
  therefore to the ledger.

## Layout, and what may import what

```
atlas/functional/   pure PyTorch. Imports torch and nothing else. CPU-testable.
atlas/tools/        standard library, plus optional Pillow/rawpy that degrade.
atlas/ply.py        standard library. Reads a gsplat PLY for geometry init.
atlas/config.py     typed config tree, layered YAML, the hash that names a run.
atlas/run.py        run directories, provenance, metrics, the results ledger.
atlas/eval.py       tonemapped metrics, the two held-out splits, the gate.
atlas/imageio.py    PNG in and out, and a bitmap font. Standard library only.
atlas/model.py      RelightSplats + Path A render. gsplat imported lazily,
                    inside render() only, so the rest is CPU-testable.
atlas/data/         loader.                             [not written yet]
atlas/train.py      trainer.                            [not written yet]
atlas/render.py     offline env-map relighting.         [not written yet]
```

`atlas.functional` must never import `gsplat`, `atlas.model`, or anything that
needs a GPU. There is a test that would notice.

## Conventions

- Transport is `[N, 3, B]`; light coefficients `[3, B]` shared, or `[N, 3, B]`
  when each primitive sees its own light (near-field training).
- Packed splat channels are **channel-major** (`c * B + k`). Only
  `pack_transport` / `unpack_transport` should know this.
- Equirectangular maps are `[H, W, C]`, row 0 at `+z`. `equirect_directions` is
  the single source of truth.
- Everything is **linear radiance**. No arithmetic on gamma-encoded values, ever.
- Every public function validates its shapes and raises `ValueError` naming the
  offending shape. Each guard gets its own named test.
- `black==22.3.0`, and that exact version.

## Validating

```bash
make check                 # lint + imports + tests. Seconds. No GPU, no gsplat.
pytest tests/gpu -q        # needs CUDA and gsplat; skips cleanly without them.
```

`tests/gpu/` holds the one thing CPU cannot reach. The CPU suite proves the two
render paths agree against `atlas.functional.composite`, a compositor written to
be obviously correct — but the renderer uses `gsplat.rasterization`, a tiled CUDA
kernel with its own sort and its own accumulation. If *it* is not linear in the
per-primitive feature, the CPU proof is a proof about the wrong program. The
parity test settles that without needing to know the kernel's weights: it renders
the same scene both ways, and whatever weights the kernel chose it chose the same
ones twice.

`docs/runner-setup.md` registers the workstation so this happens on every push
rather than by hand.
