# `gsplat.relight` — module contract

Implements the ATLAS relighting model. The full design, its positioning against
the published state of the art, and the phase gates are in
[`docs/relighting-atlas.md`](../../docs/relighting-atlas.md). This file is the
contract for the code.

## Scope

**In:** the representation of light transport, the projection of illumination
onto a basis, the two render paths and their equivalence, the near-field flash
model, offline atom prefiltering, optimal basis compression, the inverse
lighting solve, and the capture-side mathematics a loader will call — chrome-
sphere light rays, the bracket-offset fit, ambient subtraction and the held-out
splits.

**Out:** rasterisation (that is `gsplat.rasterization`), training loops (those
live in `examples/`), file I/O and dataset classes, and anything that needs a
GPU.

## Invariants

These are what the tests in `tests/relight/` exist to hold. Breaking any of them
breaks the method, not just an implementation.

1. **Outgoing radiance is linear in the illumination.** Every term in the model
   is of the form `sum_k M[c,k] * ell[c,k]`. Nothing may be added that is not.
2. **Compositing weights never depend on the light.** This is what licenses
   exchanging the order of shading and compositing.
3. **Therefore the two render paths are the same computation.**
   `composite(contract(M, ell)) == contract_screen(composite(M), ell)`, to
   floating-point rounding. Gate: `< 1e-5` in float32, `< 1e-12` in float64.
4. **Superposition is structural.** `render(E1 + E2) == render(E1) + render(E2)`
   with no penalty term anywhere. Any residue is rounding.
5. **The two projections agree.** `project_point_light` and
   `project_environment` are the same inner product. A delta light concentrated
   in one texel projects identically through both.
6. **Prefiltering commutes with the atom expansion.**
   `prefilter(sum_k ell_k A_k) == sum_k ell_k prefilter(A_k)`.
7. **A light calibration reports whether to believe it.** `solve_flash_offset`
   returns a condition number because a degenerate capture produces a wrong
   answer with a zero residual.
8. **Compression is optimal.** The basis rotation from `compress_transport` is
   the Eckart--Young truncation; no other basis of the same size does better.

## Layout

| File | Holds |
| --- | --- |
| `functional/atoms.py` | The light basis and the two projections onto it |
| `functional/transport.py` | Contraction, packing, compositing — the exactness argument |
| `functional/nearfield.py` | Flash direction, falloff and emission profile |
| `functional/prefilter.py` | Offline per-atom roughness prefiltering |
| `functional/compress.py` | Transport spectrum and optimal rank reduction |
| `functional/inverse.py` | Non-negative least-squares light recovery |
| `functional/calibration.py` | Chrome-sphere light rays, bracket-offset fit, ambient subtraction |
| `functional/splits.py` | Deterministic farthest-point held-out views and lights |

## Conventions

- Transport is `[N, 3, B]`; light coefficients are `[3, B]` shared, or
  `[N, 3, B]` when every primitive sees its own light (near-field training).
- Packed splat channels are **channel-major**: `c * B + k`. The rasteriser
  contract depends on this, so `pack_transport` / `unpack_transport` are the
  only places that should know it.
- Equirectangular maps are `[H, W, C]`, row 0 at `+z`, azimuth increasing
  across columns. `equirect_directions` is the single source of truth.
- Every public function validates its shapes and raises `ValueError` with the
  offending shape in the message. Each guard has its own named test, so that
  deleting a check fails a test that says what was deleted.

## Dependencies

PyTorch only. No SciPy, no new required dependency — the non-negative
least-squares solver is hand-rolled for exactly this reason, following the
`SCOPE.md` rule on the photogrammetry branch.

The subpackage does **not** touch the CUDA backend and is not re-exported from
`gsplat/__init__.py`. Import it explicitly:

```python
from gsplat.relight import functional as F
```

## Running the tests

```bash
python -m pytest tests/relight/ -q
```

144 tests, about 18 seconds, CPU only. They need `torch` and `pytest` and
nothing else; `import gsplat` prints "No CUDA toolkit found" on a CPU box and
otherwise behaves.

## What is deliberately not here yet

`models/` is empty. The parameter block and the trainer-facing model belong
there, but they are meaningless until the dataset contract (P0) and the trainer
(P2) exist, and writing them first would be guessing at an interface. The
functional layer is complete and tested on its own terms, which is what P1 was
for.
