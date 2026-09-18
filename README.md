# ATLAS

**All-frequency Transport via Learned Atom Splatting** — relighting a captured
object at a cost that does not depend on how complicated the illumination is.

> **Status: the mathematics is implemented and tested; nothing has rendered a
> pixel yet.** See [`HANDOVER.md`](HANDOVER.md) for exactly what is executed and
> what is not.

## The idea

One-light-at-a-time capture samples the light transport operator column by
column, and transport is linear in the illuminant. Published relightable-Gaussian
methods give that linearity away: their appearance functions are parameterised by
a single point light, so relighting under an environment map costs one render per
sampled light.

ATLAS keeps it. Each primitive's outgoing radiance is linear in the illumination
by construction, and alpha-compositing weights depend only on geometry and
opacity — never on the light. So

```
C(p) = sum_i w_i (M_i · ell) = ( sum_i w_i M_i ) · ell = T(p) · ell
```

the composited transport can be splatted once and contracted per pixel, and that
is *identical* — to floating-point rounding — to contracting per primitive and
splatting. Not the usual deferred-shading approximation; an equality, and a unit
test.

Three things follow:

1. **Relighting cost is independent of the number of lights.** Two exactly
   equivalent render paths, and the renderer picks by what changed: contract
   then splat amortises one contraction across many views; splat then contract
   makes each light change a sub-millisecond screen-space pass.
2. **Superposition is structural**, not penalised. The residue is rounding.
3. **Inverse lighting is linear least squares** over `B` unknowns — "which
   illumination makes this look like that photograph" becomes an interactive
   control.

The capture is a handheld flash, not a light stage.

## Install

```bash
pip install -e ".[dev]"            # functional layer + tests. No GPU needed.
pip install -e ".[dev,capture]"    # adds RAW/EXR/EXIF reading
pip install -e ".[dev,gpu]"        # adds gsplat; needs CUDA
```

## Use

Inspect a capture before anything else — it decides whether the data can support
relighting at all, and the loader is written against its report:

```bash
python -m atlas.tools.inspect_capture /path/to/capture -o capture_report.json
```

Validate the code:

```bash
make check     # lint + imports + 163 tests, ~20 s, CPU only
```

## Repository

```
atlas/functional/   the method: atoms, transport, near-field, prefilter,
                    compression, inverse lighting, calibration, splits.
                    Pure PyTorch, no gsplat, no CUDA.
atlas/tools/        capture inspection.
docs/               relighting-atlas.md (the design), module-contract.md
AGENTS.md           ground rules for anyone working here
HANDOVER.md         what is done, what is next, and the gate that decides
```

## Licence

Apache-2.0.
