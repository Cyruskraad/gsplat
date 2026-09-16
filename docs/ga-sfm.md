# Geometric-algebra structure from motion

Experimental. Lives in `gsplat/contrib/ga`, install with `pip install gsplat[ga]`.

## Why

Rigid motion here is represented by **motors** in 3D projective geometric algebra
(PGA), `Cl(3,0,1)`. Motors are isomorphic to unit dual quaternions and PGA
bivectors are exactly `se(3)`, so none of this is a new estimator — it is the
same geometry in different coordinates.

That isomorphism is the point of the design, not a caveat to hide. Because GA
and the usual Lie-group formulation describe the same optimum, **GA cannot buy
accuracy for free**, and any claim that it does is measuring implementation
quality rather than mathematics. What it *can* buy is structural:

- **One operator for every grade.** Points (grade 3), lines (grade 2) and planes
  (grade 1) all transform by the same sandwich `M X ~M`. Joint point + line +
  plane bundle adjustment goes through one code path instead of three
  special-cased ones with separately derived Jacobians.
- **Unconstrained pose parameterization.** A motor is `exp` of a 6-component
  bivector, with no normalization constraint and no quaternion sign ambiguity.
- **Screw interpolation for free.** `exp` of a bivector *is* a screw motion, the
  natural interpolant for rolling shutter and rig trajectories.

Every claim above is meant to be measured against a matched quaternion/`se(3)`
baseline in the same harness. Anything that cannot be measured that way should
not be claimed.

## Conventions

These are fixed once, here, and asserted in `tests/ga/test_algebra.py`. Silent
convention drift is the most common way geometric-algebra code goes subtly
wrong, so the tests pin the *geometric meaning*, not just round trips.

**Algebra.** `Cl(3,0,1)`: `e1, e2, e3` square to `+1`; the degenerate `e0`
squares to `0`.

| Object | Grade | Encoding |
| ------ | ----- | -------- |
| plane  | 1     | `a*e1 + b*e2 + c*e3 + d*e0` for `ax+by+cz+d=0` |
| line   | 2     | bivector: the join of two points, or the meet of two planes |
| point  | 3     | `e123 - x*e023 + y*e013 - z*e012` for `(x,y,z)` |
| motor  | even  | scalar + bivector + pseudoscalar (8 coefficients) |

**Axes.** The rotational bivector blades map as `e23 -> x`, `-e13 -> y`,
`e12 -> z`, right-handed.

**Incidence.** For a normalized plane and point, `plane ^ point` is the *signed
distance* (times the pseudoscalar) — not merely zero-on-incidence. `plane ^
plane` meets two planes in their common line; `point & point` joins two points
into the line through them.

**Half-angle.** Motors carry the same half-angle convention as quaternions: a
rotation of `alpha` about unit axis `n` followed by translation `t` has
bivector `w = -alpha/2 * n`, `v = -t/2`. Equivalently, the bivector `[w, v]`
corresponds to the `se(3)` twist `(-2w, -2v)`; that correspondence is the test
oracle in `tests/ga/_helpers.py`.

**Tensor layouts.** The public API is plain `torch.Tensor`, so the algebra
backend stays swappable:

- bivector `(..., 6)`: `[wx, wy, wz, vx, vy, vz]`
- motor `(..., 8)`: `[1, e01, e02, e03, e12, e13, e23, e0123]`
- point `(..., 3)`, plane `(..., 4)` as `[a,b,c,d]`, line `(..., 6)`

**Intrinsics are not in the algebra.** A pinhole projection is projective, not a
versor, so it cannot be a sandwich product. Extrinsics are motors; intrinsics
stay a separate linear map.

## Implementation notes

**The screw exponential.** A general PGA bivector is a *screw* and is not
simple, so the plain `cos + sin` rotor formula does not apply (kingdon's generic
`exp` rejects it outright). We use the invariant decomposition: split into a
simple part `B_s = omega + v_perp` (squares to the scalar `-theta^2`) and a null
part `B_p = v_par` (squares to zero). They commute, so

```
exp(B) = (cos(theta) + sinc(theta) * B_s) * (1 + B_p)
```

**The logarithm's branch.** Motors double-cover SE(3), so `log` canonicalizes to
the `scalar >= 0` hemisphere. The screw-parallel translation is then recovered
from the **pseudoscalar** coefficient (which equals `pitch * sin(theta)`), not
from `cos(theta)`: the `cos(theta)` route divides by zero at a 180-degree
rotation, which is an ordinary pose rather than an edge case.

**Regularization.** `theta = sqrt(|w|^2 + 1e-24)` keeps the axis direction
differentiable through `w = 0`. The constant sits *inside* the square root, so
it perturbs an ordinary `theta` by only `1e-24 / (2*theta)`. Using `1e-12` there
instead — the obvious choice — would put a ~1e-12 error floor on every rotation,
large enough to show up in exp/log round trips.

## Testing

The GA layer is plain PyTorch and its tests are CPU-only by design, unlike
`gsplat.geometry`, whose operators are CUDA-only and whose suites skip without a
device.

```bash
pip install -e ".[ga]"
pytest tests/ga -q
```

Two independent oracles guard the core:

1. **A 4x4 matrix exponential** written out in `tests/ga/_helpers.py` (not taken
   from SciPy, so the oracle cannot share a bug with the code under test).
   Motors are isomorphic to SE(3), so machine-precision agreement is the correct
   bar — anything less is a bug in the GA code, not a difference of formulation.
2. **The `clifford` library**, an independent implementation of the same
   algebra. Note it orders the degenerate basis vector *first*
   (`sig == [0,1,1,1]`), the opposite of kingdon, so blades are matched by their
   set of basis indices with an explicit permutation sign.

Edge cases are enumerated rather than sampled: identity, pure rotation, pure
translation, translation purely along the screw axis, rotations small enough to
underflow the axis, the branch edge, and large translations.
