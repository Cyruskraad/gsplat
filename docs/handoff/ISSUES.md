# Issues, blockers and traps

**This is the highest-value file in the directory.** Most entries are things a
reasonable person would get wrong on first attempt, and several are
measurements that *invert the obvious intuition*.

---

## 1. Blockers — what is stopping this from being finished

All three have held for the entire life of the branch. Re-verify them at the
start of a session rather than assuming; the check is seconds.

| Blocker | How to check | Consequence |
|---|---|---|
| **GitHub Actions disabled at the repo level** | `list_workflows` returns `total_count: 0` despite `.github/workflows/*.yml` existing | No CI has ever run. Manual validation stands in for it. The repo owner has said they lack access to enable it. |
| **No GPU / CUDA / compiled extension** | `nvidia-smi`; `torch.cuda.is_available()` | `train`, `extract_mesh` and `dense_mvs` have never run end to end. All texturing is verified against analytic ground truth instead of a real capture. |
| **No CUDA `colmap` binary** | `command -v colmap` | Dense MVS cannot run. |
| **No model weights fetchable** | Network policy allows package registries, 403s other hosts | Neither AI-prior recipe has been run against a real model. |

**The single highest-value action available to a human is enabling Actions**, or
one GPU run against a real capture. Nothing in the code is waiting on more code.

## 2. Open state on the PR

- PR [#3](https://github.com/Cyruskraad/gsplat/pull/3) is an **open draft**,
  mergeable clean, with **no reviews and no comments**. It needs a human.
- **Keep the PR body current.** It was refreshed at `4d379ed` and again at
  `181a786`. If it drifts again, rebuild it from the *live* body via
  `pull_request_read` — a saved draft in the scratchpad has drifted from it
  before, and rewriting from the draft silently reverts whatever landed since.

## 3. Known limitations (deliberate, documented, not bugs)

- **View selection cannot be combined with multi-page atlases.**
  `bake_mesh_texture` raises rather than half-supporting it: the MRF labels
  faces across the whole mesh and its seam levelling would have to run across
  page boundaries. This is the most substantial remaining feature gap.
- **View selection is a genuine tradeoff, not a strict win** — sharper but
  pointwise less accurate. It is off by default and must stay that way unless a
  real capture says otherwise.
- **`seam_discontinuity` never reaches zero.** Two samples either side of a
  border are different surface points, so the texture's own detail sets a
  floor. Read the before/after pair, never the absolute.
- **The ICM optimiser has no optimality bound** where alpha-expansion (graph
  cut) is within a known factor. Graph cut needs a max-flow solver, which would
  be a new hard dependency. The interface is kept optimiser-agnostic so one can
  be dropped in.
- **Non-square atlases and multi-material chart grouping** are unimplemented.
- **Appearance-embedding checkpoints** are out of scope for mesh extraction.

## 4. Traps — measured facts that contradict the obvious guess

Each of these cost real time to discover. Do not re-derive them.

### Texturing quality

1. **Sharpness/contrast is the success metric for view selection — *not* error
   against ground truth.** Pointwise L1 goes the *other* way (0.171 blended vs
   0.199 view-selected at 45′). Blending *attenuates* detail; single-view
   sampling *displaces* it, and a displaced-but-sharp texture scores worse
   pointwise while looking far better. A test asserting "view selection has
   lower L1" **would fail** and make the feature look worthless. The existing
   test asserts the tradeoff in the opposite direction on purpose.
2. **The premise is frequency-dependent.** Against the repo's default
   `_surface_pattern` (wavelength ≈ half the sphere) blending loses *nothing* —
   98–100% gradient retention out to 90′ — and view selection looks pointless.
   The effect only exists where detail sits near the misregistration scale.
   `tests/test_texturing.py::_high_frequency_pattern` exists for this.
3. **Seam levelling must average colour *along the shared edge*, not compare at
   the shared vertex.** Two views of one vertex disagree by **0.288** from pixel
   quantisation and silhouette bleed alone, where the exposure difference being
   corrected is 0.26 — the solve then fits noise larger than the signal and
   makes the atlas *worse* (0.184 → 0.221 on a scene with no exposure
   differences at all).
4. **`seam_discontinuity`'s inset must be in texels, not a fraction of the
   face.** A fractional inset reads **0.087 on a seam-free ground-truth atlas** —
   it measures the texture's own spatial variation as a seam.

### Geometry and sampling

5. **`face_visibility` is not `face_view_quality() > 0`.** Quality is *gradient
   energy*, so a face on a flat untextured surface scores zero however plainly
   it is in view. Measured on a flat-shaded sphere: 215 of 653 visible
   (face, view) pairs score exactly zero, and **12 of 224 faces score zero from
   every view while every camera sees them**. Culling on quality deletes surface
   out of the middle of an observed object.
6. **Evidence is the *maximum* over views, not the sum.** Photographing a wall
   twenty times is not more detail than photographing it twice from the same
   distance. Summing makes the recommended atlas grow with the shutter count.
7. **Round atlas sizes to the *nearest* power of two, not the next one up.**
   Rounding up quadruples the atlas for an arbitrarily small overshoot: exact
   518.1 → 1024 baked **3.88x** more texels than there were pixels to fill them.
8. **UV packing efficiency has no defensible constant** — 42.7%–73.2% across
   test meshes, non-monotonically in density. It *is* stable per mesh (≤2.8%
   over repeats), so measure it with a probe unwrap.

### open3d specifics

9. **`compute_uvatlas` SEGFAULTS (exit 139) on non-manifold input.** It does not
   raise. Manifoldness is checked up front; do not remove that check.
10. **`compute_uvatlas` is not deterministic.** Four unwraps of one mesh give
    four layouts. `_unwrap_and_rasterize` reuses existing `triangle_uvs` for
    this reason — every map must ride the same layout.
11. **open3d cannot write textured glTF/GLB.** It warns and produces a corrupt
    file. OBJ + MTL + PNG is the delivery format.
12. **`cast_rays`' barycentric convention is `(1-u-v, u, v)`** — pinned
    empirically, not assumed.

### Tooling

13. **Pillow cannot write 16-bit RGB PNG at all** (only 16-bit greyscale); it
    raises `TypeError: Cannot handle this data type`. Those go through OpenCV,
    which is **BGR** — a normal map written without reversing the channels loads
    fine and shades wrong.
14. **`py_compile` does not catch a `NameError` in a `tyro` dataclass's
    annotations.** It compiles without executing. You must additionally
    `import` each changed example script. This let a real break through once.
15. **tyro stops consuming a `List[str]` at the next `--`-prefixed token.** Bind
    the first element with `=`:
    `--extract_mesh_extra_args=--texture_seam_smoothness 0.25`.
16. **A convex test shape cannot test an occlusion guard.** On a sphere the bake
    is byte-identical with and without the occluder, because every face the
    cameras should not see is also facing away and back-face rejection already
    removes it. Use `tests/test_texturing.py::_two_quads`.

## 5. The recurring testing failure — read this before writing tests

**Five times on this branch, a test proved a *mechanism* worked while the
*call site* went unpinned.** Each time the mutation passed the entire suite:

| Mutation that escaped | Why the existing test missed it | What closed it |
|---|---|---|
| Cull on `quality > 0` instead of visibility | The nested-spheres scene is textured, so the two agree there | A flat-scene culling test |
| Sum projected areas instead of max | Every scaling test uses *ratios*, and a consistent over-count cancels out of a ratio | A test pinning absolute evidence against view count |
| `bake_texture_atlas_pages(occluder=None)` | The occlusion test supplies its own occluder either way | A test driving the page bake itself |
| Return an unmeasured decimation result | The first attempt produced the same mesh by coincidence | A stronger mutation (return an over-decimated mesh) |
| `extract_mesh.py` passing `seam_smoothness=` to `bake_mesh_texture()`, which has no such parameter | `bake_texture_atlas_view_selected` was tested directly and works; the flag guard checks the *outer* seam only (that every `--flag` `run_pipeline.py` emits is a `Config` field) | `tests/test_extract_mesh_cli.py` — a static check that every keyword `extract_mesh.py` passes to a `gsplat.photogrammetry` function exists on its real signature |

The fifth is the one worth dwelling on: it was **not** a subtle behavioural
regression but a `TypeError` on the default path, so from `fa70683` to `181a786`
*every* texture-baking run of `extract_mesh.py` — and `run_pipeline.py`'s whole
delivery stage — crashed. It survived because no test has ever called
`extract_mesh.main()`: `assert cfg.ckpt` runs before the method dispatch, so
reaching `main()` needs a GPU checkpoint even on the `poisson` path, which never
opens the file. **Until a checkpoint-free entry exists, `extract_mesh.py`'s
`main()` is untested end to end and can hold another one of these.**

**Fixing that gap immediately found a sixth, worse bug.** With `main()`
runnable (`--method mesh --mesh_path`, or `--method poisson` on a cloud alone),
the very first end-to-end run exposed a **silent frame mismatch**: `Parser` is
built with `normalize=True`, which moves the cameras, but the Poisson path read
its dense cloud straight off disk in the sparse model's raw frame. Nothing
raises — the mesh is simply textured from cameras that do not line up with it.
Measured: reading the mesh raw, **14.6%** of (vertex, view) pairs land in
frame; through `parser.transform`, **100%**. Skip the transform and
`--cull_unobserved` discards **68.6%** of faces while the script's own "the
poses or the scale are wrong" warning fires — a diagnostic written for exactly
this case and never once seen, because `main()` could not run. TSDF was
unaffected: its splats come from a checkpoint trained through the same
normalization. `Parser` applies this transform to its own `dense_points_path`
and says why in a comment; the CLI just did not.

**The lesson: after mutation-checking a mechanism, mutate the *caller* too.** If
the suite stays green, the wiring is untested no matter how good the unit test
looks. Two concrete instances from closing this one, both caught only by
mutating: a frame test that re-derived the transform inline stayed green when
`extract_mesh.py` stopped applying it, and the mesh-path guard did not cover
the Poisson path — dropping the transform there alone left the whole suite
passing, so it needed its own test.

Two more testing notes from experience here:

- **Assert your own premise, with a measured number.** Several tests would
  otherwise be vacuously satisfied. Where a premise assertion failed, the fix
  was usually that *the prediction* was wrong, not the code — e.g. the disc law
  predicted an evidence-vs-distance ratio of 2.07 where it measures 2.27,
  because the disc law describes a silhouette while the code sums each face's
  *best* view (bounded above by the head-on law at 2.40). The test now asserts
  that bracket, with both ends derived.
- **Do not trust a script's "ok" if you did not see it.** A doc-edit heredoc
  once failed silently because its output went to a backgrounded task file and
  only the pytest tail was read. `grep` for the text afterwards.
- **Check your reader before reporting a bug in the writer.** A 16-bit normal
  map reads back as `uint8` through `imageio` — its Pillow backend cannot write
  16-bit RGB PNG and also silently down-converts one on *read*. The file is
  genuinely `uint16`; OpenCV reads it as such. That looked like a third bug for
  a few minutes and was not one.
- **A perfect synthetic shape can be degenerate.** A dense cloud sampled
  exactly on a sphere makes Qhull raise `QH6239` (cocircular/cospherical) inside
  Poisson's Delaunay step. `make_synthetic_capture.py` adds radial noise by
  default, which is also what a real MVS cloud looks like.

## 5b. Photometric pose refinement does not work here (measured)

Zhou & Koltun colour-map optimisation was implemented in
`gsplat/photogrammetry/photometric_alignment.py` and **does not work well
enough to ship**. It is deliberately not exported and not on any CLI. The
module docstring carries the full trail; the short version:

- It converges to a scene-dependent **attractor of ~5-25 arcminutes regardless
  of the starting error**, including from exactly correct poses. It helps when
  poses are worse than that floor and harms when they are better. No
  regularisation weight both preserves correct poses and recovers a large error.
- **The objective's minimum is in the wrong place**, which is the finding.
  Scored with the fused colour re-estimated at each pose, the converged pose
  reaches **0.038** against **0.061 at the ground truth**. The optimiser is
  correct; the formulation is not.
- The cause is the appearance model: one colour per surface point cannot
  express how views legitimately differ (pixel footprint, obliquity, the mesh's
  own faceting inside a pixel). That difference is already 0.061 at the truth
  and is *reducible* by moving cameras, so the bias exceeds the error corrected.
- open3d ships this as `pipelines.color_map.run_rigid_optimizer` and is worse:
  it degraded pose error in every configuration tried, and **given perfect
  poses moved them to 223'**.

Ruled out by measurement, not argument: the optimiser (Adam random-walks
proportional to its learning rate; L-BFGS with a line search does not, and the
attractor stayed), the image pyramid (present at one level), the fixture's
tessellation (a *finer* mesh is worse), texture periodicity (a non-periodic
field is worse), out-of-frame clamping and a target/residual scale mismatch
(both were real bugs, both fixed, neither was the cause).

**What would fix it:** a per-view appearance term, so a legitimate difference
between views is explained rather than blamed on the pose -- per-view
exposure/gain, and a footprint-aware target (the surface colour convolved with
*that view's* pixel footprint rather than one point sample shared by all). That
is the same modelling gap multi-view super-resolution exists to close, so do
that first and revisit this after.

`tests/test_photometric_alignment.py` pins both claims, so a future fix fails
the tests loudly instead of the finding rotting into folklore.

## 5c. Photometric mesh refinement works, modestly — know its break-even

`refine_mesh_photometric` (Vu et al., TPAMI 2012) **is** exported and on the
CLI as `--refine_mesh`, unlike the alignment above. It is worth understanding
why one shipped and the other did not, because the objectives look similar:

- The alignment compares each view against **one fused colour**, so a view that
  is simply brighter looks like a view whose geometry is wrong.
- The refinement compares **z-normalised patches between views**, which is a
  correlation, so a per-view gain and offset cancel exactly.

Three numbers to know before using it:

- **Break-even is about a third of a source pixel.** Input against output
  surface error, both in source pixels: `0.00 -> 0.15`, `0.24 -> 0.28`,
  `0.48 -> 0.39`, `0.95 -> 0.70`, `1.91 -> 1.50`. The correction scales with
  the error; the cost is a roughly fixed ~0.15 px of added noise. It is not a
  limitation in practice — §"Measured, not guessed" now sizes a TSDF voxel at
  one source pixel, so a TSDF surface starts around a pixel out.
- **It needs about ten views, and the cliff is sharp.** Recovering the same
  0.95 px error: 8 views recovers 4–12%, 10 views 26.7%, 12 views 24.3%.
- **The patch must stay near one source pixel between samples.** A flat tangent
  patch is a chord of a curved surface, so an over-large patch fits best
  slightly *inside* a convex object: the objective's minimum moves from 0.000
  at a 3.6 px patch to −0.010 at 9 px and −0.020 at 18 px. Enlarging the patch
  "for robustness" would silently shrink every convex object.

**The method worked because the objective was validated before the optimiser
was written** — the reverse of §5b, and the cheapest lesson on this branch.

`tests/test_mesh_refinement.py` pins all of it, including a separate capture
*with* per-view exposure variation: without it, removing the z-normalisation —
the one decision that distinguishes this from §5b — passes every other test.

## 6. What to do next

Ordered by value.

**This section previously read "everything unblocked has been done". That was
wrong, and the way it was wrong is the useful part:** it was written from the
set of *planned features*, not from the state of the code, and a live
`TypeError` that crashed every texture-baking run sat under it for five commits
(see section 5). Check the code before repeating the claim — run the CLI, do
not just read it.

**Blocked on a human or hardware** (the real critical path):

1. **Enable GitHub Actions**, then confirm `core_tests.yml` actually runs the
   suite. Highest value by a distance.
2. **Get PR #3 reviewed**, or feedback on scope/direction.
3. **One GPU run on a real capture.** This settles the open question the metrics
   were built to answer: *is view selection worth enabling on real data?*
   `atlas_sharpness` and `seam_discontinuity` exist to answer it.
4. Run either AI-prior recipe against a real model, with weights.

**Unblocked, if you want more code** (all genuinely optional):

5. **View selection combined with multi-page atlases** — currently refused. The
   MRF already labels mesh-wide; the work is making seam levelling run across
   page boundaries and the page bake honour labels. The most substantial
   remaining feature.
6. Non-square atlases; multi-material chart grouping.
7. **Refresh the PR body** (see §2).
8. Tuning the defaults — `--texture_size` 2048, dilation 4 texels, normal-map
   cage 2% of the bbox diagonal. None has been tuned against a real scene, so
   this really wants #3 first.
