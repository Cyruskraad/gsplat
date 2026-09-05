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
- **The PR body is stale.** It was last refreshed at `92554d9` and is 4 commits
  behind: it says "130 tests" (it is 153) and does not mention culling,
  evidence-based atlas sizing, multi-page atlases, or the one-command
  forwarding fix. `docs/photogrammetry_status.md` and this directory *are*
  current. Refreshing means rebuilding from the live body via
  `pull_request_read` — a saved draft in the scratchpad has drifted from it.

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

### Multi-view super-resolution

22. **It is not strictly better than both alternatives, and the pitch said it
    would be.** Against *blending* it is a clean win on both axes at once —
    contrast 63.6% → 90.8% of the ground truth's and L1 0.1081 → 0.0759 —
    which nothing else in the module manages. Against *view selection* it is
    reliably sharper but pointwise a coin flip, and which way it lands depends
    on the regime:

    | Regime | view selection | super-resolved |
    |---|---|---|
    | 128 atlas, 48 px views (σ 0.61 tx) | 0.0785 | **0.0759** |
    | 192 atlas, 48 px views (σ 0.89 tx) | 0.0787 | **0.0784** |
    | 256 atlas, 64 px views (σ 0.89 tx) | **0.0770** | 0.0788 |
    | 384 atlas, 64 px views (σ 1.54 tx) | **0.0400** | 0.0493 |

    The reason is structural: single-view sampling reads the source pixels with
    **no forward model at all**, while this approximates the PSF as a per-view
    Gaussian in atlas space. Every approximation in that model shows up as
    pointwise error, and at wide PSFs it costs more than the deconvolution
    recovers. So it stays opt-in. Do not tighten the test to a strict win on
    synthetic data.
23. **The gradient prior enters the normal equations with a PLUS, and getting
    it wrong looks like success.** `_laplacian` returns the positive
    semi-definite graph Laplacian, so the normal equations of
    `||S T − c||² + λ Tᵀ L T` are `(SᵀWS + λL) T = SᵀWc`. Subtracting it makes
    the operator **indefinite**, CG has no descent direction, and the solve
    diverges — into an atlas measuring **464% of the ground truth's contrast**.
    `atlas_sharpness` alone calls that a triumph. The tells are the L1 (five
    times worse) and non-monotonicity in λ (0.02 → 464%, 0.1 → 168%,
    0.5 → 375%). Any test on this must bound contrast from **above** as well as
    below.
24. **More solver iterations make it worse, not better.** L1 against the truth
    is *U-shaped* in λ and in iteration count: driving the data residual down
    harder over-fits a forward model that is only an approximation. A
    monotonicity assertion on L1 is therefore false; contrast *is* monotonic in
    λ and is what to assert.
25. **Its premise is a resolution one, and small captures do not meet it.** The
    method recovers detail no single view resolves, so it needs the atlas to be
    finer than a source pixel's footprint (σ ≳ 0.5 texels). On
    `tests/test_extract_mesh_cli.py`'s deliberately tiny capture — 6 views of
    64×64 over a scene 4 units across — a source pixel covers *more* surface
    than a texel, there is nothing to recover, the prior dominates and the
    result comes out **blurrier than the blend** (0.059 vs 0.075). That is the
    regime, not a fault; the quality assertions live on a fixture that meets
    the premise.

### Level-set extraction

31. **Marching tetrahedra converges *quadratically*, not linearly.** The
    obvious expectation — halve the cell, halve the error — is wrong: measured
    mean radial error against an analytic sphere goes
    **0.002855 → 0.000685 → 0.000175** as the resolution doubles 16 → 32 → 64,
    a factor of ~4 each time. Linear interpolation along an edge places the
    crossing to second order for a smooth field. This matters for the *test*:
    a first-order bound is satisfied by a second-order method, so asserting
    the wrong law passes while silently tolerating a regression to linear.
32. **Vertices must be identified by the grid edge they lie on.** Emit one per
    tetrahedron instead and the two tetrahedra either side of a face produce
    coincident-but-distinct vertices: the mesh renders identically and has a
    boundary everywhere. Downstream that is not cosmetic — `compute_uvatlas`
    *segfaults* on non-manifold input (trap 9), so it takes out the whole
    texturing stage with exit 139.
33. **open3d's global RNG is shared state, and seeding it breaks other
    tests.** `o3d.utility.random.seed()` (used to make point sampling
    reproducible) also seeds `compute_uvatlas`, which is non-deterministic
    (trap 10). A test asserting "doubling the atlas doubles the PSF sigma"
    passed in isolation and in its own file, then **failed only in a full-suite
    run**, because a different file's seeding changed how tightly the charts
    packed and therefore the texel count. Assert the underlying rule
    (`sigma × texel_world_size` is the source pixel's footprint, invariant to
    the atlas) rather than a ratio that depends on packing.

### Photometric mesh refinement

26. **Visibility must be decided at the surface, then reused for every
    candidate.** `_view_samples` ray-casts each sample against the mesh, so a
    point displaced off the surface is occluded *by the surface it came from*.
    Asking it about candidate offsets directly rejected **all 482 candidates at
    every nonzero offset** on a correct sphere — the search had nothing to
    choose between and the mesh drifted on noise alone. The question is "if the
    surface were here instead, would the cameras agree?", and the cameras that
    see a vertex do not change because it moved a fraction of a vertex spacing.
    The default step (0.5 vertex spacings ≈ 0.077) is already **twice** the
    ray-cast tolerance (~0.036 at 3.5 units), so this is not an edge case.
27. **A Laplacian regulariser collapses a correct surface, and slowly enough to
    look like convergence.** The average of a vertex's neighbours lies inside
    any convex surface. Measured: ten rounds at strength 0.3 take a unit
    sphere's mean radius to **0.9444**. Projecting the smoothing onto the
    vertex's tangent plane holds **1.00017** — it redistributes vertices *over*
    the surface without moving the surface, and the photometric term already
    owns the normal direction, so the two never fight.
28. **`np.argmin` breaks ties toward the first offset, which is the most
    inward one.** With offsets ordered inward-to-outward, every tie and every
    all-unmeasurable vertex resolves inward. Staying put has to be the default:
    a move must strictly improve on the current position's cost.
29. **Coarse-to-fine hurts here too**, exactly as for camera refinement
    (trap 19). Recovering a sphere perturbed by 0.03: single-scale improves the
    radial error **1.95x** where three pyramid levels manage **1.21x**; and on
    an *already correct* sphere three levels drift the mean radius to **0.983**
    where single-scale holds **0.9986**. Halving the image blurs away the very
    detail the photoconsistency is measured from, so the coarse levels optimise
    noise. `num_levels` defaults to 1.
30. **The method has a noise floor, and it is worth stating.** Sampling a
    texture through finitely many finite-resolution views leaves ~**0.0068** of
    radial error on a mesh that started at exactly zero, against the
    **0.0125** it gets a 0.0244 error down to. So it cannot improve a surface
    already better than its floor, and will add up to that much noise.

### Photometric camera refinement

17. **open3d already ships this algorithm and it does not work here.**
    `open3d.pipelines.color_map.run_rigid_optimizer` made the poses **worse in
    every configuration tested** — including on *exact* poses, which it walked
    72' away from. That is not a tuning problem on the caller's side. The cause
    is structural: it optimises **per-vertex colours** as its surface proxy, so
    the objective can only see detail the mesh's vertex density carries, and a
    760-vertex sphere cannot represent the texture whose misregistration is
    being measured. (This package bakes into a UV *atlas* for exactly that
    reason.) Reuse was the right thing to try — it is an existing dependency
    and "external tools stay external" — but the measurement decided it.
18. **A rotation delta must carry the translation with it.** Parameterising a
    pose as `R = exp(δ)·R₀`, `t = t₀ + Δt` — which is what
    `bundle_adjustment._optimize` does — moves the camera *centre* when δ
    changes, because the centre is `−Rᵀt`. Reprojection BA gets away with it
    because its translation is always free to compensate. A photometric solve
    does not: at |t₀| ≈ 3.5 a 45' rotation swings the centre by 0.046 world
    units, a third of the texture's wavelength. With `t = exp(δ)·t₀ + Δt` the
    solve improves the registration 62' → 20'; without it, it makes it *worse*,
    62' → 155'.
19. **The pyramid is not what makes 45' recoverable — the re-baked target is.**
    The received claim ("a photometric objective has a tiny basin of
    convergence; single-scale will not recover 45'") is false on this fixture,
    and the numbers say why: the detail's wavelength is 5.2 px in the image and
    45' displaces a projection by 1.70 px, well inside the 2.6 px
    half-wavelength where the objective is unambiguous. **At equal work** — 9
    optimisation rounds either way — single-scale reaches 23.6' and the 3-level
    pyramid 24.75'. The pyramid earns its place further out, at 90'
    (3.39 px, past the half-wavelength) where single-scale lands at 183' and
    the pyramid at 133'. Neither *recovers* there, so the method's working
    range is a displacement below half the detail's wavelength; the pyramid
    widens the margin rather than extending the range.
20. **Contrast retention is not a fidelity metric, in either direction.**
    Trap 1 records that a sharp-but-displaced atlas scores worse on L1. The
    converse also holds: an atlas can retain *more* gradient energy than a
    perfectly-registered one, because residual misregistration **adds**
    high-frequency energy. Measured against the analytic sphere, refined poses
    gave 92.2% retention where exact poses gave 79.2% — the refined atlas is
    not better, it is noisier. Always read retention against the
    perfectly-registered ceiling, never against 100%.
21. **Rendering views from the analytic sphere while refining against the mesh
    is a confound, not a fixture.** `_SphereDataset` ray-traces the analytic
    sphere; `_unit_sphere_mesh` is a polyhedron inscribed in it, and at
    resolution 10 the sagitta is ~9% of the high-frequency pattern's
    wavelength. The photometric optimum is then genuinely not the true pose, so
    refinement spends its freedom compensating for the geometry: it moved
    *correct* cameras by 15' and made the bake 3.9% worse. That reads as "the
    method harms good poses" when it is really "the method was handed the wrong
    surface". Rendering the views from the mesh itself (as
    `examples/make_synthetic_capture.py` does) removes it, and the same test
    then shows no harm at all.

## 5. The recurring testing failure — read this before writing tests

**Nine times on this branch, a test proved a *mechanism* worked while the
*call site* went unpinned.** Three times in this session's own work, *after* writing
the test that was meant to pre-empt it — both caught only by mutating the
caller. Each time the mutation passed the entire suite:

| Mutation that escaped | Why the existing test missed it | What closed it |
|---|---|---|
| Cull on `quality > 0` instead of visibility | The nested-spheres scene is textured, so the two agree there | A flat-scene culling test |
| Sum projected areas instead of max | Every scaling test uses *ratios*, and a consistent over-count cancels out of a ratio | A test pinning absolute evidence against view count |
| `bake_texture_atlas_pages(occluder=None)` | The occlusion test supplies its own occluder either way | A test driving the page bake itself |
| Return an unmeasured decimation result | The first attempt produced the same mesh by coincidence | A stronger mutation (return an over-decimated mesh) |
| `bake_mesh_texture` never accepted the `seam_smoothness` its CLI passed | No test had ever called `extract_mesh.main()` — the `assert cfg.ckpt` before the method dispatch made a GPU checkpoint the price of reaching it | `tests/test_extract_mesh_cli.py`, driving `main()` |
| Force every pyramid level to full resolution | The pyramid test compared `num_levels=1` against `num_levels=3` at the same `alternations`, i.e. 3 optimisation rounds against 9. The extra *rounds* were doing the work the pyramid got credit for | An **equal-work** comparison (3 levels × 3 rounds vs 1 level × 9), in the regime where the objective actually aliases |
| Hardcode Poisson's normal radius back to `0.1` | The test asserted the *derived* value in `stats_out`, which was written from the derivation rather than from the value actually passed to open3d. And the reconstruction itself is a weak detector: `orient_normals_consistent_tangent_plane` plus a modest octree depth still produce a plausible sphere from normals aligned only 0.507 with the truth | Report the stat from the value about to be **used**, and measure the normals against the analytic sphere directly |
| Re-test visibility at each candidate offset (mesh refinement) | The recovery test's threshold (1.5x) was looser than the gap the mutation opens (1.95x -> 1.55x) | Tighten the threshold to the measured margin, and pin the trap itself: displaced points *are* invisible to `_view_samples` |
| Compute the refined mesh, then return the original | (pre-empted) The CLI test compares the written vertices against an unrefined run rather than reading the stats | — |
| Compute the refined poses, then discard them | The CLI test asserted the alignment *stats* reached `mesh_metrics.json`. Stats are an output of the solve, not evidence it was used | Comparing the delivered per-vertex colours, aligned vs not, against the analytic truth |

**The lesson: after mutation-checking a mechanism, mutate the *caller* too.** If
the suite stays green, the wiring is untested no matter how good the unit test
looks.

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

## 6. What to do next

Ordered by value. Everything unblocked has been done; be honest about that
rather than inventing work.

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
