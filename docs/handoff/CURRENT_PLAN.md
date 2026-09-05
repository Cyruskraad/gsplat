# CURRENT PLAN — the live one, as of `55450b7`

**This is the authoritative, up-to-date plan for the photogrammetry work.** A
new session can start from this file alone: it says what to read, where things
stand, what is worth doing next, and how to verify it.

Do not confuse it with [`../photogrammetry_texturing_plan.md`](../photogrammetry_texturing_plan.md),
which is **complete and closed** — that one is kept as the record of what was
measured during the texturing work, not as a to-do list.

*Head of PR #3: `55450b7`. 196 tests passing. Last verified against the code on
the date of that commit — if the branch has moved since, trust the code and
`git log`, then fix this file.*

---

You are continuing work on the `gsplat.photogrammetry` subpackage in this
repository (`Cyruskraad/gsplat`, open draft PR #3).

## Read these first, and only these

`docs/handoff/` is six short documents written to be the **only** orientation
you need. They are current.

1. [`README.md`](README.md) — state, and the five-minute pick-up.
2. [`SCOPE.md`](SCOPE.md) — the **explicit non-goals**. Three bind everything:
   gsplat never runs a neural network itself (priors arrive as precomputed
   files); external tools stay external; and **no new required dependencies** —
   `gsplat[mesh]` is deliberately just `open3d` + `imageio`, which is why the
   MRF optimiser and the linear solver are hand-rolled NumPy. scipy may be an
   optional accelerator, never a hard import.
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — modules, the `_view_samples` hub, the
   atlas path, the orchestration contract.
4. [`ISSUES.md`](ISSUES.md) — **read this carefully.** Many entries are
   measurements that invert the obvious guess. §5 is the recurring testing
   failure; §5b–§5d are three methods judged on measurement, one of which is
   deliberately not shipped.
5. [`PROGRESS.md`](PROGRESS.md) — what was *executed* versus only *reviewed*,
   and the fifteen bugs.
6. This file.

[`../photogrammetry.md`](../photogrammetry.md) is the user-facing guide
(current). [`../photogrammetry_status.md`](../photogrammetry_status.md) is the
long-form history — useful for why a particular decision was made, not for
where things stand.

## Branch

Work on **`claude/photogrammetry-techniques-plan-jb0pod`**, the head of PR #3.
`git checkout claude/photogrammetry-techniques-plan-jb0pod`. If your session
instructions name a different branch, branch it **from `jb0pod`, not from
`main`** — `main` has no `gsplat.photogrammetry` at all — and say so.

## Baseline, before changing anything

```bash
python -m pytest tests/test_bundle_adjustment.py tests/test_mesh_extraction.py \
    tests/test_neural_sfm.py tests/test_colmap_dataset.py \
    tests/test_photogrammetry_metrics.py tests/test_photogrammetry_pipeline.py \
    tests/test_texturing.py tests/test_extract_mesh_io.py \
    tests/test_extract_mesh_cli.py tests/test_photometric_alignment.py \
    tests/test_mesh_refinement.py tests/test_level_set.py -q
```

Expect **196 passed** (~2 min). Also confirm the machine: `nvidia-smi`,
`command -v colmap`. It has been CPU-only with no colmap in every session so
far, and that decides what any GPU-facing work can claim.

## Ground rules — each exists because ignoring it caused a real defect here

- **There is no CI.** Actions is disabled at the repo level. Validate by hand:
  ```bash
  python -m black --check --required-version 22.3.0 <files>
  python -m py_compile <files>
  cd examples && PYTHONPATH=<repo> python -c "import extract_mesh, run_pipeline, make_synthetic_capture"
  ```
  The last is not optional: `py_compile` does not evaluate a `tyro` dataclass's
  annotations, so a missing import there compiles cleanly and breaks on import.
- **Mutation-check every guard.** Revert the fix, confirm a test genuinely
  fails. A test that passes with the fix reverted is not a test.
- **Pin call sites, not just mechanisms.** This has now failed **six** times
  (`ISSUES.md` §5). After testing a function, ask separately: *what test fails
  if the caller stops calling it, or calls it wrongly?*
- **Validate an objective before building an optimiser on it.** One task built
  an optimiser on an objective whose minimum was in the wrong place and had to
  be thrown away; the mesh refinement that followed probed the objective first
  and worked. This is the cheapest lesson on the branch.
- **Assert each test's premise with a measured number**, and do not shrink a
  fixture past the point where the method under test still works — that is
  tuning the test to pass, not testing.
- **Be explicit about executed versus reviewed**, in commits and in the PR.
- **Run the CLI, don't just read it.** Both of the two most recent bugs were
  invisible to inspection and obvious on the first real run.

## Where things stand

The previous plan's four tasks are all done. Three results are worth knowing
before planning anything:

- **`refine_mesh_photometric` shipped** (Vu et al., TPAMI 2012) — vertices
  slide along their normals onto the photoconsistent surface. Break-even is
  about a third of a source pixel and it wants ~10 views; both measured, both
  in the module docstring.
- **`photometric_alignment` did not ship.** Zhou & Koltun colour-map
  optimisation converges to an attractor regardless of the starting error, and
  the objective's minimum is measurably in the wrong place (0.038 at the
  converged pose against 0.061 at ground truth). The cause is the appearance
  model: one colour per surface point cannot express how views legitimately
  differ. It is retained, unexported, with two tests pinning the diagnosis.
- **`level_set.py`** (GOF-style) is exported and CLI-wired but has **never run
  on a GPU or a real checkpoint.** The extractor is measured against analytic
  fields and the field adapter's arithmetic against closed form, both on CPU.
  Do not report it as working end to end until someone has run it.

## What is worth doing next

Pick one and do it properly; there is no obligation to take them in order.

1. **A per-view appearance model.** Per-view exposure/gain, and a
   footprint-aware target — the surface colour convolved with *that view's*
   pixel footprint rather than one point sample shared by every view. This is
   precisely the gap that sank photometric camera alignment (`ISSUES.md` §5b),
   and the same gap multi-view super-resolution (Goldlücke et al., IJCV 2014)
   exists to close. Closing it would plausibly revive both at once, which makes
   it the highest-value unblocked item.
2. **View selection combined with multi-page atlases** — currently refused by
   design. The MRF already labels mesh-wide; the work is making seam levelling
   run across page boundaries and the page bake honour labels. The most
   substantial remaining *feature*.
3. **An optimality bound for the view-selection MRF.** ICM has none, and
   alpha-expansion needs a max-flow solver `gsplat[mesh]` deliberately lacks.
   TRW-S (Kolmogorov, TPAMI 2006) is pure NumPy and gives a lower bound, so the
   pipeline could report an optimality gap instead of hoping.

**Blocked on a human, and worth more than any of the above:** enabling GitHub
Actions, a review of PR #3, and one GPU run on a real capture. Check whether
any of those has changed before assuming it has not.

## Verification, for whatever you do

1. The twelve suites above — currently **196 passed**; must stay green and grow.
2. `black --check --required-version 22.3.0`, `py_compile`, and the example
   imports on every touched file.
3. **An end-to-end CLI run**, which is the call-site pin:
   ```bash
   python examples/make_synthetic_capture.py --out_dir /tmp/capture
   python examples/extract_mesh.py --method mesh --mesh_path /tmp/capture/mesh_gt.ply \
       --data_dir /tmp/capture --data_factor 1 --test_every 10000 \
       --result_dir /tmp/out --texture_mode atlas --device cpu
   ```
4. Mutation-check each new guard; list the mutations in the commit message, as
   the existing commits do. Where you deliberately do *not* mutation-check
   something, say so next to the code and say why.
5. **Keep this file, `docs/handoff/` and the PR body current as you go** — this
   file is only worth having if the next session can trust it.

Report at the end what landed, what each thing *measured*, and what remains
blocked on a GPU, a real capture, or a human.
