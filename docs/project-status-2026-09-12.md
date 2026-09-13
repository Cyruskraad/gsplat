# Fixed-light Gaussian reconstruction: work and repository status

**Status date:** 2026-09-12  
**GitHub repository:** `Cyruskraad/gsplat`  
**Branch:** `codex/gsplat-hq-fixed-light`

## Work completed in this project chat

1. The earlier Gaussian-splatting work was audited and the failed 30,000-step
   RedObject result was rejected as a quality target because validation quality
   declined while the Gaussian population and floaters grew.
2. The gsplat COLMAP loader and trainer were upgraded for fixed-light,
   object-masked reconstruction, deterministic validation/test splits,
   mask-aware evaluation, bounded densification, resumable checkpoints, early
   stopping, best-checkpoint selection, and immutable run provenance.
3. RedObject strategy regressions and four AngelDemon factor-4 pilots were run.
   Capped MCMC was selected as the stable route to HQ and compact models.
4. Frozen factor-2 HQ and compact models were trained and evaluated. The sealed
   test set was inspected only after the configuration was frozen.
5. A native-resolution HQ refinement and the prescribed stronger silhouette
   correction were each tested once and rejected on measured quality regression.
6. Production HQ and compact models were retrained on all 115 views. Checkpoints,
   Gaussian PLY files, logs, telemetry, configuration, and orbit videos were
   retained without overwriting earlier runs.
7. A separate HQ delivery was produced on the workstation from the preserved
   600,000-Gaussian checkpoint: twelve 4096 x 4096 RGB renders, twelve alpha
   mattes, twelve RGBA renders, a 2560 x 2560 H.264 orbit, a contact sheet, the
   viewer-loadable PLY, checksums, provenance, and an offline HTML orbit viewer.
8. The delivery was transferred to the local laptop and verified by full PNG
   decoding, full 180-frame video decoding, PLY-header inspection, remote/local
   SHA-256 comparison, and package-wide checksum validation.

## Reconstruction status

| Area | Current state |
|---|---|
| Dataset | 115 images and masks; 147,613 COLMAP points; 124,830 mask-filtered initialization points |
| Scientific split | 92 train / 12 validation / 11 sealed test |
| HQ selection | Capped MCMC, 600,000 Gaussians, step 15,999 |
| Compact selection | Capped MCMC, 300,000 Gaussians, step 14,999 |
| Production retraining | Completed on all 115 views for both variants |
| Viewer/export | Both PLY files load in the gsplat viewer; orbit videos decode completely |
| Strict silhouette gate | Not met; reported exception retained, not redefined as a pass |
| Native HQ refinement | Rejected after regression |
| Pose optimization | Not run; no repeatable camera-specific double-edge evidence |

The laptop delivery is stored outside this source repository at:

```text
/Users/kazemi0001/Documents/ChatGPT/SfM reconstruction_COLMAP_remote linux/
  deliverables/angeldemon_gaussian_splat_hq_20260901T133103Z/
```

The authoritative production run remains on the Linux workstation at:

```text
/home/dhlab/results/gaussian-splatting__angel-demon-200726__20260819-0908/
  production/angeldemon-hq-all115-20260819T140305Z/
```

Large datasets, checkpoints, PLYs, renders, and videos are intentionally not
committed to GitHub. The repository contains source code and documentation;
the run directories contain immutable evidence and binary artifacts.

## Repository status

The fixed-light implementation is a linear commit series on
`codex/gsplat-hq-fixed-light`. The production jobs used code at `00ea6653`; the
branch later added sealed-test CSV export and neutralized machine-specific helper
paths. The last code-only commit before this status document is `1af2201e`.

Fresh verification performed on the workstation on 2026-09-12:

- Python compilation succeeded for every changed trainer, loader, and pipeline
  helper module.
- `python -m pytest -q tests/test_strategy.py` completed with **3 passed**.
- The working tree was clean before documentation was added.

The final synchronization check on 2026-09-12 confirmed all of the following:

```text
local HEAD == origin/codex/gsplat-hq-fixed-light
remote workstation HEAD == origin/codex/gsplat-hq-fixed-light
git status --porcelain == empty in both checkouts
```

The authenticated laptop clone, GitHub tracking branch, and Linux workstation
checkout all resolved to the same branch tip with zero commits ahead or behind.

## Separate mug reconstruction work

The mug/cup request is a separate reconstruction effort in the
`raphaelsulzer/colmap-scripts` orchestration repository. Its software/evaluation
foundation and CUDA smoke test exist locally, but no new controlled real-mug
model has passed the semantic-mask, frozen-split, multi-seed, and cavity-quality
release gates. The older 498,643-Gaussian delivery therefore remains an audited
baseline rather than a newly promoted final model. Its source changes and
generated artifacts must not be conflated with this AngelDemon gsplat branch.
