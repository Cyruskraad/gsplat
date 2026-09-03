Photogrammetry Pipeline
========================================

.. currentmodule:: gsplat

Starting from a standard :doc:`COLMAP capture <colmap>` (``<data_dir>/images/``
+ ``<data_dir>/sparse/0/``), :mod:`gsplat.photogrammetry` goes from SfM data
all the way to a textured mesh with per-stage quality metrics throughout.
The whole thing in one command, via :mod:`gsplat.photogrammetry.pipeline`:

.. code-block:: bash

    python examples/run_pipeline.py \
        --data_dir data/360_v2/garden --result_dir results/garden_pipeline

which writes ``results/garden_pipeline/pipeline_report.json`` -- every
stage's status, timing, and metrics in one place. ``--stages`` selects a
subset and ``--dry_run`` previews the commands without running them; a
stage needing something the machine lacks (a CUDA ``colmap`` build, a GPU)
is recorded as ``skipped`` rather than failing the run.

Equivalently, run the same stages by hand -- useful for iterating on one
stage at a time:

.. code-block:: bash

    # 1. Bundle adjustment: refine poses + points, writing sparse/refined.
    python examples/bundle_adjust.py --data_dir data/360_v2/garden

    # 2. Dense MVS: densify the (refined) sparse point cloud. Requires a
    #    CUDA-enabled `colmap` CLI install.
    python examples/dense_mvs.py --data_dir data/360_v2/garden \
        --colmap_dir data/360_v2/garden/sparse/refined

    # 3. Train, pointing the trainer at the refined poses / dense init.
    python examples/simple_trainer_2dgs.py \
        --data_dir data/360_v2/garden --data_factor 4 \
        --result_dir results/garden_2dgs \
        --colmap_dir data/360_v2/garden/sparse/refined \
        --dense_points_path data/360_v2/garden/dense/dense.ply

    # 4. Mesh extraction: TSDF fusion of the trained 2DGS scene's rendered
    #    depth maps, with texture baking from the training images. Requires
    #    the optional `open3d` dependency (`pip install gsplat[mesh]`).
    python examples/extract_mesh.py \
        --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/360_v2/garden --result_dir results/garden_2dgs

Each stage above writes automatic quality metrics (:mod:`gsplat.photogrammetry.metrics`)
to a ``stats/*.json`` file next to its output -- run
``python examples/summarize_photogrammetry_stats.py --result_dir results/garden_2dgs
--data_dir data/360_v2/garden`` to aggregate a manually-run sequence into one
report the same way ``run_pipeline.py`` does automatically.

Pass ``--mono_depth_dir``/``--mask_dir`` (``run_pipeline.py``, or step 3/4
above) to add monocular depth-prior supervision / exclude
externally-segmented transient content (people, vehicles, ...) from training
and mesh fusion -- see "Monocular depth-prior supervision" and
"Transient/dynamic-object masking" in ``docs/photogrammetry.md``.

Step 4 bakes per-vertex colors by default, whose resolution is limited by the
mesh's vertex density. Pass ``--texture_mode atlas`` (``--texture_size`` sets
the resolution) to UV-unwrap the mesh and bake a real texture atlas instead,
writing ``mesh.obj`` + ``mesh.mtl`` + ``mesh_0.png`` -- a mesh that loads with
its texture attached in standard DCC tools and game engines. See "Texture:
per-vertex colors or a UV atlas" in ``docs/photogrammetry.md``.

When ``--mask_dir``/``--mono_depth_dir`` are given, ``run_pipeline.py``'s
``priors`` stage also *gates* them: it flags an empty prior directory, masks
that exclude (almost) the whole frame or nothing at all, and depth maps that
are constant or mostly non-finite -- before the training stage spends hours
on them. It warns by default and fails the run under ``--strict``. See "The
``priors`` quality gate" in ``docs/photogrammetry.md``.

For a deliverable asset rather than a raw extraction, add
``--target_triangles`` and ``--normal_map`` to step 4: the mesh is decimated
to a triangle budget (quadric error metrics) and the removed detail is baked
into a normal map on its UV atlas, so the light mesh still shades like the
dense one. See "Decimation + normal maps (the delivery path)" in
``docs/photogrammetry.md``.

See ``docs/photogrammetry.md`` (repo root) for the full guide and
:doc:`../apis/photogrammetry` for the Python API.
