Photogrammetry Pipeline
========================================

.. currentmodule:: gsplat

Starting from a standard :doc:`COLMAP capture <colmap>` (``<data_dir>/images/``
+ ``<data_dir>/sparse/0/``), the scripts under ``examples/`` chain together
:mod:`gsplat.photogrammetry`'s bundle adjustment, dense MVS, and mesh
extraction to go from SfM data all the way to a textured mesh:

.. code-block:: bash

    # 1. Bundle adjustment: refine poses + points, writing sparse/refined.
    python examples/bundle_adjust.py --data_dir data/360_v2/garden

    # 2. Dense MVS: densify the (refined) sparse point cloud. Requires a
    #    CUDA-enabled `colmap` CLI install.
    python examples/dense_mvs.py --data_dir data/360_v2/garden \
        --colmap_dir data/360_v2/garden/sparse/refined

    # 3. Train as usual (see :doc:`colmap`).
    python examples/simple_trainer_2dgs.py \
        --data_dir data/360_v2/garden --data_factor 4 \
        --result_dir results/garden_2dgs

    # 4. Mesh extraction: TSDF fusion of the trained 2DGS scene's rendered
    #    depth maps, with texture baking from the training images. Requires
    #    the optional `open3d` dependency (`pip install gsplat[mesh]`).
    python examples/extract_mesh.py \
        --ckpt results/garden_2dgs/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/360_v2/garden --result_dir results/garden_2dgs

Each stage above writes automatic quality metrics (:mod:`gsplat.photogrammetry.metrics`)
to a ``stats/*.json`` file next to its output -- run
``python examples/summarize_photogrammetry_stats.py --result_dir results/garden_2dgs
--data_dir data/360_v2/garden`` to aggregate them into one report.

See ``docs/photogrammetry.md`` (repo root) for the full guide, including how
to wire the refined poses / dense point cloud into the trainer's
``Parser(..., colmap_dir=..., dense_points_path=...)`` arguments, and
:doc:`../apis/photogrammetry` for the Python API.
