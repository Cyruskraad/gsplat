gsplat.photogrammetry
===================================

`gsplat.photogrammetry` closes the SfM -> mesh loop on top of the COLMAP data
loader and 2DGS/3DGS renderers -- see :doc:`../examples/photogrammetry` for a
walkthrough and the root-level ``docs/photogrammetry.md`` guide for full CLI
usage.

``mesh_extraction`` requires the optional ``open3d`` dependency
(``pip install gsplat[mesh]``); ``dense_mvs`` requires a CUDA-enabled
``colmap`` command-line install.

.. automodule:: gsplat.photogrammetry
   :members:

Bundle Adjustment
------------------

.. automodule:: gsplat.photogrammetry.bundle_adjustment
   :members: refine_reconstruction

Dense MVS
------------------

.. automodule:: gsplat.photogrammetry.dense_mvs
   :members: run_dense_mvs

Mesh Extraction
------------------

.. automodule:: gsplat.photogrammetry.mesh_extraction
   :members: extract_mesh_tsdf, extract_mesh_poisson, bake_texture
