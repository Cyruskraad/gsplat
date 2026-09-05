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
   :members: extract_mesh_tsdf, extract_mesh_poisson, cull_unobserved_faces,
             simplify_mesh, simplify_mesh_to_error

Texturing
------------------

.. automodule:: gsplat.photogrammetry.texturing
   :members: bake_texture, bake_texture_atlas,
             bake_texture_atlas_view_selected, bake_texture_atlas_pages,
             partition_faces, bake_mesh_texture,
             face_view_quality, face_visibility, face_projected_areas,
             recommended_texture_size, select_views_mrf, level_seams,
             bake_normal_map, bake_ambient_occlusion

Neural SfM Import
------------------

.. automodule:: gsplat.photogrammetry.neural_sfm
   :members: merge_point_maps_to_tracks, write_colmap_reconstruction

Automatic Metrics
------------------

.. automodule:: gsplat.photogrammetry.metrics
   :members: point_to_mesh_distance, mesh_quality_stats, point_cloud_stats,
             reconstruction_stats, track_stats, mask_coverage_stats,
             depth_prior_stats, atlas_sharpness, seam_discontinuity

Pipeline Orchestration
-----------------------

.. automodule:: gsplat.photogrammetry.pipeline
   :members: PipelineReport, StageResult, run_stage, record_skipped,
             collect_artifact_metrics, latest_metrics, check_prior_quality,
             derive_cross_stage_metrics, cross_stage_metrics_from_artifacts,
             format_cross_stage_metrics
