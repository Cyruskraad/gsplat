"""Project-level defaults."""

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ARAGO_RUNS_ROOT = Path("/home/dhlab/Documents/arago-3d-reconstruction/runs/colmap-scripts")
RELIGHT_RUNS_ROOT = Path("/home/dhlab/Documents/arago-3d-reconstruction/runs/uv-mgs")

REQUIRED_IMAGE_COUNT = 32
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
GIT_REMOTE_REPO = "https://github.com/raphaelsulzer/colmap-scripts.git"
DEFAULT_CONFIG_FILENAME = "colmap-scripts.toml"

ARAGO_PREFIX = "RIG_"
ARAGO_START_INDEX = 15177

DEFAULT_RECONSTRUCTION_CONFIG = {
    "expected_images": REQUIRED_IMAGE_COUNT,
    "arago_names": True,
    "arago_start_index": ARAGO_START_INDEX,
    "gpu_index": 0,
    "colmap": {
        "feature_extractor": {
            "max_num_features": 50000,
            "max_num_orientations": 4,
            "use_gpu": True,
        },
        "matcher": {
            "guided_matching": True,
            "use_gpu": True,
        },
        "mapper": {
            "threads": 12,
        },
        "patch_match": {
            "max_image_size": 4000,
            "geom_consistency": True,
        },
        "fusion": {
            "min_num_pixels": 3,
            "max_reproj_error": 1.5,
        },
        "bundle_adjustment": {
            "refine_focal_length": True,
            "refine_extra_params": True,
            "refine_points3d": True,
        },
        "point_triangulator": {
            "min_num_matches": 15,
            "init_num_trials": 200,
            "init_min_num_inliers": 100,
            "init_max_error": 4.0,
            "init_min_tri_angle": 16.0,
            "init_max_forward_motion": 0.95,
            "init_max_reg_trials": 2,
        },
    },
    "thresholds": {
        "registered_images": 32,
        "points": 5000,
        "mean_track_length": 2.5,
        "observations_per_image": 150.0,
        "max_reprojection_error": 1.5,
        "dense_points": 250000,
    },
    "meshing": {
        "poisson": True,
        "delaunay": False,
        "delaunay_num_threads": 1,
        "delaunay_timeout_seconds": 900,
    },
}
