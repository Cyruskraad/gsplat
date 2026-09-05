"""Write a small, multi-view-consistent synthetic capture to disk.

The photogrammetry delivery path (`examples/extract_mesh.py`) is almost
entirely GPU-free -- only TSDF's depth rendering needs a GPU -- but until
recently nothing could *run* it without a trained checkpoint, so its CLI guards
went untested and a `TypeError` in its texture call survived five commits
(docs/handoff/ISSUES.md section 5).

The repo had two halves of the fixture needed to fix that, and neither was
enough on its own:

- `tests/test_colmap_dataset.py` writes a real COLMAP model, but pairs it with
  `rng.uniform(0, 255)` noise images, so nothing on disk is multi-view
  consistent and no texturing behaviour can be measured against it.
- `tests/test_mesh_extraction.py`'s `_SphereDataset` renders genuinely
  consistent views, but only in memory -- it never touches disk, so it cannot
  drive a CLI.

This joins them: an analytic textured surface, ray-cast into N cameras, written
as `images/*.png` plus a real `sparse/0/` COLMAP model (via
`gsplat.photogrammetry.write_colmap_reconstruction`), optionally with a dense
`.ply` and the ground-truth mesh. Because the surface colour is a closed-form
function of the 3D point, every view agrees about every point, and a correct
bake must reproduce it -- which is what makes the output usable as ground truth
and not merely as something that parses.

    python examples/make_synthetic_capture.py --out_dir /tmp/capture
    python examples/extract_mesh.py --method mesh \\
        --mesh_path /tmp/capture/mesh_gt.ply --data_dir /tmp/capture \\
        --data_factor 1 --test_every 10000 --result_dir /tmp/out \\
        --texture_mode atlas --device cpu

Two knobs reproduce the failure modes the texturing work is judged against,
both carried over from `_SphereDataset`:

- `--pose_error_arcmin` perturbs the *reported* pose while leaving the render
  untouched, which is exactly what residual SfM error looks like.
- `--exposure` gives each view a constant offset applied only where the surface
  is hit, so it cannot be recovered from the background -- what seam levelling
  exists to remove.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import tyro

from gsplat.photogrammetry import write_colmap_reconstruction


def surface_color(points: np.ndarray, frequency: float = 6.0) -> np.ndarray:
    """Closed-form RGB in [0, 1] for surface points, `(N, 3) -> (N, 3)`.

    A function of the 3D point alone, so every camera that sees a point agrees
    on its colour and a correct bake reproduces this exactly. The default
    frequency puts detail near the misregistration scale, which is the regime
    where blending measurably loses contrast -- at low frequency it does not,
    and a texturing test built on a smooth pattern proves nothing (see
    docs/photogrammetry_texturing_plan.md, "Premise, measured").
    """
    points = np.asarray(points, dtype=np.float64)
    phase = frequency * points
    rgb = 0.5 + 0.5 * np.stack(
        [
            np.sin(phase[:, 0]) * np.cos(phase[:, 1]),
            np.sin(phase[:, 1]) * np.cos(phase[:, 2]),
            np.sin(phase[:, 2]) * np.cos(phase[:, 0]),
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def fibonacci_directions(count: int) -> np.ndarray:
    """`count` near-uniformly spread unit vectors, `(count, 3)`."""
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = golden * indices
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=-1)


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """A camera-to-world 4x4 looking from `position` at `target`.

    Uses the COLMAP/OpenCV convention (x right, y *down*, z forward), matching
    what the bakers and `extract_mesh_tsdf` expect -- they invert this
    themselves to get world-to-camera.
    """
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    camtoworld = np.eye(4)
    camtoworld[:3, :3] = np.stack([right, up, forward], axis=1)
    camtoworld[:3, 3] = position
    return camtoworld


def _rotate_about_center(camtoworld: np.ndarray, axis: np.ndarray, angle: float):
    """Rotate a pose about its own optical centre (Rodrigues).

    Perturbing the *reported* pose only -- the render is unchanged -- is what
    makes this residual SfM error rather than a different capture.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    out = camtoworld.copy()
    out[:3, :3] = R @ camtoworld[:3, :3]
    return out


@dataclass
class Config:
    # Directory to write the capture into (created if absent).
    out_dir: str = "data/synthetic_capture"
    # Number of camera views, spread over a Fibonacci sphere.
    num_views: int = 16
    # Rendered image size, in pixels.
    width: int = 128
    height: int = 128
    # Pinhole focal length, in pixels.
    focal: float = 170.0
    # Sphere radius and camera distance, in scene units.
    radius: float = 1.0
    cam_dist: float = 3.5
    # Tessellation of the ground-truth sphere written to mesh_gt.ply.
    mesh_resolution: int = 20
    # Spatial frequency of the surface pattern. Higher puts detail nearer the
    # misregistration scale, where blending measurably loses contrast.
    frequency: float = 6.0
    # Number of sparse SfM points (with real cross-view tracks).
    num_points: int = 400
    # Also write dense/dense.ply, for --method poisson.
    write_dense: bool = True
    # Points in the dense cloud.
    num_dense_points: int = 20000
    # Radial noise on the dense cloud, as a fraction of the radius. Real MVS
    # clouds are noisy, and a *perfectly* cospherical cloud is degenerate for
    # the Delaunay step inside Poisson reconstruction -- Qhull raises QH6239
    # rather than returning a surface. It also makes the cloud-to-mesh fit
    # metric mean something instead of being trivially zero.
    dense_noise: float = 0.004
    # Simulated residual SfM pose error, in arcminutes. Perturbs the reported
    # pose only; the rendered image is unchanged.
    pose_error_arcmin: float = 0.0
    # Per-view constant exposure offset, drawn from [-exposure, exposure] and
    # applied only where the surface is hit.
    exposure: float = 0.0
    seed: int = 0


def build(cfg: Config) -> dict:
    """Write the capture and return the paths and arrays it produced."""
    rng = np.random.default_rng(cfg.seed)

    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=cfg.radius, resolution=cfg.mesh_resolution
    )
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    K = np.array(
        [
            [cfg.focal, 0.0, cfg.width / 2.0],
            [0.0, cfg.focal, cfg.height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    positions = fibonacci_directions(cfg.num_views) * cfg.cam_dist
    target = np.zeros(3)
    true_poses = np.stack([look_at(p, target) for p in positions], axis=0)

    images_dir = os.path.join(cfg.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Pixel ray directions in camera space, shared by every view.
    xs, ys = np.meshgrid(np.arange(cfg.width), np.arange(cfg.height))
    pixels = np.stack(
        [
            (xs.ravel() + 0.5 - K[0, 2]) / K[0, 0],
            (ys.ravel() + 0.5 - K[1, 2]) / K[1, 1],
            np.ones(cfg.width * cfg.height),
        ],
        axis=-1,
    )

    image_names = []
    for i in range(cfg.num_views):
        camtoworld = true_poses[i]
        directions = pixels @ camtoworld[:3, :3].T
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origins = np.broadcast_to(camtoworld[:3, 3], directions.shape)
        rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = hits["t_hit"].numpy()
        hit = np.isfinite(t_hit)

        image = np.zeros((cfg.width * cfg.height, 3))
        surface = origins[hit] + directions[hit] * t_hit[hit][:, None]
        colors = surface_color(surface, frequency=cfg.frequency)
        if cfg.exposure > 0:
            colors = colors + rng.uniform(-cfg.exposure, cfg.exposure)
        image[hit] = np.clip(colors, 0.0, 1.0)
        image = (image.reshape(cfg.height, cfg.width, 3) * 255.0).round()

        name = f"img{i:03d}.png"
        imageio.imwrite(os.path.join(images_dir, name), image.astype(np.uint8))
        image_names.append(name)

    # Reported poses: the true ones, optionally perturbed. The images above
    # were rendered from the *true* poses and are not re-rendered, so the
    # perturbation is residual registration error rather than a different
    # capture.
    poses = true_poses.copy()
    if cfg.pose_error_arcmin > 0:
        angle = np.deg2rad(cfg.pose_error_arcmin / 60.0)
        for i in range(cfg.num_views):
            poses[i] = _rotate_about_center(true_poses[i], rng.normal(size=3), angle)

    # Sparse points with real cross-view tracks: sample the surface, then keep
    # each (point, view) observation that is actually visible, so the model is
    # geometrically consistent rather than merely well-formed.
    surface_points = fibonacci_directions(cfg.num_points) * cfg.radius
    point_colors = (
        surface_color(surface_points, frequency=cfg.frequency) * 255
    ).astype(np.uint8)

    # Batched per view rather than per (point, view): one `cast_rays` call with
    # all of a view's candidate rays instead of num_points separate calls.
    observations_per_point = [[] for _ in range(len(surface_points))]
    for i in range(cfg.num_views):
        viewmat = np.linalg.inv(poses[i])
        local = (viewmat[:3, :3] @ surface_points.T + viewmat[:3, 3:4]).T
        uvw = (K @ local.T).T
        in_front = local[:, 2] > 1e-6
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < cfg.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < cfg.height)
        )
        candidates = np.nonzero(in_front & in_bounds)[0]
        if candidates.size == 0:
            continue
        origin = poses[i][:3, 3]
        offsets = surface_points[candidates] - origin
        distances = np.linalg.norm(offsets, axis=-1)
        directions = offsets / distances[:, None]
        rays = np.concatenate(
            [np.broadcast_to(origin, directions.shape), directions], axis=-1
        ).astype(np.float32)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        # Visible only where the first surface hit is the point itself; the
        # tolerance scales with distance so it means the same thing near and far.
        visible = np.isfinite(t_hit) & (
            np.abs(t_hit - distances) < 1e-2 * distances + 1e-3
        )
        for local_idx in np.nonzero(visible)[0]:
            p_idx = int(candidates[local_idx])
            observations_per_point[p_idx].append(
                (i, (float(uv[p_idx, 0]), float(uv[p_idx, 1])))
            )

    tracks = []
    kept_points = []
    kept_rgb = []
    for p_idx, observations in enumerate(observations_per_point):
        # A point only one image saw gives bundle adjustment no cross-view
        # constraint, so it is not a track.
        if len(observations) >= 2:
            tracks.append(observations)
            kept_points.append(surface_points[p_idx])
            kept_rgb.append(point_colors[p_idx])

    if not tracks:
        raise RuntimeError(
            "no point was seen by two or more views -- check num_views, focal "
            "and cam_dist; a COLMAP model with no cross-view tracks is useless."
        )

    sparse_dir = os.path.join(cfg.out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    write_colmap_reconstruction(
        image_names=image_names,
        camtoworlds=poses,
        Ks=K,
        image_sizes=(cfg.width, cfg.height),
        points_xyz=np.asarray(kept_points),
        tracks=tracks,
        output_dir=sparse_dir,
        points_rgb=np.asarray(kept_rgb, dtype=np.uint8),
    )

    mesh_path = os.path.join(cfg.out_dir, "mesh_gt.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    dense_path = None
    if cfg.write_dense:
        dense_dir = os.path.join(cfg.out_dir, "dense")
        os.makedirs(dense_dir, exist_ok=True)
        dense_dirs = fibonacci_directions(cfg.num_dense_points)
        radii = cfg.radius * (
            1.0 + rng.normal(scale=cfg.dense_noise, size=(cfg.num_dense_points, 1))
        )
        dense_xyz = dense_dirs * radii
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(dense_xyz)
        cloud.colors = o3d.utility.Vector3dVector(
            surface_color(dense_xyz, frequency=cfg.frequency)
        )
        # Outward normals are exact on a sphere; giving them saves Poisson a
        # normal estimate whose default radius is an absolute scene-unit
        # constant (see docs/handoff -- deriving it is open work).
        # Radial jitter leaves the true normal radial, so these stay exact.
        cloud.normals = o3d.utility.Vector3dVector(dense_dirs)
        dense_path = os.path.join(dense_dir, "dense.ply")
        o3d.io.write_point_cloud(dense_path, cloud)

    return {
        "data_dir": cfg.out_dir,
        "images_dir": images_dir,
        "sparse_dir": sparse_dir,
        "mesh_path": mesh_path,
        "dense_path": dense_path,
        "image_names": image_names,
        "camtoworlds": poses,
        "true_camtoworlds": true_poses,
        "K": K,
        "num_tracks": len(tracks),
    }


def main(cfg: Config) -> None:
    out = build(cfg)
    print(
        f"[make_synthetic_capture] wrote {len(out['image_names'])} images, "
        f"{out['num_tracks']} tracked points to {out['data_dir']}"
    )
    print(f"[make_synthetic_capture] ground-truth mesh: {out['mesh_path']}")
    if out["dense_path"]:
        print(f"[make_synthetic_capture] dense cloud: {out['dense_path']}")


if __name__ == "__main__":
    main(tyro.cli(Config))
