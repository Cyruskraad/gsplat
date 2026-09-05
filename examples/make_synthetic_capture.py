# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write a complete, multi-view-consistent synthetic capture to disk.

The photogrammetry pipeline's stages have always been testable *individually*
against synthetic input, but not from the CLI inwards, because the two halves
of a usable fixture lived in different tests and neither was enough on its own:

- ``tests/test_colmap_dataset.py``'s ``_build_synthetic_reconstruction`` writes
  a **real COLMAP model** -- but pairs it with ``rng.uniform(0, 255)`` noise
  images, so nothing on disk is multi-view consistent. Bake a texture from it
  and you get noise.
- ``tests/test_mesh_extraction.py``'s ``_SphereDataset`` renders **genuinely
  consistent views** -- but only in memory, so no CLI can be pointed at it.

This joins them. It writes a ``<data_dir>`` that ``examples/datasets/colmap.py``
``Parser`` loads like any real capture::

    <data_dir>/images/000000.png ...   ray-cast renders of a textured mesh
    <data_dir>/sparse/0/               a real COLMAP model (cameras/images/points3D)
    <data_dir>/dense/dense.ply         a dense point cloud sampled on the surface
    <data_dir>/mesh/surface.obj        the ground-truth mesh the views were rendered from

which is what makes ``extract_mesh.py`` and ``run_pipeline.py`` runnable on a
machine with no GPU, no ``colmap`` and no capture data -- see
``tests/test_extract_mesh_cli.py``.

Views are rendered by ray-casting the mesh with the same
``open3d.t.geometry.RaycastingScene`` :mod:`gsplat.photogrammetry.texturing`
uses for occlusion, and shaded by an analytic function of the hit point. So the
images are exactly consistent with the mesh written beside them -- including
its self-occlusion, which an analytic sphere cannot express (a convex shape
hides nothing from itself; see ``docs/handoff/ISSUES.md``).

Example:

    python examples/make_synthetic_capture.py --data_dir /tmp/capture \\
        --shape sphere_on_plane --num_views 12 --width 128 --height 128

Then, with no checkpoint and no GPU:

    python examples/extract_mesh.py --data_dir /tmp/capture --data_factor 1 \\
        --mesh_path /tmp/capture/mesh/surface.obj --texture_mode atlas \\
        --result_dir /tmp/out

Requires `open3d` (`pip install gsplat[mesh]`) and `pycolmap`.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import tyro

from gsplat.photogrammetry.neural_sfm import write_colmap_reconstruction


def surface_pattern(points: np.ndarray) -> np.ndarray:
    """Colour a surface point, at a spatial frequency detail tests can see.

    Deliberately high-frequency: a wavelength of ~0.14 world units, about three
    times the displacement a 45' pose error produces at the default camera
    distance. ``tests/test_mesh_extraction.py``'s ``_surface_pattern`` has a
    wavelength of roughly half the sphere, which is far coarser than any
    misregistration this pipeline models -- blending several views of it barely
    blurs it at all, so measurements of what averaging costs come out as zero.
    See ``docs/handoff/ISSUES.md`` § "The premise is frequency-dependent".
    """
    points = np.asarray(points, dtype=np.float64)
    return np.stack(
        [
            0.5 + 0.45 * np.sin(45.0 * points[..., 0]),
            0.5 + 0.45 * np.sin(43.0 * points[..., 1]),
            0.5 + 0.45 * np.sin(47.0 * points[..., 2]),
        ],
        axis=-1,
    )


def build_shape(shape: str, resolution: int = 20) -> o3d.geometry.TriangleMesh:
    """The ground-truth surface the capture photographs.

    ``sphere`` is the convex baseline. ``sphere_on_plane`` adds a ground quad
    the sphere rests on, which is what makes the fixture able to exercise
    occlusion and unobserved geometry at all: on a convex shape every face a
    camera should not see is also facing away, so back-face rejection removes
    it and an occlusion guard passes whether or not it works.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    if shape == "sphere":
        pass
    elif shape == "sphere_on_plane":
        # A quad under the sphere, tessellated enough to carry texture and to
        # survive decimation as a surface rather than as two triangles.
        plane = o3d.geometry.TriangleMesh.create_box(width=4.0, height=4.0, depth=0.2)
        # Top face at z = -1.15, a 0.15 gap below the sphere's south pole. They
        # must not *touch*: the pole sits at exactly (0, 0, -1), which is also a
        # vertex of the subdivided box, so a plane at z = -1.0 merges the two
        # surfaces into a non-manifold vertex -- and open3d's compute_uvatlas
        # segfaults on non-manifold input, so the atlas bake refuses it outright
        # (docs/handoff/ISSUES.md). The gap costs nothing: the sphere still
        # occludes the plane and still casts contact occlusion onto it.
        plane.translate((-2.0, -2.0, -1.35))
        plane = plane.subdivide_midpoint(number_of_iterations=2)
        mesh = mesh + plane
    else:
        raise ValueError(
            f"Unknown shape {shape!r}, expected 'sphere' or " "'sphere_on_plane'."
        )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def _camera_poses(
    num_views: int, cam_dist: float, elevation_limit: float = 1.0
) -> np.ndarray:
    """Near-uniform camera placement on a sphere around the origin.

    Fibonacci spacing, the same arrangement ``_SphereDataset`` uses, so (almost)
    every point on the subject is seen by at least one view.

    ``elevation_limit`` < 1 keeps the cameras away from directly underneath,
    which is what a real capture circling a subject does -- and is what leaves
    the underside genuinely unobserved for ``--cull_unobserved`` to find.
    """
    camtoworlds = np.zeros((num_views, 4, 4))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_views):
        cz = 1.0 - 2.0 * (i + 0.5) / num_views
        cz = float(np.clip(cz, -elevation_limit, 1.0))
        r_xy = np.sqrt(max(1.0 - cz * cz, 0.0))
        theta = golden * i
        cam_dir = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), cz])
        cam_dir /= np.linalg.norm(cam_dir)
        cam_pos = cam_dist * cam_dir

        forward = -cam_dir
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        camtoworld = np.eye(4)
        # Columns (right, -up, forward): the OpenCV convention the rest of the
        # pipeline reads, with +x right, +y down, +z into the scene.
        camtoworld[:3, :3] = np.stack([right, -up, forward], axis=1)
        camtoworld[:3, 3] = cam_pos
        camtoworlds[i] = camtoworld
    return camtoworlds


def _perturb_rotation(rotation: np.ndarray, arcmin: float, rng) -> np.ndarray:
    """Rotate about a random axis by ``arcmin``, in place at the optical centre.

    Applied to the pose that gets *reported* while the image keeps the pose it
    was rendered from -- which is exactly what residual SfM error is: the pose
    you have does not quite match the photograph it belongs to, so two views
    disagree about where a surface point lands.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(arcmin / 60.0)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rot = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    return rotation @ rot


def render_views(
    mesh: o3d.geometry.TriangleMesh,
    camtoworlds: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    pattern=surface_pattern,
    exposure: float = 0.0,
    background: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Ray-cast ``mesh`` from each pose and shade hits by ``pattern``.

    Returns ``(N, height, width, 3)`` uint8 images. Rays are cast with open3d's
    ``RaycastingScene``, so occlusion is exact and matches what
    :func:`gsplat.photogrammetry.texturing._view_samples` will later compute
    against the same surface.
    """
    rng = np.random.default_rng(seed)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    # +0.5 puts the ray through the pixel *centre*; the projection convention
    # elsewhere in the pipeline has integer coordinates at pixel corners.
    x1 = (xs - K[0, 2] + 0.5) / K[0, 0]
    y1 = (ys - K[1, 2] + 0.5) / K[1, 1]
    dirs_cam = np.stack([x1, y1, np.ones_like(x1)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    images = np.zeros((len(camtoworlds), height, width, 3), dtype=np.uint8)
    for i, camtoworld in enumerate(camtoworlds):
        R = camtoworld[:3, :3]
        origin = camtoworld[:3, 3]
        dirs_world = np.einsum("ij,hwj->hwi", R, dirs_cam)
        rays = np.concatenate(
            [np.broadcast_to(origin, dirs_world.shape), dirs_world], axis=-1
        ).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))
        t_hit = hits["t_hit"].numpy().reshape(height, width)
        hit = np.isfinite(t_hit)

        image = np.full((height, width, 3), background, dtype=np.float64)
        points = origin[None, None, :] + dirs_world * t_hit[..., None]
        image[hit] = pattern(points[hit])
        if exposure:
            # Only where the subject is: shifting the background too would make
            # the offset trivially recoverable from the empty frame, and no
            # real auto-exposure difference is recoverable that way.
            image[hit] += rng.uniform(-exposure, exposure)
        images[i] = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return images


def _tracks_from_visibility(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    camtoworlds: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    min_views: int = 2,
) -> Tuple[np.ndarray, list]:
    """Which cameras actually see each point, and where it lands in each.

    A COLMAP track has to be *earned*: a point claimed as observed in a view
    that cannot see it is a lie the bundle adjuster would then try to satisfy.
    So each candidate is projected, bounds-checked and ray-cast against the
    mesh, and only points surviving in ``min_views`` views become tracks.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    observations = [[] for _ in range(len(points))]
    for view_idx, camtoworld in enumerate(camtoworlds):
        viewmat = np.linalg.inv(camtoworld)
        cam_pos = camtoworld[:3, 3]
        Xc = (viewmat[:3, :3] @ points.T + viewmat[:3, 3:4]).T
        uvw = (K @ Xc.T).T
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
        ok = (
            (Xc[:, 2] > 1e-4)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        candidates = np.nonzero(ok)[0]
        if candidates.size == 0:
            continue
        dirs = points[candidates] - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=1)
        dirs_n = dirs / np.clip(dists, 1e-8, None)[:, None]
        rays = np.concatenate(
            [np.repeat(cam_pos[None, :], len(candidates), axis=0), dirs_n], axis=1
        ).astype(np.float32)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        # The nearest hit along the ray is this point itself, i.e. nothing
        # else on the surface stands in front of it.
        visible = np.abs(t_hit - dists) < (1e-2 * dists + 1e-3)
        for idx in candidates[visible]:
            observations[idx].append((view_idx, (float(uv[idx, 0]), float(uv[idx, 1]))))

    keep = [i for i, obs in enumerate(observations) if len(obs) >= min_views]
    return np.asarray(keep, dtype=np.int64), [observations[i] for i in keep]


@dataclass
class Config:
    # Directory to write the capture to (becomes --data_dir for the pipeline).
    data_dir: str = "data/synthetic_capture"
    # Ground-truth surface. "sphere_on_plane" is the one that can exercise
    # occlusion and unobserved geometry; a sphere alone is convex.
    shape: Literal["sphere", "sphere_on_plane"] = "sphere_on_plane"
    # Sphere tessellation. Higher is a denser ground-truth mesh.
    resolution: int = 20
    # How many views to render.
    num_views: int = 12
    # Rendered image size, in pixels.
    width: int = 128
    height: int = 128
    # Focal length in pixels. Scaling it with width/height is how you ask for
    # "the same capture, shot at higher resolution".
    focal: Optional[float] = None
    # Camera distance from the origin, in scene units.
    cam_dist: float = 3.5
    # Keep cameras above this fraction of the way down the sphere, so the
    # subject's underside stays genuinely unobserved (as in a real capture
    # that circles rather than tumbles).
    elevation_limit: float = 0.45
    # Rotate each *reported* pose by this angle about its own optical centre,
    # leaving the rendered image alone: residual SfM error, exactly.
    pose_error_arcmin: float = 0.0
    # Give each view its own constant brightness offset drawn from
    # [-exposure, exposure], applied only where it sees the subject. This is
    # the auto-exposure/white-balance drift that makes faces textured from
    # different photographs meet at a visible step.
    exposure: float = 0.0
    # How many points to put in dense/dense.ply.
    num_dense_points: int = 20000
    # How many of the sparse SfM points to attempt tracks for.
    num_sparse_points: int = 400
    # Seed for pose-error directions, exposure offsets and point sampling.
    seed: int = 0


def main(cfg: Config) -> dict:
    """Write the capture, and return a summary of what was written."""
    rng = np.random.default_rng(cfg.seed)
    focal = cfg.focal if cfg.focal is not None else 1.35 * max(cfg.width, cfg.height)
    K = np.array(
        [[focal, 0.0, cfg.width / 2.0], [0.0, focal, cfg.height / 2.0], [0.0, 0.0, 1.0]]
    )

    mesh = build_shape(cfg.shape, resolution=cfg.resolution)
    true_camtoworlds = _camera_poses(
        cfg.num_views, cfg.cam_dist, elevation_limit=cfg.elevation_limit
    )

    images = render_views(
        mesh,
        true_camtoworlds,
        K,
        cfg.width,
        cfg.height,
        exposure=cfg.exposure,
        seed=cfg.seed,
    )

    # The images were rendered from the true poses; only the poses we *report*
    # to COLMAP are perturbed.
    reported = true_camtoworlds.copy()
    if cfg.pose_error_arcmin:
        for i in range(len(reported)):
            reported[i, :3, :3] = _perturb_rotation(
                reported[i, :3, :3], cfg.pose_error_arcmin, rng
            )

    image_dir = os.path.join(cfg.data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_names = [f"{i:06d}.png" for i in range(cfg.num_views)]
    for name, image in zip(image_names, images):
        imageio.imwrite(os.path.join(image_dir, name), image)

    # Sparse model: points sampled on the surface, kept only where they are
    # genuinely visible in at least two views.
    # open3d's sampler has no per-call seed argument on every version, so the
    # determinism is set globally here -- a capture that changed shape between
    # runs would make every measurement against it unreproducible.
    o3d.utility.random.seed(cfg.seed)
    sparse_pcd = mesh.sample_points_uniformly(number_of_points=cfg.num_sparse_points)
    sparse_xyz = np.asarray(sparse_pcd.points)
    keep, tracks = _tracks_from_visibility(
        mesh, sparse_xyz, true_camtoworlds, K, cfg.width, cfg.height
    )
    if keep.size == 0:
        raise RuntimeError(
            "No sampled point was visible in two or more views, so there are "
            "no tracks to write. Check --cam_dist and --focal: the subject is "
            "probably outside every frame."
        )
    sparse_xyz = sparse_xyz[keep]
    sparse_rgb = (np.clip(surface_pattern(sparse_xyz), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )

    sparse_dir = os.path.join(cfg.data_dir, "sparse", "0")
    write_colmap_reconstruction(
        image_names=image_names,
        camtoworlds=reported,
        Ks=K,
        image_sizes=(cfg.width, cfg.height),
        points_xyz=sparse_xyz,
        tracks=tracks,
        output_dir=sparse_dir,
        points_rgb=sparse_rgb,
    )

    # Dense cloud, as COLMAP's patch-match fusion would leave it.
    dense_pcd = mesh.sample_points_uniformly(number_of_points=cfg.num_dense_points)
    dense_xyz = np.asarray(dense_pcd.points)
    dense_pcd.colors = o3d.utility.Vector3dVector(
        np.clip(surface_pattern(dense_xyz), 0.0, 1.0)
    )
    dense_dir = os.path.join(cfg.data_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)
    dense_path = os.path.join(dense_dir, "dense.ply")
    o3d.io.write_point_cloud(dense_path, dense_pcd)

    # The ground-truth surface itself, so --mesh_path has something to texture
    # and a test has something to measure against.
    mesh_dir = os.path.join(cfg.data_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, "surface.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    summary = {
        "data_dir": cfg.data_dir,
        "num_views": cfg.num_views,
        "image_size": [cfg.width, cfg.height],
        "num_sparse_points": int(len(sparse_xyz)),
        "num_dense_points": int(len(dense_xyz)),
        "num_mesh_triangles": int(len(mesh.triangles)),
        "mesh_path": mesh_path,
        "dense_path": dense_path,
        "sparse_dir": sparse_dir,
        "pose_error_arcmin": cfg.pose_error_arcmin,
        "exposure": cfg.exposure,
    }
    print(
        f"[make_synthetic_capture] wrote {cfg.num_views} "
        f"{cfg.width}x{cfg.height} views of '{cfg.shape}' to {cfg.data_dir}: "
        f"{summary['num_sparse_points']} sparse points, "
        f"{summary['num_dense_points']} dense points, "
        f"{summary['num_mesh_triangles']} ground-truth triangles"
    )
    return summary


if __name__ == "__main__":
    main(tyro.cli(Config))
