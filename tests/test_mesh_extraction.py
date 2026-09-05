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

"""Tests for gsplat.photogrammetry.mesh_extraction, on fully synthetic data
(no trained splats / GPU needed): `_tsdf_fuse` and `extract_mesh_poisson`
reconstruct an analytic unit sphere from ray-traced depth maps / a point
cloud respectively.
"""

import numpy as np
import pytest

pytest.importorskip(
    "open3d", reason="open3d is not installed (pip install gsplat[mesh])"
)

from gsplat.photogrammetry.mesh_extraction import _tsdf_fuse, extract_mesh_poisson


def _make_sphere_views(num_views=10, radius=1.0, cam_dist=3.0, width=200, height=150):
    """Ray-traced (color, depth) views of a unit sphere centered at the origin,
    from cameras placed on a larger sphere around it.
    """
    K = np.array(
        [[220.0, 0, width / 2], [0, 220.0, height / 2], [0, 0, 1.0]], dtype=np.float64
    )
    views = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        phi = np.pi / 2 + 0.3 * np.sin(3 * theta)
        cam_pos = cam_dist * np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w = np.stack([right, -up, forward], axis=1)
        R_w2c = R_c2w.T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = -R_w2c @ cam_pos

        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x1 = (xs - K[0, 2] + 0.5) / K[0, 0]
        y1 = (ys - K[1, 2] + 0.5) / K[1, 1]
        dirs_cam = np.stack([x1, y1, np.ones_like(x1)], axis=-1)  # (H, W, 3)
        dirs_cam_norm = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
        dirs_world = np.einsum("ij,hwj->hwi", R_c2w, dirs_cam_norm)

        oc = cam_pos
        b = np.einsum("hwi,i->hw", dirs_world, oc) * 2
        c = np.dot(oc, oc) - radius**2
        disc = b**2 - 4 * c
        valid = disc >= 0
        sqrt_disc = np.sqrt(np.clip(disc, 0, None))
        t0 = (-b - sqrt_disc) / 2
        hit = valid & (t0 > 0)

        ray_depth = np.where(hit, t0, 0.0)
        # ray-depth -> z-depth: divide by |unnormalized camera-space dir| / z,
        # i.e. multiply by cos(angle to principal axis).
        cos_angle = 1.0 / np.linalg.norm(dirs_cam, axis=-1)
        z_depth = (ray_depth * cos_angle).astype(np.float32)

        color = np.zeros((height, width, 3), dtype=np.uint8)
        color[hit] = np.array([200, 120, 60], dtype=np.uint8)

        views.append({"color": color, "depth": z_depth, "K": K, "extrinsic": extrinsic})
    return views


def test_tsdf_fuse_reconstructs_sphere():
    views = _make_sphere_views()
    mesh = _tsdf_fuse(views, voxel_size=0.02, sdf_trunc=0.08, depth_trunc=10.0)

    assert len(mesh.vertices) > 50
    assert len(mesh.triangles) > 50

    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    mean_radius_err = np.mean(np.abs(radii - 1.0))
    assert (
        mean_radius_err < 0.1
    ), f"mesh doesn't look like a unit sphere: {mean_radius_err}"


def test_extract_mesh_poisson_reconstructs_sphere():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(2000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    normals = points.copy()  # outward normals on a unit sphere == the points
    colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))

    mesh = extract_mesh_poisson(points, colors, normals=normals, depth=6)

    assert len(mesh.vertices) > 50
    verts = np.asarray(mesh.vertices)
    radii = np.linalg.norm(verts, axis=1)
    mean_radius_err = np.mean(np.abs(radii - 1.0))
    assert (
        mean_radius_err < 0.15
    ), f"mesh doesn't look like a unit sphere: {mean_radius_err}"


def _surface_pattern(points):
    """A deterministic, high-frequency color for any point on the unit sphere.

    Used as ground truth: it is a function of the *surface point* only, so
    every camera that sees a point agrees on its color and a correct bake must
    reproduce it exactly (up to pixel quantization), independent of which views
    contributed.
    """
    points = np.asarray(points, dtype=np.float64)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack(
        [
            0.5 + 0.45 * np.sin(6.0 * np.arctan2(y, x)),
            0.5 + 0.45 * np.sin(7.0 * z),
            0.5 + 0.45 * np.cos(5.0 * x),
        ],
        axis=-1,
    )


class _SphereDataset:
    """A `Dataset`-like sequence of analytic views of a colored unit sphere.

    Yields the dicts `bake_texture`/`bake_texture_atlas` consume
    (`camtoworld`, `K`, `image`), with each image ray-traced against the
    analytic sphere and shaded by `_surface_pattern` at the hit point.
    """

    def __init__(
        self,
        num_views=24,
        radius=1.0,
        cam_dist=3.5,
        width=192,
        height=192,
        focal=260.0,
        pattern=None,
        pose_error_arcmin=0.0,
        exposure=0.0,
        seed=0,
    ):
        """
        Args:
            pattern: Surface color function, defaulting to `_surface_pattern`.
                Overridden by tests that need detail at a *specific* spatial
                frequency -- `_surface_pattern` is deliberately smooth relative
                to a camera pixel, which makes it useless for measuring what
                blending does to high-frequency detail.
            pose_error_arcmin: Rotate each *reported* `camtoworld` by this angle
                about the camera's own optical centre, leaving the rendered
                image alone. That is exactly residual SfM error: the pose you
                have does not quite match the photograph it belongs to, so
                views disagree about where a surface point lands.
            focal: Focal length in pixels. Scaling it together with
                `width`/`height` is how a test asks for "the same capture, shot
                at higher resolution" -- the subject's projected area then
                scales as focal squared, which is an exact, analytic handle on
                anything that reasons about pixel evidence.
            exposure: Give each view its own constant brightness offset,
                drawn uniformly from [-exposure, exposure], applied only where
                the view actually sees the sphere. Simulates the auto-exposure
                and white-balance drift that makes neighbouring faces textured
                from different photographs meet at a visible step.
            seed: For the pose-error directions and exposure offsets, so a
                perturbed dataset is reproducible.
        """
        torch = pytest.importorskip("torch")
        if pattern is None:
            pattern = _surface_pattern
        rng = np.random.default_rng(seed)
        self.width, self.height = width, height
        K = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]],
            dtype=np.float64,
        )
        self._items = []
        # Fibonacci sphere: near-uniform camera coverage, so (almost) every
        # point on the sphere is seen by at least one view.
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(num_views):
            cz = 1.0 - 2.0 * (i + 0.5) / num_views
            r_xy = np.sqrt(max(1.0 - cz * cz, 0.0))
            theta = golden * i
            cam_dir = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), cz])
            cam_pos = cam_dist * cam_dir

            forward = -cam_dir
            world_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(forward, world_up)) > 0.99:
                world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            R_c2w = np.stack([right, -up, forward], axis=1)
            camtoworld = np.eye(4)
            camtoworld[:3, :3] = R_c2w
            camtoworld[:3, 3] = cam_pos

            ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            x1 = (xs - K[0, 2] + 0.5) / K[0, 0]
            y1 = (ys - K[1, 2] + 0.5) / K[1, 1]
            dirs_cam = np.stack([x1, y1, np.ones_like(x1)], axis=-1)
            dirs_world = np.einsum(
                "ij,hwj->hwi",
                R_c2w,
                dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True),
            )
            b = 2.0 * np.einsum("hwi,i->hw", dirs_world, cam_pos)
            c = np.dot(cam_pos, cam_pos) - radius**2
            disc = b**2 - 4 * c
            hit = (disc >= 0) & (-b - np.sqrt(np.clip(disc, 0, None)) > 0)
            t0 = (-b - np.sqrt(np.clip(disc, 0, None))) / 2.0
            hit_points = cam_pos[None, None, :] + dirs_world * t0[..., None]

            image = np.zeros((height, width, 3), dtype=np.float64)
            image[hit] = pattern(hit_points[hit])
            if exposure:
                # Only where the sphere is: shifting the background too would
                # make the offset recoverable from the empty frame.
                image[hit] += rng.uniform(-exposure, exposure)
            image = (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)

            # The image was rendered from the true pose; only the pose we
            # *report* is perturbed.
            if pose_error_arcmin:
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = np.deg2rad(pose_error_arcmin / 60.0)
                cross = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                rot = (
                    np.eye(3)
                    + np.sin(angle) * cross
                    + (1.0 - np.cos(angle)) * (cross @ cross)
                )
                camtoworld[:3, :3] = camtoworld[:3, :3] @ rot

            self._items.append(
                {
                    "camtoworld": torch.from_numpy(camtoworld),
                    "K": torch.from_numpy(K),
                    "image": torch.from_numpy(image.astype(np.float32)),
                }
            )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def _unit_sphere_mesh(resolution=10):
    o3d = pytest.importorskip("open3d")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    mesh.compute_vertex_normals()
    return mesh


def test_bake_texture_colors_vertices_from_views():
    """`bake_texture` reproduces the analytic surface color at each vertex."""
    from gsplat.photogrammetry.mesh_extraction import bake_texture

    mesh = _unit_sphere_mesh(resolution=10)
    dataset = _SphereDataset()

    bake_texture(mesh, dataset)

    vertices = np.asarray(mesh.vertices)
    baked = np.asarray(mesh.vertex_colors)
    assert baked.shape == vertices.shape
    # A vertex sits on the analytic sphere, so its ground-truth color is
    # exactly the pattern evaluated there.
    expected = _surface_pattern(vertices / np.linalg.norm(vertices, axis=1)[:, None])
    mean_err = np.abs(baked - expected).mean()
    assert mean_err < 0.05, f"vertex colors don't match ground truth: {mean_err}"


def test_bake_texture_atlas_matches_ground_truth_through_mesh_uvs():
    """The atlas is correct *as addressed by the mesh's own UVs*.

    This is the test that pins the UV convention any external tool will use:
    it looks each triangle corner's `triangle_uvs` entry up in the returned
    texture (OBJ convention, v=0 at the bottom of the image) and checks the
    texel there carries that corner's ground-truth surface color. A flipped or
    transposed atlas fails this even though the bake itself is self-consistent.
    """
    from gsplat.photogrammetry.mesh_extraction import bake_texture_atlas

    mesh = _unit_sphere_mesh(resolution=10)
    dataset = _SphereDataset()
    texture_size = 256

    mesh, texture = bake_texture_atlas(
        mesh, dataset, texture_size=texture_size, dilation=4
    )

    assert texture.shape == (texture_size, texture_size, 3)
    assert texture.dtype == np.uint8
    assert len(mesh.triangle_uvs) == 3 * len(mesh.triangles)
    assert len(mesh.textures) == 1

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)

    corner_xyz = vertices[triangles].reshape(-1, 3)
    corner_uv = uvs.reshape(-1, 2)
    # OBJ/`compute_uvatlas` convention: u -> column, v -> row from the bottom.
    cols = np.clip((corner_uv[:, 0] * texture_size).astype(int), 0, texture_size - 1)
    rows = np.clip(
        ((1.0 - corner_uv[:, 1]) * texture_size).astype(int), 0, texture_size - 1
    )
    sampled = texture[rows, cols] / 255.0
    expected = _surface_pattern(
        corner_xyz / np.linalg.norm(corner_xyz, axis=1)[:, None]
    )

    mean_err = np.abs(sampled - expected).mean()
    assert mean_err < 0.08, f"atlas doesn't match ground truth at mesh UVs: {mean_err}"

    # Sanity-check the convention is actually being exercised: sampling the
    # vertically-flipped texel must be measurably worse.
    flipped = texture[texture_size - 1 - rows, cols] / 255.0
    assert np.abs(flipped - expected).mean() > 2 * mean_err


def test_bake_texture_atlas_resolves_detail_finer_than_the_mesh():
    """The atlas carries sub-triangle detail that vertex colors cannot.

    On a deliberately coarse mesh, per-vertex colors can only represent the
    pattern at vertex density, while the atlas samples it per texel -- so
    against the same ground truth the atlas must be substantially more
    accurate. This is the whole reason the atlas path exists.
    """
    from gsplat.photogrammetry.mesh_extraction import bake_texture, bake_texture_atlas

    dataset = _SphereDataset()
    texture_size = 256

    coarse = _unit_sphere_mesh(resolution=4)
    atlas_mesh, texture = bake_texture_atlas(
        _unit_sphere_mesh(resolution=4), dataset, texture_size=texture_size, dilation=4
    )
    bake_texture(coarse, dataset)

    # Evaluate both at triangle centroids -- points strictly *between*
    # vertices, where the two representations genuinely differ.
    triangles = np.asarray(coarse.triangles)
    vertices = np.asarray(coarse.vertices)
    centroids = vertices[triangles].mean(axis=1)
    expected = _surface_pattern(centroids / np.linalg.norm(centroids, axis=1)[:, None])

    vertex_colors = np.asarray(coarse.vertex_colors)
    vertex_pred = vertex_colors[triangles].mean(axis=1)

    uvs = np.asarray(atlas_mesh.triangle_uvs).reshape(-1, 3, 2)
    centroid_uv = uvs.mean(axis=1)
    cols = np.clip((centroid_uv[:, 0] * texture_size).astype(int), 0, texture_size - 1)
    rows = np.clip(
        ((1.0 - centroid_uv[:, 1]) * texture_size).astype(int), 0, texture_size - 1
    )
    atlas_pred = texture[rows, cols] / 255.0

    atlas_err = np.abs(atlas_pred - expected).mean()
    vertex_err = np.abs(vertex_pred - expected).mean()
    assert atlas_err < vertex_err, (
        f"atlas ({atlas_err}) should beat vertex colors ({vertex_err}) on "
        "sub-triangle detail"
    )


def test_bake_texture_atlas_rejects_non_manifold_mesh():
    """Non-manifold input must raise, not crash the process.

    open3d's `compute_uvatlas` segfaults on a non-manifold mesh rather than
    raising, which would take a whole pipeline run down; `bake_texture_atlas`
    checks up front instead.
    """
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import bake_texture_atlas

    # Three triangles sharing one edge -- a classic non-manifold edge.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    assert not mesh.is_edge_manifold(allow_boundary_edges=True)

    with pytest.raises(ValueError, match="non-manifold"):
        bake_texture_atlas(mesh, _SphereDataset(num_views=1), texture_size=32)


def test_bake_texture_atlas_rejects_empty_mesh():
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import bake_texture_atlas

    empty = o3d.geometry.TriangleMesh()
    with pytest.raises(ValueError, match="no triangles"):
        bake_texture_atlas(empty, _SphereDataset(num_views=1), texture_size=32)

    with pytest.raises(ValueError, match="texture_size"):
        bake_texture_atlas(
            _unit_sphere_mesh(resolution=4),
            _SphereDataset(num_views=1),
            texture_size=0,
        )


def test_bake_texture_atlas_writes_a_textured_obj(tmp_path):
    """The returned mesh round-trips through OBJ with its texture attached."""
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import bake_texture_atlas

    mesh, texture = bake_texture_atlas(
        _unit_sphere_mesh(resolution=8),
        _SphereDataset(num_views=8),
        texture_size=64,
        dilation=2,
    )
    out = tmp_path / "mesh.obj"
    assert o3d.io.write_triangle_mesh(str(out), mesh)
    assert out.exists()
    assert (tmp_path / "mesh.mtl").exists()
    assert (tmp_path / "mesh_0.png").exists()

    loaded = o3d.io.read_triangle_mesh(str(out), True)
    assert len(loaded.textures) >= 1
    assert np.asarray(loaded.triangle_uvs).shape == (3 * len(mesh.triangles), 2)
    round_tripped = np.asarray(
        [np.asarray(t) for t in loaded.textures if np.asarray(t).size][0]
    )
    assert round_tripped.shape == texture.shape
    np.testing.assert_array_equal(round_tripped, texture)


def test_fill_texture_holes_pads_across_seams_without_wrapping():
    """Holes are filled from real neighbors only -- never across the border."""
    from gsplat.photogrammetry.mesh_extraction import _fill_texture_holes

    texture = np.zeros((5, 5, 3), dtype=np.float64)
    filled = np.zeros((5, 5), dtype=bool)
    texture[2, 2] = [1.0, 0.0, 0.0]
    filled[2, 2] = True

    grown = _fill_texture_holes(texture, filled, iterations=1)
    # One iteration grows into the 4-neighborhood, and nowhere else.
    np.testing.assert_allclose(grown[1, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grown[3, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grown[2, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grown[2, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(grown[0, 0], [0.0, 0.0, 0.0])

    # A color on the left edge must not wrap around to the right edge.
    texture = np.zeros((4, 4, 3), dtype=np.float64)
    filled = np.zeros((4, 4), dtype=bool)
    texture[1, 0] = [0.0, 1.0, 0.0]
    filled[1, 0] = True
    grown = _fill_texture_holes(texture, filled, iterations=1)
    np.testing.assert_allclose(grown[1, 3], [0.0, 0.0, 0.0])

    # Zero iterations is a no-op that doesn't mutate the caller's array.
    untouched = _fill_texture_holes(texture, filled, iterations=0)
    np.testing.assert_allclose(untouched, texture)


def test_bake_mesh_texture_dispatches_and_falls_back():
    """The CLI entry point picks a mode, and degrades instead of failing."""
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import bake_mesh_texture

    dataset = _SphereDataset(num_views=8)

    # "vertex" mode reports no atlas, so callers know to write .ply.
    mesh, texture = bake_mesh_texture(
        _unit_sphere_mesh(resolution=6), dataset, mode="vertex"
    )
    assert texture is None
    assert mesh.has_vertex_colors()

    # "atlas" mode returns the texture, so callers know to write .obj.
    mesh, texture = bake_mesh_texture(
        _unit_sphere_mesh(resolution=6), dataset, mode="atlas", texture_size=64
    )
    assert texture is not None and texture.shape == (64, 64, 3)
    assert len(mesh.triangle_uvs) == 3 * len(mesh.triangles)

    with pytest.raises(ValueError, match="Unknown texture mode"):
        bake_mesh_texture(_unit_sphere_mesh(resolution=4), dataset, mode="bogus")

    # A mesh that can't be unwrapped falls back to vertex colors with a
    # warning rather than losing the mesh at the end of a long run...
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    non_manifold = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    non_manifold.compute_vertex_normals()

    with pytest.warns(RuntimeWarning, match="falling back to per-vertex"):
        mesh, texture = bake_mesh_texture(
            non_manifold, dataset, mode="atlas", texture_size=32
        )
    assert texture is None
    assert mesh.has_vertex_colors()

    # ...unless the caller explicitly asks to see the failure.
    with pytest.raises(ValueError, match="non-manifold"):
        bake_mesh_texture(
            non_manifold,
            dataset,
            mode="atlas",
            texture_size=32,
            allow_atlas_fallback=False,
        )


def test_simplify_mesh_hits_the_triangle_budget_and_keeps_the_shape():
    """Decimation should cut triangles hard while staying on the surface."""
    from gsplat.photogrammetry.mesh_extraction import simplify_mesh
    from gsplat.photogrammetry.metrics import point_to_mesh_distance

    dense = _unit_sphere_mesh(resolution=20)
    assert len(dense.triangles) > 1000

    low = simplify_mesh(dense, target_triangles=200)
    assert len(low.triangles) <= 260, len(low.triangles)
    assert len(low.triangles) >= 100
    # The input is left alone.
    assert len(dense.triangles) > 1000

    # Points sampled from the dense mesh still lie close to the decimated one:
    # decimation removed triangles, not geometry.
    sampled = np.asarray(dense.sample_points_uniformly(2000).points)
    fit = point_to_mesh_distance(sampled, low)
    assert fit["mean"] < 0.02, fit["mean"]

    with pytest.raises(ValueError, match="target_triangles"):
        simplify_mesh(dense, target_triangles=0)


def _bumpy_sphere(resolution=48, amplitude=0.06):
    """A unit sphere displaced radially by a smooth high-frequency bump.

    A plain sphere is useless for testing normal-map baking: its interpolated
    vertex normals are already essentially the exact analytic normals, so a
    decimated sphere has no detail for the map to recover and an 8-bit map can
    only add quantization noise. This surface carries detail a smooth low-poly
    genuinely cannot represent, which is the case the feature exists for.
    """
    o3d = pytest.importorskip("open3d")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    unit = np.asarray(mesh.vertices)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    radius = 1.0 + amplitude * np.sin(6 * np.arctan2(unit[:, 1], unit[:, 0])) * np.sin(
        5 * np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    )
    mesh.vertices = o3d.utility.Vector3dVector(unit * radius[:, None])
    mesh.compute_vertex_normals()
    return mesh


def _closest_point_normals(dense, points):
    """Dense-mesh normals at the points nearest `points`.

    Deliberately a *closest-point* query -- a different code path from the
    along-the-normal ray cast `bake_normal_map` uses -- so the test checks the
    bake against an independent answer rather than against itself.
    """
    o3d = pytest.importorskip("open3d")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dense))
    found = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
    corner_normals = np.asarray(dense.vertex_normals)[
        np.asarray(dense.triangles)[found["primitive_ids"].numpy()]
    ]
    bary = found["primitive_uvs"].numpy()
    weights = np.stack(
        [1.0 - bary[:, 0] - bary[:, 1], bary[:, 0], bary[:, 1]], axis=-1
    )[..., None]
    normals = (corner_normals * weights).sum(axis=1)
    return normals / np.clip(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None
    )


@pytest.mark.parametrize("space", ["tangent", "object"])
def test_bake_normal_map_recovers_detail_the_low_poly_lacks(space):
    """The baked map must reconstruct the dense surface's normals on a
    low-poly that cannot represent them -- the whole point of the feature.

    Ground truth comes from an independent closest-point query against the
    dense mesh, not from the bake's own ray cast.
    """
    from gsplat.photogrammetry.mesh_extraction import (
        _unwrap_and_rasterize,
        bake_normal_map,
        simplify_mesh,
    )

    dense = _bumpy_sphere()
    low = simplify_mesh(_unit_sphere_mesh(resolution=8), target_triangles=200)
    texture_size = 256

    low, normal_map, stats = bake_normal_map(
        dense, low, texture_size=texture_size, space=space
    )

    assert normal_map.shape == (texture_size, texture_size, 3)
    assert normal_map.dtype == np.uint8
    assert stats["hit_fraction"] > 0.8, stats
    assert stats["space"] == space
    assert stats["num_hits"] <= stats["num_texels"]

    # Recover the same texel frame the bake used, then decode the map there.
    atlas = _unwrap_and_rasterize(low, texture_size, with_tangents=(space == "tangent"))
    decoded = normal_map[atlas.rows, atlas.cols] / 255.0 * 2.0 - 1.0
    if space == "tangent":
        bitangent = np.cross(atlas.normals, atlas.tangents)
        world = (
            decoded[:, 0:1] * atlas.tangents
            + decoded[:, 1:2] * bitangent
            + decoded[:, 2:3] * atlas.normals
        )
    else:
        world = decoded
    world /= np.clip(np.linalg.norm(world, axis=1, keepdims=True), 1e-12, None)

    truth = _closest_point_normals(dense, atlas.positions)
    baked_err = np.abs(world - truth).mean()
    base_err = np.abs(atlas.normals - truth).mean()

    # The low-poly is genuinely wrong about this surface; the map should fix
    # most of that, not merely tie.
    assert base_err > 0.08, f"test surface has too little detail to recover: {base_err}"
    assert baked_err < base_err / 3.0, (
        f"normal map ({baked_err}) should be far closer to the dense surface "
        f"than the low-poly's own normals ({base_err})"
    )


def test_bake_normal_map_tangent_space_is_mostly_flat_and_valid():
    """A tangent-space map of a smooth surface should sit near +Z.

    A map whose blue channel is not dominant means the tangent frame is wrong
    -- the classic symptom of a flipped bitangent or an un-orthogonalized
    tangent, which looks like inverted lighting in a renderer.
    """
    from gsplat.photogrammetry.mesh_extraction import bake_normal_map, simplify_mesh

    dense = _unit_sphere_mesh(resolution=24)
    low = simplify_mesh(_unit_sphere_mesh(resolution=6), target_triangles=120)
    _, normal_map, stats = bake_normal_map(dense, low, texture_size=128)

    decoded = normal_map.reshape(-1, 3) / 255.0 * 2.0 - 1.0
    assert (decoded[:, 2] > 0).mean() > 0.98, "tangent-space normals must face +Z"
    # Unit length, so the map is physically decodable rather than washed out.
    lengths = np.linalg.norm(decoded, axis=1)
    assert np.abs(lengths - 1.0).mean() < 0.05


def test_bake_normal_map_validates_its_inputs():
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import bake_normal_map

    dense = _unit_sphere_mesh(resolution=8)
    low = _unit_sphere_mesh(resolution=6)

    with pytest.raises(ValueError, match="Unknown normal-map space"):
        bake_normal_map(dense, low, texture_size=32, space="world")

    with pytest.raises(ValueError, match="high mesh with no triangles"):
        bake_normal_map(o3d.geometry.TriangleMesh(), low, texture_size=32)

    # The low mesh goes through the same manifold guard as the color atlas.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    non_manifold = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    non_manifold.compute_vertex_normals()
    with pytest.raises(ValueError, match="non-manifold"):
        bake_normal_map(dense, non_manifold, texture_size=32)


def test_albedo_and_normal_maps_share_one_uv_layout():
    """An asset's albedo and normal map must be addressed by the same UVs.

    open3d's `compute_uvatlas` is *not* deterministic -- unwrapping the same
    mesh twice gives different layouts. So baking a color atlas and then a
    normal map must reuse the first bake's UVs; otherwise the normal map is
    sampled through the albedo's coordinates and the asset is silently wrong.
    """
    from gsplat.photogrammetry.mesh_extraction import (
        _unwrap_and_rasterize,
        bake_normal_map,
        bake_texture_atlas,
        simplify_mesh,
    )

    dense = _bumpy_sphere()
    low = simplify_mesh(_unit_sphere_mesh(resolution=8), target_triangles=200)

    low, _ = bake_texture_atlas(low, _SphereDataset(num_views=8), texture_size=64)
    albedo_uvs = np.asarray(low.triangle_uvs).copy()

    low, normal_map, _ = bake_normal_map(dense, low, texture_size=64)
    np.testing.assert_allclose(np.asarray(low.triangle_uvs), albedo_uvs)

    # And the guard is real: two independent unwraps of the same mesh disagree.
    fresh_a = _unwrap_and_rasterize(low, 64, reuse_uvs=False).triangle_uvs
    fresh_b = _unwrap_and_rasterize(low, 64, reuse_uvs=False).triangle_uvs
    assert not np.allclose(fresh_a, fresh_b), (
        "compute_uvatlas became deterministic -- the reuse path is no longer "
        "load-bearing and this test's premise needs revisiting"
    )


def test_bake_ambient_occlusion_matches_known_geometry():
    """AO must be ~fully open on a convex shape and measurably closed inside a
    concave one -- the two cases whose answer is known without a renderer.

    A sphere is convex, so no ray leaving the surface into the outward
    hemisphere can hit it again: AO must be ~1 everywhere. A torus is not, and
    its inner ring sees a large part of its own tube: AO there must be clearly
    lower than on the outer ring. A bake that returned a constant (or that
    self-intersected at the ray origin and blackened everything) fails both.
    """
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import (
        _unwrap_and_rasterize,
        bake_ambient_occlusion,
    )

    sphere = _unit_sphere_mesh(resolution=20)
    sphere, ao_map, stats = bake_ambient_occlusion(
        sphere, texture_size=96, num_samples=48
    )
    assert ao_map.shape == (96, 96, 3)
    assert ao_map.dtype == np.uint8
    assert stats["num_samples"] == 48
    assert stats["mean_ao"] > 0.98, stats
    assert stats["min_ao"] > 0.8, stats

    torus = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.0, tube_radius=0.35, radial_resolution=60, tubular_resolution=30
    )
    torus.compute_vertex_normals()
    torus, ao_map, stats = bake_ambient_occlusion(
        torus, texture_size=96, num_samples=48
    )
    atlas = _unwrap_and_rasterize(torus, 96, with_tangents=True)
    openness = ao_map[atlas.rows, atlas.cols][:, 0] / 255.0
    # Distance from the torus axis separates the inner ring from the outer.
    axis_distance = np.linalg.norm(atlas.positions[:, :2], axis=1)
    inner = axis_distance < 0.85
    outer = axis_distance > 1.15
    assert inner.any() and outer.any()
    assert openness[inner].mean() < 0.9
    assert openness[outer].mean() > 0.95
    assert openness[inner].mean() < openness[outer].mean() - 0.15


def test_bake_ambient_occlusion_is_deterministic_and_validated():
    """Same seed, same map -- a Monte-Carlo bake nobody can reproduce is not
    a reviewable artifact."""
    from gsplat.photogrammetry.mesh_extraction import bake_ambient_occlusion

    # Bake twice onto the *same* mesh: after the first bake it carries UVs, so
    # the second reuses them. Two separately unwrapped copies would legitimately
    # differ, because open3d's unwrapper is not deterministic -- the seed fixes
    # the sampling, not the atlas.
    sphere = _unit_sphere_mesh(resolution=8)
    sphere, map_a, stats_a = bake_ambient_occlusion(
        sphere, texture_size=64, num_samples=16, seed=7
    )
    sphere, map_b, stats_b = bake_ambient_occlusion(
        sphere, texture_size=64, num_samples=16, seed=7
    )
    np.testing.assert_array_equal(map_a, map_b)
    assert stats_a == stats_b

    with pytest.raises(ValueError, match="num_samples"):
        bake_ambient_occlusion(sphere, texture_size=32, num_samples=0)


def test_bake_ambient_occlusion_shares_the_atlas_with_the_other_maps():
    """All three maps of an asset must be addressed by one UV layout."""
    from gsplat.photogrammetry.mesh_extraction import (
        bake_ambient_occlusion,
        bake_normal_map,
        bake_texture_atlas,
        simplify_mesh,
    )

    dense = _bumpy_sphere()
    low = simplify_mesh(_unit_sphere_mesh(resolution=8), target_triangles=200)

    low, _ = bake_texture_atlas(low, _SphereDataset(num_views=8), texture_size=64)
    albedo_uvs = np.asarray(low.triangle_uvs).copy()
    low, _, _ = bake_normal_map(dense, low, texture_size=64)
    low, _, _ = bake_ambient_occlusion(low, occluder_mesh=dense, texture_size=64)

    np.testing.assert_allclose(np.asarray(low.triangle_uvs), albedo_uvs)


def test_bake_ambient_occlusion_cages_rays_against_a_separate_occluder():
    """Baking against a *different* mesh must lift ray origins clear of it.

    Decimation cuts corners, so most of a simplified mesh's surface lies
    *inside* the dense mesh it came from (~80% of texels here). A ray starting
    under the occluder hits it immediately, so without a cage the whole map
    bakes uniformly dark -- a result that looks like heavy occlusion and is
    entirely an artifact.
    """
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import (
        _unwrap_and_rasterize,
        bake_ambient_occlusion,
        simplify_mesh,
    )

    dense = _bumpy_sphere()
    low = simplify_mesh(dense, target_triangles=400)

    # Establish the premise: the low surface really does sit inside the dense.
    atlas = _unwrap_and_rasterize(low, 96)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dense))
    signed = scene.compute_signed_distance(
        o3d.core.Tensor(atlas.positions.astype(np.float32))
    ).numpy()
    assert (signed < 0).mean() > 0.5, "expected the decimated surface to dip inside"

    _, _, caged = bake_ambient_occlusion(
        simplify_mesh(dense, target_triangles=400),
        occluder_mesh=dense,
        texture_size=96,
        num_samples=32,
    )
    # The default cross-mesh cage must clear the deepest excursion.
    assert caged["cage"] > float(-signed.min())
    assert caged["mean_ao"] > 0.9, caged

    # Force the self-occlusion cage on the cross-mesh bake and the artifact
    # comes straight back -- so the cage is what is doing the work.
    _, _, uncaged = bake_ambient_occlusion(
        simplify_mesh(dense, target_triangles=400),
        occluder_mesh=dense,
        texture_size=96,
        num_samples=32,
        cage=1e-4 * 2.0,
    )
    assert uncaged["mean_ao"] < 0.5, uncaged


class _ContaminatedSphereDataset(_SphereDataset):
    """`_SphereDataset` with a transient occluder over part of some frames.

    Stands in for what a real capture does to a texture bake: something walks
    through the scene, a surface goes specular, one camera is slightly
    misregistered. Those views disagree with the rest, and a plain weighted
    mean has no way to prefer the majority.

    The occluder covers only *part* of each affected frame, which matters: a
    whole-frame corruption makes some surface points majority-wrong, and no
    estimator centred on the majority can recover those. That regime is what
    `--mask_dir` exists for; this one is what robust fusion is for.
    """

    def __init__(self, num_views=36, num_corrupted=6, blob_fraction=0.35, **kwargs):
        super().__init__(num_views=num_views, **kwargs)
        torch = pytest.importorskip("torch")
        for index in range(num_corrupted):
            item = dict(self._items[index])
            image = item["image"].numpy().copy()
            width = int(image.shape[1] * blob_fraction)
            # Slide the blob across frames, so no surface point is hidden
            # behind it in a majority of the views that see it.
            offset = (index * 11) % max(1, image.shape[1] - width)
            image[:, offset : offset + width] = np.array(
                [255.0, 0.0, 255.0], dtype=np.float32
            )
            item["image"] = torch.from_numpy(image)
            self._items[index] = item


def _contamination_weight_fraction(mesh, dataset, points, normals):
    """Per point, the share of observation *weight* that is the occluder.

    Used to establish a test's premise rather than assume it: robust fusion
    can only recover points whose bad observations are a minority.
    """
    o3d = pytest.importorskip("open3d")
    from gsplat.photogrammetry.mesh_extraction import _view_samples

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bad = np.zeros(len(points))
    total = np.zeros(len(points))
    magenta = np.array([1.0, 0.0, 1.0])
    for chunk, sampled, weight, _view in _view_samples(
        scene, o3d, dataset, points, normals, None, 1 << 20
    ):
        is_bad = np.all(np.abs(sampled - magenta) < 0.02, axis=1)
        np.add.at(total, chunk, weight)
        np.add.at(bad, chunk, weight * is_bad)
    seen = total > 0
    return bad[seen] / total[seen]


def test_outlier_clipping_rejects_views_that_disagree():
    """Sigma-clipped fusion must recover the true colour where the mean can't.

    Ground truth is the analytic pattern the clean views were rendered from,
    so this measures both estimators against the same known answer rather than
    against each other.
    """
    from gsplat.photogrammetry.mesh_extraction import bake_texture

    dataset = _ContaminatedSphereDataset()
    mesh = _unit_sphere_mesh(resolution=10)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    # Premise: the occluder must be a per-point minority, or nothing centred
    # on the majority could possibly fix it.
    contamination = _contamination_weight_fraction(mesh, dataset, vertices, normals)
    assert contamination.max() < 0.5, contamination.max()
    assert contamination.mean() > 0.05, "the occluder should actually be present"

    plain = _unit_sphere_mesh(resolution=10)
    bake_texture(plain, dataset)
    robust = _unit_sphere_mesh(resolution=10)
    bake_texture(robust, dataset, outlier_sigma=1.5)

    expected = _surface_pattern(vertices / np.linalg.norm(vertices, axis=1)[:, None])
    plain_err = np.abs(np.asarray(plain.vertex_colors) - expected).mean()
    robust_err = np.abs(np.asarray(robust.vertex_colors) - expected).mean()

    # The contamination must actually have hurt, or the test proves nothing.
    assert plain_err > 0.02, plain_err
    assert robust_err < plain_err / 2.0, (plain_err, robust_err)


def test_outlier_clipping_leaves_clean_data_alone():
    """With no disagreement there is nothing to reject, so the robust bake
    must match the plain one closely -- clipping must not eat good samples."""
    from gsplat.photogrammetry.mesh_extraction import bake_texture

    dataset = _SphereDataset(num_views=24)
    plain = _unit_sphere_mesh(resolution=10)
    bake_texture(plain, dataset)
    robust = _unit_sphere_mesh(resolution=10)
    bake_texture(robust, dataset, outlier_sigma=1.5)

    difference = np.abs(
        np.asarray(plain.vertex_colors) - np.asarray(robust.vertex_colors)
    ).mean()
    assert difference < 0.02, difference


def test_outlier_clipping_keeps_sparsely_observed_points():
    """A point seen by only a couple of views keeps its mean.

    With two or three samples the "spread" is noise, and clipping against it
    would throw away good data and leave the point unshaded.
    """
    from gsplat.photogrammetry.mesh_extraction import _bake_points_from_views

    mesh = _unit_sphere_mesh(resolution=10)
    dataset = _ContaminatedSphereDataset(num_views=24, num_corrupted=6)
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    _, plain_weight = _bake_points_from_views(mesh, dataset, points, normals)
    _, robust_weight = _bake_points_from_views(
        mesh, dataset, points, normals, outlier_sigma=1.5, min_views_for_clipping=100
    )
    # Every point falls under the threshold, so nothing may be clipped at all.
    np.testing.assert_allclose(robust_weight, plain_weight)

    # And no point that was observed ends up with no colour at all.
    _, clipped_weight = _bake_points_from_views(
        mesh, dataset, points, normals, outlier_sigma=0.01
    )
    assert (clipped_weight[plain_weight > 0] > 0).all()


# ---------------------------------------------------------------------------
# Decimation to a fit target, rather than to a triangle count
# ---------------------------------------------------------------------------


def _sphere_cloud(num_points=20000, seed=0):
    """Points exactly on the unit sphere -- the fit's analytic ground truth."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(num_points, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def test_point_spacing_matches_a_grid_of_known_spacing():
    """A cubic grid's k-NN spacing is known in closed form.

    For k=4 on a grid of pitch `s`, every interior, face and edge point has at
    least four neighbours at exactly `s`; only the eight corners reach further,
    to `sqrt(2)*s`. So the mean over the whole grid is pinned between `s` and
    `(3 + sqrt(2))/4 * s` -- checkable without reference to the implementation.
    """
    from gsplat.photogrammetry.mesh_extraction import _point_spacing

    pitch = 0.37
    axis = np.arange(10) * pitch
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)

    spacing = _point_spacing(points)
    assert pitch <= spacing <= (3.0 + np.sqrt(2.0)) / 4.0 * pitch, spacing
    # Corners are 8 of 1000 points, so the mean sits very near the pitch.
    assert spacing == pytest.approx(pitch, rel=0.01)

    # Exactly linear in scale: a cloud twice as spread out is twice as coarse.
    assert _point_spacing(points * 3.0) == pytest.approx(3.0 * spacing)


def test_point_spacing_rejects_a_cloud_too_small_to_measure():
    from gsplat.photogrammetry.mesh_extraction import _point_spacing

    with pytest.raises(ValueError, match="at least 5 points"):
        _point_spacing(np.zeros((3, 3)))


def test_simplify_to_error_trades_triangles_for_fit_monotonically():
    """A looser fit target must buy a smaller mesh, and the target must hold.

    Ground truth is analytic: the reference cloud lies exactly on the unit
    sphere, so cloud-to-mesh distance is the mesh's true deviation from the
    sphere, not an artifact of a noisy reference.
    """
    from gsplat.photogrammetry.mesh_extraction import simplify_mesh_to_error
    from gsplat.photogrammetry.metrics import point_to_mesh_distance

    points = _sphere_cloud()
    dense = _unit_sphere_mesh(resolution=40)

    results = {}
    for ratio in (0.25, 1.0, 4.0):
        mesh, stats = simplify_mesh_to_error(dense, points, error_over_spacing=ratio)
        results[ratio] = stats
        assert stats["target_met"], stats
        # The guarantee is about the *returned* mesh, so re-measure it here
        # rather than trusting the search's own bookkeeping. Quadric
        # decimation is only roughly monotone in the triangle count, so a
        # binary search's final bracket is not by itself a promise.
        measured = point_to_mesh_distance(points, mesh)["mean"]
        assert measured <= stats["max_error"], (measured, stats["max_error"])
        assert measured == pytest.approx(stats["error_after"])
        assert len(mesh.triangles) == stats["triangles_after"]

    counts = [results[r]["triangles_after"] for r in (0.25, 1.0, 4.0)]
    assert counts == sorted(counts, reverse=True), counts
    # Premise: the targets must actually separate, or the ordering above is
    # satisfied trivially by three identical meshes.
    assert counts[0] > 2 * counts[-1], counts
    # And the whole point: a loose target is a large reduction.
    assert results[4.0]["reduction"] > 0.9, results[4.0]["reduction"]


def test_simplify_to_error_leaves_a_mesh_that_already_misses_the_target():
    """Decimating can only move the surface further from the cloud.

    So a target the input mesh already misses has no solution below it, and
    the honest answer is the input back with `target_met` False -- not a
    smaller mesh that quietly misses it by more.
    """
    from gsplat.photogrammetry.mesh_extraction import simplify_mesh_to_error

    points = _sphere_cloud()
    dense = _unit_sphere_mesh(resolution=20)

    mesh, stats = simplify_mesh_to_error(dense, points, max_error=1e-9)
    assert mesh is dense
    assert stats["target_met"] is False
    assert stats["triangles_after"] == stats["triangles_before"]
    assert stats["reduction"] == 0.0
    assert stats["num_probes"] == 0
    # Premise: the input really does miss this target, so nothing above is
    # about an unreachable code path.
    assert stats["error_before"] > 1e-9


def test_simplify_to_error_agrees_whichever_way_the_target_is_given():
    """`error_over_spacing` is just `max_error` in units of the cloud."""
    from gsplat.photogrammetry.mesh_extraction import (
        _point_spacing,
        simplify_mesh_to_error,
    )

    points = _sphere_cloud()
    dense = _unit_sphere_mesh(resolution=30)
    spacing = _point_spacing(points)

    by_ratio, ratio_stats = simplify_mesh_to_error(
        dense, points, error_over_spacing=2.0
    )
    by_error, error_stats = simplify_mesh_to_error(
        dense, points, max_error=2.0 * spacing
    )
    assert ratio_stats["max_error"] == pytest.approx(error_stats["max_error"])
    assert ratio_stats["triangles_after"] == error_stats["triangles_after"]
    np.testing.assert_allclose(
        np.asarray(by_ratio.vertices), np.asarray(by_error.vertices)
    )
    # The ratio route reports the spacing it resolved against; the absolute
    # route has no cloud scale to report and says so rather than inventing one.
    assert ratio_stats["point_spacing"] == pytest.approx(spacing)
    assert error_stats["point_spacing"] is None


def test_simplify_to_error_rejects_ambiguous_or_unmeasurable_input():
    o3d = pytest.importorskip("open3d")

    from gsplat.photogrammetry.mesh_extraction import simplify_mesh_to_error

    mesh = _unit_sphere_mesh(resolution=6)
    points = _sphere_cloud(num_points=500)

    with pytest.raises(ValueError, match="exactly one"):
        simplify_mesh_to_error(mesh, points)
    with pytest.raises(ValueError, match="exactly one"):
        simplify_mesh_to_error(mesh, points, max_error=0.1, error_over_spacing=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        simplify_mesh_to_error(mesh, points, max_error=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        simplify_mesh_to_error(mesh, points, error_over_spacing=-1.0)
    with pytest.raises(ValueError, match="nothing to measure the fit against"):
        simplify_mesh_to_error(mesh, np.zeros((0, 3)), max_error=0.1)
    with pytest.raises(ValueError, match="no triangles"):
        simplify_mesh_to_error(o3d.geometry.TriangleMesh(), points, max_error=0.1)


# ---------------------------------------------------------------------------
# Culling geometry no camera observed
# ---------------------------------------------------------------------------


def _nested_spheres(resolution=10, inner_radius=0.4):
    """An outer shell with a second shell sealed inside it.

    Ground truth that needs no renderer: no camera outside the outer sphere can
    see *any* face of the inner one, and every face of the outer one is seen
    from somewhere. The inner sphere's faces are the last block of triangles,
    which is what makes "was exactly the right set removed?" answerable.
    """
    o3d = pytest.importorskip("open3d")

    outer = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    inner = o3d.geometry.TriangleMesh.create_sphere(
        radius=inner_radius, resolution=resolution
    )
    combined = outer + inner
    combined.compute_vertex_normals()
    return combined, len(outer.triangles), len(inner.triangles)


def test_cull_removes_exactly_the_geometry_no_camera_can_see():
    """Both directions matter, and the false-positive one matters more.

    Leaving unseen geometry behind wastes triangles; culling *observed*
    geometry destroys the asset. This checks the sealed inner shell is entirely
    gone and the outer shell is entirely intact.
    """
    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces

    mesh, num_outer, num_inner = _nested_spheres()
    dataset = _SphereDataset(num_views=16)

    culled, stats = cull_unobserved_faces(mesh, dataset, clean=False)

    # Premise: the inner shell really is invisible from every view, and the
    # outer shell really is visible. Without this the assertions below could
    # pass on a scene where there was nothing to cull.
    assert stats["observation_histogram"][0] == num_inner, stats[
        "observation_histogram"
    ]

    assert stats["num_faces_before"] == num_outer + num_inner
    assert stats["num_culled"] == num_inner
    assert stats["num_faces_after"] == num_outer
    assert stats["fraction_culled"] == pytest.approx(
        num_inner / (num_outer + num_inner)
    )

    # And the survivors are the outer shell specifically, not merely the right
    # *number* of faces: every remaining centroid sits out near radius 1.
    vertices = np.asarray(culled.vertices)
    centroid_radii = np.linalg.norm(vertices[np.asarray(culled.triangles)].mean(1), 1)
    assert centroid_radii.min() > 0.9, centroid_radii.min()


def test_cull_does_not_mutate_the_caller_s_mesh():
    """`remove_triangles_by_mask` is in-place; the input must survive it.

    The pipeline keeps the pre-cull mesh around -- it is what `bake_normal_map`
    takes its detail from -- so quietly emptying it would break the delivery
    path in a way no assertion about the *returned* mesh would catch.
    """
    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces

    mesh, num_outer, num_inner = _nested_spheres(resolution=6)
    before = len(mesh.triangles)

    culled, _ = cull_unobserved_faces(mesh, _SphereDataset(num_views=8), clean=False)

    assert len(mesh.triangles) == before
    assert len(culled.triangles) < before


def test_cull_is_monotone_in_min_views():
    """Demanding more views can only remove more faces."""
    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces

    mesh, _, _ = _nested_spheres(resolution=8)
    dataset = _SphereDataset(num_views=8)

    counts = [
        cull_unobserved_faces(mesh, dataset, min_views=n, clean=False)[1][
            "num_faces_after"
        ]
        for n in (1, 3, 4)
    ]
    assert counts == sorted(counts, reverse=True), counts
    # Premise: the thresholds must actually separate, or this is vacuous.
    # Measured on this scene: 224 faces survive min_views=1, 172 survive 3,
    # and 33 survive 4 -- every outer face is seen by 2 to 4 of the 8 views.
    assert counts[0] > counts[-1], counts

    # Past that the guard takes over rather than returning an empty mesh: no
    # face here is seen by 5 views, so demanding 5 has no solution.
    with pytest.raises(ValueError, match="Every one of the"):
        cull_unobserved_faces(mesh, dataset, min_views=5, clean=False)


def test_cull_refuses_to_empty_the_mesh():
    """Culling *everything* means the mesh and the dataset disagree.

    Wrong poses, wrong scale, or a mesh in another coordinate frame -- not a
    subject with a large unseen back. Returning an empty mesh at the end of a
    long run would hide exactly that.
    """
    o3d = pytest.importorskip("open3d")

    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces

    far = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    far.translate((100.0, 100.0, 100.0))
    far.compute_vertex_normals()

    with pytest.raises(ValueError, match="Every one of the"):
        cull_unobserved_faces(far, _SphereDataset(num_views=6))


def test_cull_rejects_bad_arguments():
    o3d = pytest.importorskip("open3d")

    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces

    mesh = _unit_sphere_mesh(resolution=6)
    dataset = _SphereDataset(num_views=4)
    with pytest.raises(ValueError, match="min_views must be positive"):
        cull_unobserved_faces(mesh, dataset, min_views=0)
    with pytest.raises(ValueError, match="no triangles"):
        cull_unobserved_faces(o3d.geometry.TriangleMesh(), dataset)


def test_visibility_is_not_the_same_question_as_quality():
    """The distinction culling rests on, measured rather than asserted.

    `face_view_quality` is gradient energy over the projection, so a face on a
    *flat* surface scores ~0 however plainly it is in view -- there is no
    detail there to measure. Culling on `quality == 0` would therefore delete
    geometry every camera saw. This pins that the two disagree, and by how
    much, so nobody later "simplifies" the cull to reuse the quality matrix.
    """
    from gsplat.photogrammetry.texturing import face_view_quality, face_visibility

    flat = lambda points: np.full(np.asarray(points).shape, 0.6)  # noqa: E731
    mesh = _unit_sphere_mesh(resolution=8)
    dataset = _SphereDataset(num_views=8, pattern=flat)

    visible = face_visibility(mesh, dataset)
    quality = face_view_quality(mesh, dataset)

    # Every face of this sphere is seen from somewhere, by 8 views around it.
    assert visible.any(axis=1).all()
    # A third of the (face, view) pairs that *are* visible score exactly zero,
    # because the surface they see carries no gradient. Measured: 215 of 653.
    visible_but_zero = int((visible & (quality == 0)).sum())
    assert visible_but_zero > 0.25 * int(visible.sum()), (
        visible_but_zero,
        int(visible.sum()),
    )
    # And the set that matters: faces scoring zero from *every* view, which a
    # quality-based cull would destroy outright despite every camera seeing
    # them. Measured: 12 of 224.
    would_be_lost = int((visible.any(axis=1) & (quality.max(axis=1) == 0)).sum())
    assert would_be_lost > 0, "flat scene should leave some faces scoring zero"


def test_cull_keeps_observed_faces_on_a_flat_untextured_surface():
    """The trap from `test_visibility_is_not_the_same_question_as_quality`,
    pinned where it would actually do damage.

    That test shows the visibility and quality matrices disagree. This one
    shows the *cull* must consult the right one: on a flat scene 12 of these
    224 faces score zero quality from every view despite every camera seeing
    them, so a cull rewritten to reuse the quality matrix silently deletes
    surface off the middle of an observed object.
    """
    from gsplat.photogrammetry.mesh_extraction import cull_unobserved_faces
    from gsplat.photogrammetry.texturing import face_view_quality

    flat = lambda points: np.full(np.asarray(points).shape, 0.6)  # noqa: E731
    mesh = _unit_sphere_mesh(resolution=8)
    dataset = _SphereDataset(num_views=8, pattern=flat)

    # Premise: quality-based culling really would destroy faces here.
    quality = face_view_quality(mesh, dataset)
    at_risk = int((quality.max(axis=1) == 0).sum())
    assert at_risk > 0, "flat scene should leave faces scoring zero everywhere"

    culled, stats = cull_unobserved_faces(mesh, dataset, clean=False)
    assert stats["num_culled"] == 0, (stats, at_risk)
    assert stats["observation_histogram"][0] == 0
    assert len(culled.triangles) == len(mesh.triangles)


# --- Derived reconstruction parameters (Task 3b) -----------------------------
#
# `voxel_size=0.01`, `sdf_trunc=0.04`, `depth_trunc=10.0` and Poisson's
# `radius=0.1` were the last absolute scene-unit constants in the pipeline, on
# a branch whose stated premise is that such a number means nothing on its own.
# These tests pin the property that replaces them: the *same relative* result
# at any scene scale.


def test_derived_tsdf_parameters_scale_with_the_scene():
    """A voxel is the world size of a source pixel at the depth it observes.

    So at 10x the scene scale every derived length must be 10x, exactly. A
    fixed constant is what cannot do this.
    """
    from gsplat.photogrammetry.mesh_extraction import derive_tsdf_parameters

    base = derive_tsdf_parameters(_make_sphere_views(num_views=8))
    scaled = derive_tsdf_parameters(
        _make_sphere_views(num_views=8, radius=10.0, cam_dist=30.0)
    )

    for key in ("voxel_size", "sdf_trunc", "depth_trunc"):
        assert scaled[key] == pytest.approx(
            10.0 * base[key], rel=1e-6
        ), f"{key} did not scale with the scene: {base[key]} -> {scaled[key]}"
    # The truncation is a multiple of the voxel, not an independent constant.
    assert base["sdf_trunc"] == pytest.approx(4.0 * base["voxel_size"])
    # Premise: depth_trunc must not crop the geometry it is meant to keep.
    assert base["depth_trunc"] > base["max_depth"]


def test_derived_tsdf_beats_the_old_constant_away_from_its_scale():
    """The constant was tuned for one scene scale, and only works there.

    Measured: at 10x scale the old 0.01/0.04 default produces an **empty
    mesh**; at 0.1x it produces 4.9x the relative error. The derived value
    gives the same relative fit at every scale.
    """
    from gsplat.photogrammetry.mesh_extraction import _tsdf_fuse
    from gsplat.photogrammetry.metrics import point_to_mesh_distance

    def relative_error(scale, **kwargs):
        views = _make_sphere_views(num_views=8, radius=scale, cam_dist=3.0 * scale)
        mesh = _tsdf_fuse(views, **kwargs)
        if len(mesh.triangles) == 0:
            return None
        directions = np.random.default_rng(0).normal(size=(2000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        truth = directions * scale
        return point_to_mesh_distance(truth, mesh)["mean"] / scale

    fixed = dict(voxel_size=0.01, sdf_trunc=0.04, depth_trunc=10.0)

    # Premise: at the scale the constant was tuned for, the two agree -- so the
    # comparison below is about scale, not about the derivation being better
    # in general.
    assert relative_error(1.0) == pytest.approx(relative_error(1.0, **fixed), rel=0.1)

    # The derived value holds its relative accuracy across a 100x span.
    small, large = relative_error(0.1), relative_error(10.0)
    assert small is not None and large is not None
    assert small == pytest.approx(large, rel=0.2)

    # The constant does not.
    assert (
        relative_error(10.0, **fixed) is None
    ), "the old constant no longer fails at 10x scale -- re-measure the claim"
    assert relative_error(0.1, **fixed) > 3.0 * small


def test_derived_poisson_normal_radius_scales_with_the_cloud():
    """Poisson's `radius=0.1` was the same kind of constant.

    At 10x the scene, a fixed radius sees a tenth of the neighbourhood and the
    surface degrades; the derived radius holds its relative fit.
    """
    from gsplat.photogrammetry.mesh_extraction import extract_mesh_poisson
    from gsplat.photogrammetry.metrics import point_to_mesh_distance

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(6000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def relative_error(scale, **kwargs):
        points = directions * scale
        stats: dict = {}
        mesh = extract_mesh_poisson(points, depth=7, stats_out=stats, **kwargs)
        return point_to_mesh_distance(points, mesh)["mean"] / scale, stats

    derived_small, stats_small = relative_error(1.0)
    derived_large, stats_large = relative_error(10.0)
    assert stats_small["normal_radius_derived"] is True
    assert stats_large["normal_radius"] == pytest.approx(
        10.0 * stats_small["normal_radius"], rel=1e-6
    )
    assert derived_small == pytest.approx(derived_large, rel=0.2)

    # Premise: at 1x the fixed radius is fine, so what follows is about scale.
    fixed_small, _ = relative_error(1.0, normal_radius=0.1)
    assert fixed_small == pytest.approx(derived_small, rel=0.2)

    fixed_large, _ = relative_error(10.0, normal_radius=0.1)
    assert (
        fixed_large > 3.0 * derived_large
    ), "a fixed 0.1 radius no longer degrades at 10x scale -- re-measure"


def test_an_explicit_parameter_does_not_un_derive_the_others():
    """Overriding one TSDF parameter must leave the rest derived.

    Filling all three from the derivation only when *all* are None would make
    `--voxel_size 0.02` silently restore the old 10.0 depth cutoff too.
    """
    from gsplat.photogrammetry.mesh_extraction import (
        _tsdf_fuse,
        derive_tsdf_parameters,
    )

    views = _make_sphere_views(num_views=6)
    derived = derive_tsdf_parameters(views)

    stats: dict = {}
    _tsdf_fuse(views, voxel_size=0.02, stats_out=stats)
    assert stats["voxel_size"] == 0.02
    assert stats["sdf_trunc"] == pytest.approx(derived["sdf_trunc"])
    assert stats["depth_trunc"] == pytest.approx(derived["depth_trunc"])


def test_deriving_tsdf_parameters_needs_some_depth():
    """All-zero depth maps must say so, not divide by nothing."""
    from gsplat.photogrammetry.mesh_extraction import derive_tsdf_parameters

    views = _make_sphere_views(num_views=3)
    for view in views:
        view["depth"] = np.zeros_like(view["depth"])
    with pytest.raises(ValueError, match="valid depth"):
        derive_tsdf_parameters(views)


def test_the_voxel_size_survives_a_speck_of_near_geometry():
    """The median, not the minimum, over pixel footprints.

    The minimum is whatever single pixel grazed the nearest surface, so a few
    stray near-depth samples -- a speck of foreground, a splat artefact in
    front of the camera -- would size the entire volume. At 1/20th the depth
    that is a 20x finer voxel over the whole scene, which is a memory blow-up
    rather than a quality gain.

    Pinning this because it is exactly the kind of choice a later reader
    "simplifies": the minimum looks like the conservative option.
    """
    from gsplat.photogrammetry.mesh_extraction import derive_tsdf_parameters

    clean = _make_sphere_views(num_views=6)
    contaminated = _make_sphere_views(num_views=6)
    for view in contaminated:
        depth = view["depth"]
        valid = np.nonzero(depth > 0)
        # 20 pixels out of thousands, twenty times closer than the surface.
        picks = (valid[0][:20], valid[1][:20])
        depth[picks] = depth[picks] / 20.0

    baseline = derive_tsdf_parameters(clean)["voxel_size"]
    spoiled = derive_tsdf_parameters(contaminated)["voxel_size"]

    # Premise: the contamination really is there and really is extreme, or the
    # robustness below is vacuous.
    clean_min = min(float(v["depth"][v["depth"] > 0].min()) for v in clean)
    dirty_min = min(float(v["depth"][v["depth"] > 0].min()) for v in contaminated)
    assert dirty_min < clean_min / 10.0

    assert spoiled == pytest.approx(baseline, rel=0.02), (
        f"20 stray near pixels moved the voxel size {baseline:.6g} -> "
        f"{spoiled:.6g}; the estimator is not robust to them"
    )
