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

    def __init__(self, num_views=24, radius=1.0, cam_dist=3.5, width=192, height=192):
        torch = pytest.importorskip("torch")
        self.width, self.height = width, height
        focal = 260.0
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
            image[hit] = _surface_pattern(hit_points[hit])
            image = (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)

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
