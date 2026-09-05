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
"""Baking a mesh's surface appearance from the training images.

Everything that samples the training views or writes into a UV atlas lives
here; :mod:`gsplat.photogrammetry.mesh_extraction` builds the surface, this
module dresses it.

- :func:`bake_texture` colors a mesh's vertices, with occlusion-aware,
  view-angle-weighted blending across views, and optional iterative
  sigma-clipped robust fusion for captures whose views disagree.
- :func:`bake_texture_atlas` bakes that same color signal into a UV-unwrapped
  texture atlas, so the result carries detail beyond the mesh's vertex density
  and loads with its texture in standard DCC tools and game engines.
- :func:`bake_texture_atlas_view_selected` bakes that atlas from *one chosen
  view per face* instead of a blend (:func:`face_view_quality` +
  :func:`select_views_mrf`), which keeps the high-frequency detail blending
  destroys -- at the cost of pointwise accuracy, so it is opt-in.
- :func:`bake_normal_map` records a dense mesh's normals onto a decimated
  mesh's atlas, and :func:`bake_ambient_occlusion` records how much of the sky
  each point can see. With the albedo atlas these are the standard
  photogrammetry map set, and they all share one UV layout.
- :func:`bake_mesh_texture` is the dispatcher the CLIs use.

Requires the optional ``open3d`` dependency: ``pip install gsplat[mesh]``.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._open3d import _require_open3d


def _bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional pixel coordinates ``uv`` (x, y).

    Every bake in this module reads its colours through here. Nearest-neighbour
    would be simpler, and is what this did first, but a surface point almost
    never lands on a pixel centre: rounding to one throws away up to half a
    pixel of the projection's accuracy, and it does so *differently in each
    view*, so the views disagree about a point's colour by more than they need
    to. Measured on the analytic sphere over 16 views: mean per-sample error
    against ground truth 0.0707 -> 0.0572, and mean pairwise disagreement
    between two views of one vertex 0.263 -> 0.217.

    That second number is the one that matters most. It is the noise floor the
    seam-levelling solve has to see past (:func:`level_seams`), and it is what
    robust fusion spends its passes separating from real outliers.

    ``uv`` is in pixel units with integer coordinates at pixel *corners* (the
    convention the projection produces), so half a pixel comes off before
    interpolating. Sampling is clamped at the border rather than wrapped -- a
    sample past the edge must take the edge pixel, never the opposite side of
    the frame. The clamp on ``x``/``y`` already forces the fractional weight to
    zero there, so the ``minimum`` on ``x1``/``y1`` never changes a result; it
    is kept so that the intent survives a later change to the clip.
    """
    height, width = image.shape[:2]
    x = np.clip(uv[:, 0] - 0.5, 0.0, width - 1.0)
    y = np.clip(uv[:, 1] - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _view_samples(scene, o3d, dataset, points, normals, max_views, chunk_size):
    """Yield ``(indices, colors, weights, view_index)`` per visible (point, view).

    One pass over the dataset: project every point into each camera, drop the
    out-of-frame ones, ray-cast away the occluded ones, and weight what is left
    by view-direction/normal alignment and inverse distance. Factored out so a
    robust estimator can run the same sampling twice without duplicating it,
    and so view selection can reuse exactly this visibility test rather than
    reimplementing the subtle part.
    """
    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    for i in range(num_views):
        data = dataset[i]
        camtoworld = data["camtoworld"].numpy()
        K = data["K"].numpy()
        image = data["image"].numpy() / 255.0  # (H, W, 3) in [0, 1]
        height, width = image.shape[:2]
        cam_pos = camtoworld[:3, 3]
        viewmat = np.linalg.inv(camtoworld)

        Xc = (viewmat[:3, :3] @ points.T + viewmat[:3, 3:4]).T  # (P, 3)
        in_front = Xc[:, 2] > 1e-4
        uvw = (K @ Xc.T).T
        uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        candidates = np.nonzero(in_front & in_bounds)[0]
        if candidates.size == 0:
            continue

        for start in range(0, candidates.size, chunk_size):
            chunk = candidates[start : start + chunk_size]
            dirs = points[chunk] - cam_pos[None, :]
            dists = np.linalg.norm(dirs, axis=1)
            dirs_n = dirs / np.clip(dists, 1e-8, None)[:, None]
            rays = np.concatenate(
                [np.repeat(cam_pos[None, :], len(chunk), axis=0), dirs_n], axis=1
            ).astype(np.float32)
            hit_t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            # Keep only points whose nearest ray hit is (approximately)
            # themselves, i.e. not occluded by other geometry in this view.
            visible = np.abs(hit_t - dists) < (1e-2 * dists + 1e-3)
            chunk = chunk[visible]
            if chunk.size == 0:
                continue
            dirs_n = dirs_n[visible]
            dists = dists[visible]

            sampled = _bilinear(image, uv[chunk])  # (K, 3)

            cos_weight = np.clip((normals[chunk] * -dirs_n).sum(-1), 0.0, 1.0)
            dist_weight = 1.0 / np.clip(dists, 1e-3, None)
            weight = cos_weight * dist_weight + 1e-6
            # `chunk` indexes each point at most once per view, so callers can
            # use plain in-place adds rather than the much slower np.add.at.
            yield chunk, sampled, weight, i


def _bake_points_from_views(
    mesh,
    dataset,
    points: np.ndarray,
    normals: np.ndarray,
    max_views: Optional[int] = None,
    chunk_size: int = 1 << 20,
    outlier_sigma: Optional[float] = None,
    outlier_iterations: int = 3,
    min_views_for_clipping: int = 3,
):
    """Accumulate occlusion-aware, view-weighted colors for surface points.

    Shared by :func:`bake_texture` (which bakes at mesh vertices) and
    :func:`bake_texture_atlas` (which bakes at texel positions), so both
    produce the same color signal and only differ in where they sample it.

    For each point, projects into every (or up to ``max_views``) training
    camera, discards out-of-frame projections and -- via ray casting against
    ``mesh`` itself -- occluded ones, and accumulates the remaining observed
    pixel colors weighted by view-direction/surface-normal alignment and
    inverse distance.

    Args:
        mesh: The ``open3d.geometry.TriangleMesh`` to ray-cast against for
            occlusion. ``points`` are expected to lie on its surface.
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        points: (P, 3) surface points to bake.
        normals: (P, 3) unit-length outward normals, one per point.
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).
        chunk_size: Max points ray-cast at once, bounding peak memory when
            baking the millions of texels a large atlas can contain.
        outlier_sigma: If set, discard observations further than this many
            standard deviations from the point's own weighted mean color, then
            re-average what survives. A plain mean is dragged by *any*
            disagreement between views -- a pedestrian, a specular highlight, a
            slightly misregistered camera -- and blends it into the texture as
            ghosting.
        outlier_iterations: How many times to re-estimate and re-clip. One pass
            is weak, because the contaminated samples inflate the very spread
            they are measured against: with a quarter of views wrong, the
            outliers sit right at a 1.5-sigma threshold and mostly survive.
            Each further round re-centres on the survivors, so the spread
            shrinks and the outliers separate cleanly. Each round costs one
            more sampling pass over the dataset.

            **What this cannot do:** it only recovers points whose bad
            observations are a *minority*. A surface hidden behind a
            pedestrian in most of the views that see it has no majority to
            fall back on, and clipping will happily converge on the
            pedestrian. That case is what ``--mask_dir`` transient masking is
            for; robust fusion is the complement, cleaning up the residual
            disagreement -- specular highlights, slight misregistration, mask
            leakage -- that masking does not catch.
        min_views_for_clipping: Points observed by fewer views than this keep
            their plain mean. With two or three samples a "spread" is noise,
            and clipping against it would reject good data.

    Returns:
        ``(color_accum, weight_accum)``: a (P, 3) sum of weighted colors and
        the (P,) sum of weights. Points with ``weight_accum == 0`` were never
        observed.
    """
    o3d = _require_open3d()

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    num_points = points.shape[0]
    color_accum = np.zeros((num_points, 3), dtype=np.float64)
    weight_accum = np.zeros((num_points,), dtype=np.float64)
    if num_points == 0:
        return color_accum, weight_accum

    def stream():
        return _view_samples(
            scene, o3d, dataset, points, normals, max_views, chunk_size
        )

    square_accum = np.zeros((num_points, 3), dtype=np.float64)
    view_counts = np.zeros((num_points,), dtype=np.int64)
    for chunk, sampled, weight, _ in stream():
        color_accum[chunk] += sampled * weight[:, None]
        weight_accum[chunk] += weight
        if outlier_sigma is not None:
            square_accum[chunk] += (sampled**2) * weight[:, None]
            view_counts[chunk] += 1

    if outlier_sigma is None:
        return color_accum, weight_accum

    seen = weight_accum > 0
    sparse = view_counts < min_views_for_clipping

    def estimate(sum_color, sum_weight, sum_square):
        """Weighted mean and RGB-distance spread from running sums."""
        ok = sum_weight > 0
        mean = np.zeros_like(sum_color)
        mean[ok] = sum_color[ok] / sum_weight[ok, None]
        variance = np.zeros_like(sum_color)
        variance[ok] = np.clip(
            sum_square[ok] / sum_weight[ok, None] - mean[ok] ** 2, 0.0, None
        )
        # Scale the threshold the way the residual is measured (an RGB
        # distance), floored at one 8-bit step so a uniformly-colored surface
        # doesn't reject its own faultless samples as "outliers".
        return mean, np.maximum(np.sqrt(variance.sum(axis=1)), 1.0 / 255.0)

    mean, spread = estimate(color_accum, weight_accum, square_accum)
    clipped_color, clipped_weight = color_accum, weight_accum

    for _ in range(max(outlier_iterations, 1)):
        round_color = np.zeros_like(color_accum)
        round_weight = np.zeros_like(weight_accum)
        round_square = np.zeros_like(square_accum)
        for chunk, sampled, weight, _ in stream():
            residual = np.linalg.norm(sampled - mean[chunk], axis=1)
            keep = sparse[chunk] | (residual <= outlier_sigma * spread[chunk])
            if not keep.any():
                continue
            kept = chunk[keep]
            kept_color = sampled[keep]
            kept_weight = weight[keep]
            round_color[kept] += kept_color * kept_weight[:, None]
            round_weight[kept] += kept_weight
            round_square[kept] += (kept_color**2) * kept_weight[:, None]

        # A point whose every observation was rejected keeps what it had: an
        # unshaded hole would be worse than a possibly-contaminated color.
        emptied = (round_weight <= 0) & seen
        round_color[emptied] = clipped_color[emptied]
        round_weight[emptied] = clipped_weight[emptied]
        round_square[emptied] = square_accum[emptied]

        clipped_color, clipped_weight = round_color, round_weight
        mean, spread = estimate(round_color, round_weight, round_square)

    return clipped_color, clipped_weight


def bake_texture(
    mesh,
    dataset,
    max_views: Optional[int] = None,
    outlier_sigma: Optional[float] = None,
    outlier_iterations: int = 3,
):
    """Bake per-vertex colors onto ``mesh`` from ``dataset``'s training images.

    Colors each mesh vertex by the occlusion-aware, view-weighted blend
    described in :func:`_bake_points_from_views`.

    This produces per-vertex colors, whose effective resolution is the mesh's
    own vertex density. For image-resolution detail independent of tessellation
    -- and for a mesh that loads with its texture in standard DCC tools and
    game engines -- use :func:`bake_texture_atlas` instead.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh`` (e.g. from
            :func:`extract_mesh_tsdf` or :func:`extract_mesh_poisson`).
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).
        outlier_sigma: Robust fusion -- see :func:`_bake_points_from_views`.

    Returns:
        ``mesh``, with ``vertex_colors`` set in place (and returned for
        chaining).
    """
    o3d = _require_open3d()

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    color_accum, weight_accum = _bake_points_from_views(
        mesh,
        dataset,
        vertices,
        vertex_normals,
        max_views=max_views,
        outlier_sigma=outlier_sigma,
        outlier_iterations=outlier_iterations,
    )

    has_color = weight_accum > 0
    vertex_colors = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.zeros((vertices.shape[0], 3))
    )
    vertex_colors[has_color] = color_accum[has_color] / weight_accum[has_color, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors, 0.0, 1.0))
    return mesh


def _fill_texture_holes(texture: np.ndarray, filled: np.ndarray, iterations: int):
    """Pad baked texels outward into unfilled ones by nearest-neighbor growth.

    Two kinds of texel end up unfilled: those outside every UV island, and
    those inside an island that no camera ever observed. Renderers sample an
    atlas bilinearly (and build mipmaps from it), so both kinds bleed the fill
    color across every UV seam and into every unobserved patch. Growing the
    baked colors a few texels outward removes that artifact.

    Args:
        texture: (S, S, 3) float texture, unfilled texels at 0.
        filled: (S, S) bool mask of texels that carry a baked color.
        iterations: How many texels to grow outward.

    Returns:
        A new (S, S, 3) float texture with the padding applied.
    """
    texture = texture.copy()
    filled = filled.copy()
    for _ in range(max(iterations, 0)):
        if filled.all():
            break
        neighbor_sum = np.zeros_like(texture)
        neighbor_count = np.zeros(filled.shape, dtype=np.float64)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted_tex = np.roll(texture, shift, axis=axis)
            shifted_filled = np.roll(filled, shift, axis=axis)
            # np.roll wraps around; blank the wrapped-in edge so colors can't
            # leak from one side of the atlas to the other.
            edge = 0 if shift > 0 else -1
            if axis == 0:
                shifted_tex[edge, :] = 0.0
                shifted_filled[edge, :] = False
            else:
                shifted_tex[:, edge] = 0.0
                shifted_filled[:, edge] = False
            neighbor_sum += shifted_tex * shifted_filled[..., None]
            neighbor_count += shifted_filled
        grow = (~filled) & (neighbor_count > 0)
        if not grow.any():
            break
        texture[grow] = neighbor_sum[grow] / neighbor_count[grow][:, None]
        filled |= grow
    return texture


@dataclass
class _AtlasTexels:
    """The per-texel surface frame of a UV-unwrapped mesh.

    ``rows``/``cols`` index the texels the rasterizer covered; ``positions``,
    ``normals`` and (optionally) ``tangents`` are the interpolated surface
    frame at each of those texels, in the same order.
    """

    triangle_uvs: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    tangents: Optional[np.ndarray] = None


def _vertex_tangents(vertices, vertex_normals, triangles, corner_uvs):
    """Per-vertex tangents from the UV parameterization (Lengyel's method).

    Accumulates each triangle's UV-aligned tangent onto its vertices, then
    orthogonalizes against the vertex normal. Vertices on a UV seam belong to
    charts with different orientations, so they get a blended tangent -- the
    standard artifact of not splitting vertices per chart, confined to the
    seam texels the dilation pass already covers.
    """
    tangents = np.zeros_like(vertices)
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    duv1 = corner_uvs[:, 1] - corner_uvs[:, 0]
    duv2 = corner_uvs[:, 2] - corner_uvs[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    # A degenerate UV triangle carries no direction; skip it rather than
    # dividing by ~0 and poisoning its vertices' accumulated tangent.
    usable = np.abs(det) > 1e-12
    per_face = np.zeros_like(edge1)
    per_face[usable] = (
        edge1[usable] * duv2[usable, 1:2] - edge2[usable] * duv1[usable, 1:2]
    ) / det[usable, None]
    for corner in range(3):
        np.add.at(tangents, triangles[:, corner], per_face)

    tangents -= vertex_normals * (vertex_normals * tangents).sum(-1, keepdims=True)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    # A vertex whose tangents cancelled out gets an arbitrary perpendicular
    # rather than a zero vector, so the tangent frame stays well-defined.
    degenerate = lengths[:, 0] < 1e-12
    if degenerate.any():
        fallback = np.tile(np.array([1.0, 0.0, 0.0]), (int(degenerate.sum()), 1))
        parallel = np.abs(vertex_normals[degenerate][:, 0]) > 0.9
        fallback[parallel] = np.array([0.0, 1.0, 0.0])
        normals_d = vertex_normals[degenerate]
        fallback -= normals_d * (normals_d * fallback).sum(-1, keepdims=True)
        tangents[degenerate] = fallback
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.clip(lengths, 1e-12, None)


def _unwrap_and_rasterize(
    mesh,
    texture_size: int,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    with_tangents: bool = False,
    reuse_uvs: bool = True,
) -> "_AtlasTexels":
    """UV-unwrap ``mesh`` and recover each texel's surface frame.

    Shared by :func:`bake_texture_atlas` and :func:`bake_normal_map`, which
    differ only in what they bake into the atlas, never in how the atlas is
    built or which texels it covers.

    **open3d's ``compute_uvatlas`` is not deterministic** -- unwrapping the
    same mesh twice yields different UV layouts. So when ``mesh`` already
    carries ``triangle_uvs``, they are reused rather than recomputed
    (``reuse_uvs=False`` forces a fresh unwrap). Without that, baking an
    albedo atlas and then a normal map for one mesh would produce two maps
    with *different* UV layouts, and the second would be applied through the
    first's coordinates -- a silently broken asset.

    It also requires a manifold mesh and **segfaults** rather than raising on
    non-manifold input, so that is checked up front: a crash in the middle of
    a long pipeline run would otherwise take the whole run down.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``texture_size`` is not
            positive, or ``mesh`` is non-manifold.
    """
    o3d = _require_open3d()

    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}.")
    if len(mesh.triangles) == 0:
        raise ValueError("Cannot UV-unwrap a mesh with no triangles.")
    # Boundary edges are fine (and normal for TSDF output) -- only edges shared
    # by more than two triangles, or vertices joining disconnected fans, break
    # the unwrapper.
    if not mesh.is_edge_manifold(allow_boundary_edges=True):
        raise ValueError(
            "Cannot UV-unwrap a mesh with non-manifold edges (open3d's "
            "compute_uvatlas requires a manifold mesh and crashes on this "
            "input). Run `mesh.remove_non_manifold_edges()` first, or use "
            "bake_texture() for per-vertex colors instead."
        )
    if not mesh.is_vertex_manifold():
        raise ValueError(
            "Cannot UV-unwrap a mesh with non-manifold vertices (open3d's "
            "compute_uvatlas requires a manifold mesh and crashes on this "
            "input). Use bake_texture() for per-vertex colors instead."
        )

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    num_triangles = len(mesh.triangles)
    existing_uvs = np.asarray(mesh.triangle_uvs)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    if reuse_uvs and existing_uvs.shape[0] == 3 * num_triangles:
        t_mesh.triangle["texture_uvs"] = o3d.core.Tensor(
            existing_uvs.reshape(num_triangles, 3, 2).astype(np.float32)
        )
    else:
        t_mesh.compute_uvatlas(
            size=texture_size if unwrap_size is None else unwrap_size,
            gutter=gutter,
            max_stretch=max_stretch,
        )

    # UV coordinates are per triangle corner, so unwrapping leaves the mesh's
    # vertices and triangles untouched -- the atlas can be attached straight
    # back onto the input mesh by the caller.
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    corner_uvs = t_mesh.triangle["texture_uvs"].numpy()  # (F, 3, 2)

    attrs = {"_bake_xyz", "_bake_normal", "_bake_coverage"}
    t_mesh.vertex["_bake_xyz"] = o3d.core.Tensor(vertices.astype(np.float32))
    t_mesh.vertex["_bake_normal"] = o3d.core.Tensor(vertex_normals.astype(np.float32))
    # A constant-1 attribute bakes to exactly the texels the rasterizer covered,
    # which is what marks a texel as being on the surface. Testing the baked
    # position against 0 instead would misclassify a surface passing through
    # the world origin.
    t_mesh.vertex["_bake_coverage"] = o3d.core.Tensor(
        np.ones((vertices.shape[0], 1), dtype=np.float32)
    )
    if with_tangents:
        tangents = _vertex_tangents(
            vertices, vertex_normals, np.asarray(mesh.triangles), corner_uvs
        )
        t_mesh.vertex["_bake_tangent"] = o3d.core.Tensor(tangents.astype(np.float32))
        attrs.add("_bake_tangent")

    baked = t_mesh.bake_vertex_attr_textures(
        texture_size, attrs, margin=margin, fill=0.0, update_material=False
    )
    covered = baked["_bake_coverage"].numpy()[..., 0] > 0.5
    rows, cols = np.nonzero(covered)

    positions = baked["_bake_xyz"].numpy()[rows, cols].astype(np.float64)
    normals = baked["_bake_normal"].numpy()[rows, cols].astype(np.float64)
    # Barycentric interpolation of unit vertex normals doesn't preserve length;
    # renormalize so downstream dot products stay comparable across texels.
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)

    texel_tangents = None
    if with_tangents:
        texel_tangents = baked["_bake_tangent"].numpy()[rows, cols].astype(np.float64)
        # Re-orthogonalize after interpolation, for the same reason.
        texel_tangents -= normals * (normals * texel_tangents).sum(-1, keepdims=True)
        texel_tangents /= np.clip(
            np.linalg.norm(texel_tangents, axis=1, keepdims=True), 1e-8, None
        )

    return _AtlasTexels(
        triangle_uvs=corner_uvs.reshape(-1, 2).astype(np.float64),
        rows=rows,
        cols=cols,
        positions=positions,
        normals=normals,
        tangents=texel_tangents,
    )


def bake_texture_atlas(
    mesh,
    dataset,
    texture_size: int = 2048,
    max_views: Optional[int] = None,
    outlier_sigma: Optional[float] = None,
    outlier_iterations: int = 3,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
):
    """Bake a UV-unwrapped texture atlas onto ``mesh`` from the training images.

    Unlike :func:`bake_texture`, whose per-vertex colors can only carry as
    much detail as the mesh is tessellated for, this UV-unwraps ``mesh`` and
    bakes one color per *texel*, so texture detail is independent of vertex
    density -- and the result is what standard DCC tools and game engines
    expect a textured mesh to look like.

    The pipeline is: UV-unwrap via open3d's ``compute_uvatlas``; rasterize the
    mesh's per-vertex positions and normals into the atlas via
    ``bake_vertex_attr_textures`` to recover each texel's 3D surface point;
    color those points with the same occlusion-aware, view-weighted blend
    :func:`bake_texture` uses (see :func:`_bake_points_from_views`); then pad
    the result outward across UV seams (see :func:`_fill_texture_holes`).

    The returned mesh carries ``triangle_uvs`` and ``textures``, so writing it
    with ``open3d.io.write_triangle_mesh("mesh.obj", mesh)`` emits the ``.obj``,
    its ``.mtl``, and the texture ``.png`` together. Note that ``.ply`` cannot
    carry a UV atlas -- write ``.obj`` on this path.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``. Must be manifold (see
            below).
        dataset: An ``examples.datasets.colmap.Dataset``-like object yielding
            dicts with ``"camtoworld"`` (4, 4), ``"K"`` (3, 3), and ``"image"``
            ((H, W, 3), values in [0, 255]).
        texture_size: Width/height of the (square) texture, in texels.
        max_views: If given, only the first ``max_views`` dataset images are
            used (for speed on large datasets).
        outlier_sigma: Robust fusion -- see :func:`_bake_points_from_views`.
            Worth enabling on any real capture, where views disagree.
        unwrap_size: Texture size assumed while unwrapping, which sets the
            scale ``gutter`` is measured against. Defaults to ``texture_size``.
        gutter: Space around each UV island, in texels, passed to
            ``compute_uvatlas``.
        margin: Extra texels rasterized around each UV island by
            ``bake_vertex_attr_textures``.
        max_stretch: Per-chart stretch tolerance for ``compute_uvatlas``.
            Lower values cut the mesh into more, less distorted charts.
        dilation: How many texels to grow baked colors outward across seams
            and unobserved patches.

    Returns:
        ``(mesh, texture)``: ``mesh`` with ``triangle_uvs``/``textures`` set in
        place, and the (``texture_size``, ``texture_size``, 3) ``uint8`` texture
        as a numpy array.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``texture_size`` is not
            positive, or ``mesh`` is non-manifold. open3d's ``compute_uvatlas``
            requires a manifold mesh and *segfaults* rather than raising on
            non-manifold input, so this is checked up front -- a crash in the
            middle of a long pipeline run would otherwise take the whole run
            down. :func:`extract_mesh_tsdf`/:func:`extract_mesh_poisson` output
            has already been through ``remove_non_manifold_edges``; fall back
            to :func:`bake_texture` for a mesh that still fails the check.
    """
    o3d = _require_open3d()

    atlas = _unwrap_and_rasterize(
        mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
    )
    rows, cols = atlas.rows, atlas.cols

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if rows.size > 0:
        color_accum, weight_accum = _bake_points_from_views(
            mesh,
            dataset,
            atlas.positions,
            atlas.normals,
            max_views=max_views,
            outlier_sigma=outlier_sigma,
            outlier_iterations=outlier_iterations,
        )
        observed = weight_accum > 0
        texture[rows[observed], cols[observed]] = (
            color_accum[observed] / weight_accum[observed, None]
        )
        filled[rows[observed], cols[observed]] = True

    texture = _fill_texture_holes(texture, filled, dilation)
    texture = (np.clip(texture, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    mesh.textures = [o3d.geometry.Image(texture)]
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(mesh.triangles), dtype=np.int32)
    )
    return mesh, texture


def bake_mesh_texture(
    mesh,
    dataset,
    mode: str = "vertex",
    texture_size: int = 2048,
    max_views: Optional[int] = None,
    outlier_sigma: Optional[float] = None,
    outlier_iterations: int = 3,
    allow_atlas_fallback: bool = True,
    view_selection: bool = False,
    mrf_smoothness: float = 1.0,
    stats_out: Optional[dict] = None,
):
    """Bake texture onto ``mesh`` in either supported form.

    The one entry point the CLIs use, so the choice between per-vertex colors
    and a UV atlas -- and what happens when a mesh can't be unwrapped -- is
    decided in one place rather than in each script.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object (see
            :func:`bake_texture`).
        mode: ``"vertex"`` for :func:`bake_texture`, ``"atlas"`` for
            :func:`bake_texture_atlas`.
        texture_size: Atlas width/height in texels (``"atlas"`` mode only).
        max_views: If given, only the first ``max_views`` dataset images are
            used.
        outlier_sigma: Robust fusion -- see :func:`_bake_points_from_views`.
        allow_atlas_fallback: In ``"atlas"`` mode, whether a mesh that can't be
            UV-unwrapped falls back to per-vertex colors (with a warning)
            instead of raising. Defaults to ``True``: this runs at the end of a
            long training run or pipeline, where losing the mesh entirely is
            the worse outcome.
        view_selection: In ``"atlas"`` mode, texture each face from a single
            chosen view rather than blending -- see
            :func:`bake_texture_atlas_view_selected` for the tradeoff this
            makes. Ignored in ``"vertex"`` mode, which has no faces to label.
        mrf_smoothness: Seam penalty, for ``view_selection``.
        stats_out: If given, a dict updated in place with the view-selection
            stats. An out-parameter rather than a third return value because
            several callers unpack this function's ``(mesh, texture)`` pair,
            and a selection that fell back to blending has no stats to report.

    Returns:
        ``(mesh, texture)``. ``texture`` is the ``uint8`` atlas in ``"atlas"``
        mode, and ``None`` for per-vertex colors -- which is also what a
        fallback returns, so callers can use it to decide whether to write
        ``.obj`` (a UV atlas needs one) or ``.ply``.
    """
    if mode == "vertex":
        return (
            bake_texture(
                mesh,
                dataset,
                max_views=max_views,
                outlier_sigma=outlier_sigma,
                outlier_iterations=outlier_iterations,
            ),
            None,
        )
    if mode != "atlas":
        raise ValueError(
            f"Unknown texture mode {mode!r}, expected 'vertex' or 'atlas'."
        )

    try:
        if view_selection:
            mesh, texture, stats = bake_texture_atlas_view_selected(
                mesh,
                dataset,
                texture_size=texture_size,
                max_views=max_views,
                mrf_smoothness=mrf_smoothness,
                outlier_sigma=outlier_sigma,
                outlier_iterations=outlier_iterations,
            )
            if stats_out is not None:
                stats_out.update(stats)
            return mesh, texture
        return bake_texture_atlas(
            mesh,
            dataset,
            texture_size=texture_size,
            max_views=max_views,
            outlier_sigma=outlier_sigma,
            outlier_iterations=outlier_iterations,
        )
    except ValueError as e:
        if not allow_atlas_fallback:
            raise
        warnings.warn(
            f"UV-atlas texture baking failed ({e}); falling back to per-vertex "
            "colors. The mesh is still written, but without a texture atlas.",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            bake_texture(
                mesh,
                dataset,
                max_views=max_views,
                outlier_sigma=outlier_sigma,
                outlier_iterations=outlier_iterations,
            ),
            None,
        )


def bake_normal_map(
    high_mesh,
    low_mesh,
    texture_size: int = 2048,
    space: str = "tangent",
    cage: Optional[float] = None,
    max_distance: Optional[float] = None,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
    bits: int = 8,
):
    """Bake ``high_mesh``'s surface normals onto ``low_mesh``'s UV atlas.

    The other half of the standard photogrammetry delivery path: after
    :func:`simplify_mesh` cuts a dense extraction down to a workable triangle
    budget, this records the detail that was removed as a normal map, so the
    low-poly mesh *shades* like the dense one. It is what makes decimation
    nearly free visually, and it is why production scans ship as a light mesh
    plus texture maps rather than as raw multi-million-triangle geometry.

    For each texel of ``low_mesh``'s atlas, a ray is cast from just outside the
    low surface, inward along its own normal, onto ``high_mesh``; the high
    mesh's normal at the hit point (barycentrically interpolated) is what gets
    stored. Texels whose ray misses -- and texels in the atlas margin, which
    lie off the surface by construction -- fall back to the low mesh's own
    normal, i.e. "no change from the base geometry".

    Args:
        high_mesh: The dense mesh to take detail from.
        low_mesh: The decimated mesh to bake onto. Must be manifold; it gains
            ``triangle_uvs`` in place, matching the returned map.
        texture_size: Width/height of the (square) normal map, in texels.
        space: ``"tangent"`` stores normals relative to each texel's surface
            frame -- the portable choice, and what engines expect, since it
            stays valid if the mesh is transformed. ``"object"`` stores them
            in mesh space: simpler and immune to UV-seam tangent artifacts,
            and a common choice for static scanned assets.
        cage: How far outside the low surface each ray starts, in scene units.
            It must clear the gap between the two meshes; defaults to 2% of
            ``low_mesh``'s bounding-box diagonal.
        max_distance: Reject hits farther than this from the ray origin.
            Defaults to twice ``cage``, so a ray that punches through a gap
            and hits unrelated geometry behind it is discarded.
        bits: Bit depth per channel, 8 (default) or 16. The encoding
            ``0.5 + 0.5 * n`` spends the whole range on [-1, 1], so 8 bits
            resolve a normal deviation no finer than ``2/255 ~ 0.0078`` --
            about 0.45 degrees of tilt, and a hard floor on what the map can
            carry however dense ``high_mesh`` is. At 16 bits that floor drops
            to ``3.1e-5``. Worth reaching for when the low mesh is already
            close to the high one (a light decimation, or a mesh whose detail
            is genuinely fine), which is exactly where an 8-bit map stops
            recovering anything and starts adding quantization noise. The cost
            is a file twice the size, and not every downstream tool reads
            16-bit PNGs.
        unwrap_size, gutter, margin, max_stretch, dilation: As in
            :func:`bake_texture_atlas`.

    Returns:
        ``(low_mesh, normal_map, stats)``:

        - ``low_mesh`` with ``triangle_uvs`` set, matching the returned map.
        - ``normal_map``, a (``texture_size``, ``texture_size``, 3) array --
          ``uint8``, or ``uint16`` when ``bits=16`` -- encoded the usual way
          (``0.5 + 0.5 * n``, so a flat texel is ~(128, 128, 255) in tangent
          space at 8 bits).
        - ``stats``: ``num_texels``, ``num_hits``, ``hit_fraction`` and the
          ``cage``/``max_distance`` actually used. A low ``hit_fraction`` means
          most texels fell back to the base normal and the map is doing
          nothing -- almost always a ``cage`` too small to span the gap
          between the two meshes. Worth checking before shipping the asset,
          which is why it is returned rather than left to be eyeballed.

    Raises:
        ValueError: If ``space`` is not ``"tangent"``/``"object"``, if
            ``bits`` is not 8 or 16, or
            ``low_mesh`` cannot be unwrapped (see
            :func:`bake_texture_atlas`), or ``high_mesh`` has no triangles.
    """
    o3d = _require_open3d()

    if space not in ("tangent", "object"):
        raise ValueError(
            f"Unknown normal-map space {space!r}, expected 'tangent' or 'object'."
        )
    if bits not in (8, 16):
        raise ValueError(f"Normal-map bits must be 8 or 16, got {bits}.")
    if len(high_mesh.triangles) == 0:
        raise ValueError("Cannot bake normals from a high mesh with no triangles.")

    if not high_mesh.has_vertex_normals():
        high_mesh.compute_vertex_normals()

    atlas = _unwrap_and_rasterize(
        low_mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
        with_tangents=(space == "tangent"),
    )

    if cage is None:
        extent = np.asarray(low_mesh.get_max_bound()) - np.asarray(
            low_mesh.get_min_bound()
        )
        cage = 0.02 * float(np.linalg.norm(extent))
        cage = max(cage, 1e-6)
    if max_distance is None:
        max_distance = 2.0 * cage

    positions, normals = atlas.positions, atlas.normals
    # Default to the low mesh's own normal: for a texel with no hit that is
    # exactly right -- "this surface is already as detailed as the base mesh".
    detailed = normals.copy()
    num_hits = 0

    if positions.shape[0] > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(high_mesh))
        origins = positions + normals * cage
        rays = np.concatenate([origins, -normals], axis=1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays))
        hit_t = result["t_hit"].numpy()
        primitives = result["primitive_ids"].numpy()
        bary = result["primitive_uvs"].numpy()
        hit = np.isfinite(hit_t) & (hit_t <= max_distance)

        num_hits = int(hit.sum())
        if hit.any():
            high_triangles = np.asarray(high_mesh.triangles)[primitives[hit]]
            corner_normals = np.asarray(high_mesh.vertex_normals)[high_triangles]
            u, v = bary[hit, 0], bary[hit, 1]
            # open3d's primitive_uvs weight the corners (1-u-v, u, v) --
            # verified by reconstructing hit positions from them.
            weights = np.stack([1.0 - u - v, u, v], axis=-1)[..., None]
            detailed[hit] = (corner_normals * weights).sum(axis=1)
            detailed /= np.clip(
                np.linalg.norm(detailed, axis=1, keepdims=True), 1e-12, None
            )

    if space == "tangent":
        tangent = atlas.tangents
        bitangent = np.cross(normals, tangent)
        encoded = np.stack(
            [
                (detailed * tangent).sum(-1),
                (detailed * bitangent).sum(-1),
                (detailed * normals).sum(-1),
            ],
            axis=-1,
        )
    else:
        encoded = detailed

    normal_map = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if atlas.rows.size > 0:
        normal_map[atlas.rows, atlas.cols] = 0.5 + 0.5 * encoded
        filled[atlas.rows, atlas.cols] = True
    # Unfilled texels would otherwise decode to a zero-length normal; the
    # dilation pass spreads real values across seams the same way the color
    # atlas does.
    normal_map = _fill_texture_holes(normal_map, filled, dilation)
    if not filled.all():
        # Anything still untouched gets "flat": +Z in tangent space, and the
        # neutral 0.5 grey in object space.
        neutral = np.array([0.5, 0.5, 1.0]) if space == "tangent" else np.full(3, 0.5)
        normal_map[~_grown_mask(filled, dilation)] = neutral
    levels = 2**bits - 1
    normal_map = (
        (np.clip(normal_map, 0.0, 1.0) * levels)
        .round()
        .astype(np.uint8 if bits == 8 else np.uint16)
    )

    low_mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    num_texels = int(positions.shape[0])
    stats = {
        "num_texels": num_texels,
        "num_hits": num_hits,
        "hit_fraction": float(num_hits / num_texels) if num_texels else 0.0,
        "cage": float(cage),
        "max_distance": float(max_distance),
        "space": space,
        "bits": int(bits),
        # The smallest normal deviation this encoding can represent at all.
        # Detail finer than this is quantization noise rather than recovered
        # geometry, so it is the number that says whether the bake was worth
        # doing on this pair of meshes.
        "quantization_floor": float(2.0 / levels),
    }
    return low_mesh, normal_map, stats


def _grown_mask(filled: np.ndarray, iterations: int) -> np.ndarray:
    """Which texels :func:`_fill_texture_holes` reaches in ``iterations`` steps."""
    grown = filled.copy()
    for _ in range(max(iterations, 0)):
        if grown.all():
            break
        neighbors = np.zeros_like(grown)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted = np.roll(grown, shift, axis=axis)
            edge = 0 if shift > 0 else -1
            if axis == 0:
                shifted[edge, :] = False
            else:
                shifted[:, edge] = False
            neighbors |= shifted
        grown |= neighbors
    return grown


def bake_ambient_occlusion(
    mesh,
    occluder_mesh=None,
    texture_size: int = 2048,
    num_samples: int = 64,
    max_distance: Optional[float] = None,
    cage: Optional[float] = None,
    seed: int = 0,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
    max_rays_per_batch: int = 1 << 22,
):
    """Bake an ambient-occlusion map onto ``mesh``'s UV atlas.

    The third map of the standard photogrammetry set, after albedo
    (:func:`bake_texture_atlas`) and normals (:func:`bake_normal_map`). AO
    records how much of the sky each point of the surface can actually see, so
    creases, contact points and cavities darken -- the cue that stops a scanned
    asset reading as flat under ambient light, and the one thing neither of the
    other two maps carries.

    Each texel casts ``num_samples`` rays over the cosine-weighted hemisphere
    about its normal (Malley's method: a uniform disk projected up, which is
    exactly the distribution the occlusion integral is weighted by, so a plain
    mean of the results is an unbiased estimate). The value stored is the
    fraction that escaped -- 1.0 fully open, 0.0 fully enclosed.

    Args:
        mesh: The ``open3d.geometry.TriangleMesh`` to bake onto. Must be
            manifold; it gains ``triangle_uvs`` in place, reusing any it
            already carries so this map lines up with the others.
        occluder_mesh: What the rays are tested against. Defaults to ``mesh``
            itself (self-occlusion). Pass the pre-decimation mesh to capture
            detail the decimated one no longer has, exactly as
            :func:`bake_normal_map` does.
        texture_size: Width/height of the (square) map, in texels.
        num_samples: Rays per texel. The estimate is a Monte-Carlo average, so
            its noise falls as ``1/sqrt(num_samples)``; 64 is a reasonable
            preview and a few hundred is smooth.
        max_distance: Occluders farther than this are ignored, which is what
            keeps AO a *local* contact cue rather than a global one. Defaults
            to half the mesh's bounding-box diagonal.
        cage: How far along the normal each ray starts, in scene units. When
            ``occluder_mesh`` is a *different* mesh this must clear the gap
            between the two surfaces, and defaults to 2% of the bounding-box
            diagonal to do so. Decimation cuts corners, so most of a
            simplified mesh's surface lies *inside* the dense one it came from
            -- measured at 80% of texels on a decimated bumpy sphere -- and a
            ray starting under the occluder hits it immediately, baking a
            uniformly dark map that looks plausible and is entirely wrong. For
            self-occlusion the default is a hair off the surface, just enough
            to avoid re-hitting the originating triangle.
        seed: Seeds the hemisphere sampling. Re-baking the *same* mesh gives
            the same map, since it then carries the UVs from the first bake;
            two separately unwrapped copies of one mesh will differ, because
            open3d's unwrapper is not deterministic (see
            :func:`_unwrap_and_rasterize`).
        unwrap_size, gutter, margin, max_stretch, dilation: As in
            :func:`bake_texture_atlas`.
        max_rays_per_batch: Cap on rays cast at once. A large atlas times a
            high sample count is easily hundreds of millions of rays, so they
            are cast in batches to bound peak memory.

    Returns:
        ``(mesh, ao_map, stats)``: the mesh with ``triangle_uvs`` set; the
        (``texture_size``, ``texture_size``, 3) ``uint8`` grayscale map; and
        ``stats`` with ``num_texels``, ``num_samples``, ``mean_ao``,
        ``min_ao`` and the ``max_distance`` used. A ``mean_ao`` of essentially
        1.0 means nothing occluded anything -- expected for a convex shape,
        and a sign the distance is too small for anything else.

    Raises:
        ValueError: If ``num_samples`` is not positive, or ``mesh`` cannot be
            unwrapped (see :func:`bake_texture_atlas`).
    """
    o3d = _require_open3d()

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    occluder = mesh if occluder_mesh is None else occluder_mesh
    if len(occluder.triangles) == 0:
        raise ValueError(
            "Cannot bake ambient occlusion against a mesh with no triangles."
        )

    atlas = _unwrap_and_rasterize(
        mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
        with_tangents=True,
    )

    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    diagonal = float(np.linalg.norm(extent))
    if max_distance is None:
        max_distance = 0.5 * diagonal
    if cage is None:
        # Against a different mesh the origin has to clear the gap between the
        # two surfaces; against itself it only has to clear its own triangle.
        cage = (0.02 if occluder_mesh is not None else 1e-4) * diagonal
    cage = max(cage, 1e-9)

    positions, normals = atlas.positions, atlas.normals
    tangents = atlas.tangents
    bitangents = np.cross(normals, tangents)
    num_texels = int(positions.shape[0])
    openness = np.ones(num_texels, dtype=np.float64)

    if num_texels > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occluder))
        rng = np.random.default_rng(seed)
        origins_all = positions + normals * cage
        batch = max(1, max_rays_per_batch // num_samples)
        for start in range(0, num_texels, batch):
            stop = min(start + batch, num_texels)
            count = stop - start
            # Malley's method: a uniformly sampled disk lifted onto the
            # hemisphere is cosine-distributed about the normal.
            r1 = rng.random((count, num_samples))
            r2 = rng.random((count, num_samples))
            radius = np.sqrt(r1)
            angle = 2.0 * np.pi * r2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.sqrt(np.clip(1.0 - r1, 0.0, None))
            directions = (
                x[..., None] * tangents[start:stop, None, :]
                + y[..., None] * bitangents[start:stop, None, :]
                + z[..., None] * normals[start:stop, None, :]
            )
            origins = np.repeat(origins_all[start:stop, None, :], num_samples, axis=1)
            rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6)
            hit_t = (
                scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"]
                .numpy()
                .reshape(count, num_samples)
            )
            occluded = np.isfinite(hit_t) & (hit_t < max_distance)
            openness[start:stop] = 1.0 - occluded.mean(axis=1)

    ao_map = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    if num_texels > 0:
        ao_map[atlas.rows, atlas.cols] = openness[:, None]
        filled[atlas.rows, atlas.cols] = True
    ao_map = _fill_texture_holes(ao_map, filled, dilation)
    # Texels the dilation never reaches are outside every chart; leave them
    # fully open rather than black, so a stray sample can't darken the asset.
    ao_map[~_grown_mask(filled, dilation)] = 1.0
    ao_map = (np.clip(ao_map, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    stats = {
        "num_texels": num_texels,
        "num_samples": int(num_samples),
        "mean_ao": float(openness.mean()) if num_texels else 1.0,
        "min_ao": float(openness.min()) if num_texels else 1.0,
        "max_distance": float(max_distance),
        "cage": float(cage),
    }
    return mesh, ao_map, stats


def _face_adjacency(triangles: np.ndarray) -> np.ndarray:
    """Pairs of faces sharing an edge, as an ``(E, 2)`` array.

    Built by sorting each face's three vertex-pairs into canonical edge keys and
    grouping with ``np.unique`` -- no graph library needed. Edges shared by more
    than two faces (non-manifold) contribute no pair rather than a
    combinatorial explosion of them.
    """
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    faces = np.tile(np.arange(len(triangles)), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, faces = edges[order], faces[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    # A manifold interior edge appears exactly twice and so shows up as a
    # single True; a boundary edge once (no True); a non-manifold edge more
    # often, where the runs below deliberately pair only consecutive duplicates.
    starts = np.nonzero(same)[0]
    if starts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.stack([faces[starts], faces[starts + 1]], axis=1)
    return pairs[pairs[:, 0] != pairs[:, 1]].astype(np.int64)


def _gradient_summed_area(image: np.ndarray) -> np.ndarray:
    """Summed-area table of an image's gradient magnitude.

    Lets the mean gradient over any axis-aligned box be read in O(1), which is
    what makes scoring every (face, view) pair affordable: the alternative is a
    variable-size slice per pair, in Python, F x V times.
    """
    # float64 throughout: training images arrive as float32, and accumulating
    # a 200k-entry cumulative sum in float32 leaves box readouts wrong by up to
    # 6e-5 -- comparable to a typical gradient magnitude over a flat patch, and
    # enough to drive the readout negative. That error lands directly in the
    # (face, view) quality scores this table exists to compute.
    image = np.asarray(image, dtype=np.float64)
    gray = image.mean(axis=2) if image.ndim == 3 else image
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, :-1] = np.abs(np.diff(gray, axis=1))
    grad_y[:-1, :] = np.abs(np.diff(gray, axis=0))
    magnitude = grad_x + grad_y
    table = np.zeros((magnitude.shape[0] + 1, magnitude.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = magnitude.cumsum(axis=0).cumsum(axis=1)
    return table


def _box_means(table: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """Mean of the tabulated quantity over each ``[x0, x1) x [y0, y1)`` box.

    Clamped at zero. The four-corner difference of a cumulative sum loses
    precision catastrophically when a nearly-empty box is read out of a large
    table, and can come back a few times ``1e-5`` *below* zero -- which is
    impossible for a mean of gradient magnitudes. It matters because the caller
    feeds this to ``-log()``: a negative quality becomes ``NaN``, ``np.argmin``
    returns the index of a ``NaN`` in preference to any real value, and the
    face would be textured from the one view that cannot see it.
    """
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    return np.maximum(total / area, 0.0)


def _project_triangles(vertices, triangles, camtoworld, K):
    """Project each triangle's corners into one view -> ``(T, 3, 2)`` pixels."""
    viewmat = np.linalg.inv(camtoworld)
    corners = vertices[triangles]  # (T, 3, 3)
    cam = (viewmat[:3, :3] @ corners.reshape(-1, 3).T + viewmat[:3, 3:4]).T
    uvw = (K @ cam.T).T
    return (uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)).reshape(-1, 3, 2)


def _triangle_pixel_areas(uv):
    """Area of each projected triangle, in square pixels, from ``(T, 3, 2)``."""
    edge1 = uv[:, 1] - uv[:, 0]
    edge2 = uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])


def face_projected_areas(mesh, dataset, max_views: Optional[int] = None) -> np.ndarray:
    """The largest area, in source pixels, each face covers in any view seeing it.

    How much photographic evidence exists for that patch of surface. A face
    that only ever appears twelve pixels across was never photographed in
    enough detail to fill more than twelve texels, however large an atlas it is
    given; one that fills a thousand pixels in a close-up carries detail a
    small atlas would throw away. Summed over the mesh this is what
    :func:`recommended_texture_size` turns into an atlas resolution.

    The *maximum* over views rather than the mean or the sum: texture detail is
    limited by the best look the capture ever got at a surface, not by how many
    mediocre looks it got.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        max_views: If given, only the first ``max_views`` images are consulted.

    Returns:
        ``(F,)`` float array of square pixels. Zero for a face no view sees.
    """
    _require_open3d()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        raise ValueError("Cannot measure projected areas for a mesh with no triangles.")

    visible = face_visibility(mesh, dataset, max_views=max_views)
    areas = np.zeros(len(triangles), dtype=np.float64)
    for view_index in range(visible.shape[1]):
        seen = visible[:, view_index]
        if not seen.any():
            continue
        data = dataset[view_index]
        uv = _project_triangles(
            vertices, triangles[seen], data["camtoworld"].numpy(), data["K"].numpy()
        )
        areas[seen] = np.maximum(areas[seen], _triangle_pixel_areas(uv))
    return areas


def recommended_texture_size(
    mesh,
    dataset,
    max_views: Optional[int] = None,
    texels_per_pixel: float = 1.0,
    packing_efficiency: Optional[float] = None,
    probe_size: int = 512,
    min_size: int = 256,
    max_size: int = 16384,
):
    """Pick an atlas resolution from how much photographic evidence exists.

    ``--texture_size`` is the same wrong question ``--target_triangles`` was:
    the right value depends on the capture, and guessing it is only checked
    afterwards by looking at the result. Too small throws away detail the
    photographs actually contain; too large invents texels no photograph can
    fill, and pays for them in memory, upload time and bake time for nothing.

    The evidence is :func:`face_projected_areas` summed over the mesh -- the
    source pixels covering each patch of surface, counted at the best look the
    capture ever got at it. One texel per such pixel is the point where the
    atlas stops discarding detail and starts inventing it.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``, ideally after culling
            (:func:`~gsplat.photogrammetry.mesh_extraction.cull_unobserved_faces`)
            and decimation, so the answer describes the mesh that ships.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        max_views: If given, only the first ``max_views`` images are consulted.
        texels_per_pixel: Texels to allocate per source pixel. 1.0 matches the
            evidence; 2.0 supersamples (4x the texels) and is occasionally
            worth it if the atlas will be viewed magnified, though it cannot
            add detail that was never photographed.
        packing_efficiency: Fraction of the atlas a UV unwrap actually covers,
            the rest being the gaps between charts. ``None`` (the default)
            **measures it on this mesh** with one probe unwrap, because there
            is no defensible constant: measured across test spheres it ranges
            from 42.7% to 73.2%, and not monotonically in mesh density. It is
            a stable property of the mesh rather than noise, though -- five
            repeated unwraps of one mesh spread by at most 2.8%, and by 0.0%
            on two of the three tried -- which is what makes probing worth the
            extra unwrap. Pass a number to skip the probe.
        probe_size: Atlas size for that probe. Coverage drifts slightly with
            it (45.7% / 43.8% / 42.7% / 42.1% at 128 / 256 / 512 / 1024 on one
            mesh, as coarse rasterization over-counts small triangles), so 512
            trades a little accuracy for a much cheaper unwrap than the sizes
            actually being recommended.
        min_size: Never recommend below this.
        max_size: Never recommend above this, however much evidence there is.

    Returns:
        ``(size, stats)``. ``size`` is the *nearest* power of two, which is what
        GPUs and mipmapping want -- nearest rather than next-up, because
        rounding up quadruples the atlas for an arbitrarily small overshoot. ``stats`` reports ``total_source_pixels``,
        ``exact_size`` (before rounding and clamping -- the number to compare
        against, since the rounding quantises everything else),
        ``num_faces``/``num_unobserved_faces``, the ``texels_per_pixel`` and
        ``packing_efficiency`` used, and ``clamped`` (whether a bound bit).

    Raises:
        ValueError: If ``mesh`` has no triangles, either factor is not
            positive, or the size bounds are not a valid range. Measuring the
            packing also inherits :func:`_unwrap_and_rasterize`'s requirement
            that the mesh be manifold -- pass ``packing_efficiency`` explicitly
            to skip that.
    """
    _require_open3d()

    if texels_per_pixel <= 0:
        raise ValueError(f"texels_per_pixel must be positive, got {texels_per_pixel}.")
    if packing_efficiency is not None and not 0 < packing_efficiency <= 1:
        raise ValueError(
            f"packing_efficiency must be in (0, 1], got {packing_efficiency}."
        )
    if probe_size <= 0:
        raise ValueError(f"probe_size must be positive, got {probe_size}.")
    if min_size <= 0 or max_size < min_size:
        raise ValueError(
            f"Need 0 < min_size <= max_size, got {min_size} and {max_size}."
        )

    measured_packing = packing_efficiency is None
    if measured_packing:
        # One probe unwrap, purely to see how much of an atlas this mesh's
        # charts actually cover. `_unwrap_and_rasterize` does not mutate the
        # mesh, so this cannot leave behind a UV layout scaled for the probe
        # size -- which would then be reused by the real bake and waste a
        # gutter sized for the wrong resolution.
        probe = _unwrap_and_rasterize(mesh, probe_size)
        packing_efficiency = float(probe.rows.size) / float(probe_size**2)
        if packing_efficiency <= 0:
            raise ValueError(
                "The probe unwrap covered no texels at all, so there is no "
                "packing efficiency to measure. The mesh is probably "
                "degenerate."
            )

    areas = face_projected_areas(mesh, dataset, max_views=max_views)
    total_pixels = float(areas.sum())
    needed = total_pixels * texels_per_pixel / packing_efficiency
    exact = float(np.sqrt(needed))

    if exact <= 0:
        size = min_size
    else:
        # Nearest power of two *in log space*, not the next one up. Rounding up
        # quadruples the atlas for an arbitrarily small overshoot: measured on
        # a test sphere, an exact size of 518.1 rounded up to 1024 and baked
        # 3.88x more texels than there were source pixels to fill them. Nearest
        # lands on 512 there and covers 0.97x the evidence -- 3% under, against
        # 288% over. Under-provisioning by a few percent costs a little detail;
        # over-provisioning by 4x costs four times the memory and upload for
        # detail that does not exist.
        size = int(2 ** np.round(np.log2(exact)))
    clamped = size < min_size or size > max_size
    size = int(np.clip(size, min_size, max_size))

    return size, {
        "size": size,
        "exact_size": exact,
        "total_source_pixels": total_pixels,
        "num_faces": int(len(areas)),
        "num_unobserved_faces": int((areas <= 0).sum()),
        "texels_per_pixel": float(texels_per_pixel),
        "packing_efficiency": float(packing_efficiency),
        "packing_efficiency_measured": bool(measured_packing),
        "clamped": bool(clamped),
    }


def face_visibility(mesh, dataset, max_views: Optional[int] = None) -> np.ndarray:
    """Which views can actually see each face, as an ``(F, V)`` bool array.

    A face counts as visible from a view when its centroid projects inside that
    image, faces towards it, and the ray from the camera reaches it without
    hitting other geometry first -- the same test every bake in this module
    uses for a *point*, applied to face centroids, so "visible" means one thing
    across the module rather than three subtly different things.

    **This is not the same question as :func:`face_view_quality` being
    non-zero**, and conflating them deletes real geometry. Quality is gradient
    energy over the projection, so a face that is perfectly visible on a flat,
    untextured surface -- a painted wall, a clear sky reflection -- scores
    ~0 because there is no detail there to measure, not because no camera saw
    it. Anything deciding whether a face was *observed* must ask this.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        max_views: If given, only the first ``max_views`` images are consulted.

    Returns:
        ``(F, V)`` bool array.

    Raises:
        ValueError: If ``mesh`` has no triangles.
    """
    o3d = _require_open3d()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        raise ValueError("Cannot compute face visibility for a mesh with no triangles.")

    centroids = vertices[triangles].mean(axis=1)
    face_normals = np.asarray(
        mesh.triangle_normals
        if mesh.has_triangle_normals()
        else mesh.compute_triangle_normals().triangle_normals
    )
    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    visible = np.zeros((len(triangles), num_views), dtype=bool)
    for chunk, _colors, _weights, view_index in _view_samples(
        scene, o3d, dataset, centroids, face_normals, max_views, 1 << 20
    ):
        visible[chunk, view_index] = True
    return visible


def face_view_quality(mesh, dataset, max_views: Optional[int] = None):
    """Score how well each view could texture each face.

    The data term of the view-selection MRF (see :func:`select_views_mrf`).
    Following Waechter et al., a view's worth for a face is the *gradient energy
    over the face's projection* -- which rewards being close and
    fronto-parallel (a large projection) and being in focus (a strong gradient)
    in one number, and correctly demotes a blurred or motion-smeared view whose
    geometry is otherwise ideal.

    Visibility comes from :func:`_view_samples`, so occlusion is decided by
    exactly the same ray cast the blended bakes use rather than a second,
    subtly-different implementation.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        max_views: If given, only the first ``max_views`` images are scored.

    Returns:
        ``(F, V)`` float array. Zero means the view cannot texture that face at
        all -- occluded, out of frame, or facing away.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        raise ValueError("Cannot score views for a mesh with no triangles.")

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    quality = np.zeros((len(triangles), num_views), dtype=np.float64)

    # Visibility, from the same ray cast the blended bakes use.
    visible = face_visibility(mesh, dataset, max_views=max_views)

    # 3D face area, the part of the projected area that doesn't depend on view.
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    for view_index in range(num_views):
        seen = visible[:, view_index]
        if not seen.any():
            continue
        data = dataset[view_index]
        camtoworld = data["camtoworld"].numpy()
        K = data["K"].numpy()
        image = data["image"].numpy() / 255.0
        height, width = image.shape[:2]

        uv = _project_triangles(vertices, triangles[seen], camtoworld, K)
        projected_area = _triangle_pixel_areas(uv)

        x0 = np.clip(np.floor(uv[..., 0].min(axis=1)), 0, width - 1).astype(np.int64)
        x1 = np.clip(np.ceil(uv[..., 0].max(axis=1)) + 1, 1, width).astype(np.int64)
        y0 = np.clip(np.floor(uv[..., 1].min(axis=1)), 0, height - 1).astype(np.int64)
        y1 = np.clip(np.ceil(uv[..., 1].max(axis=1)) + 1, 1, height).astype(np.int64)
        x1 = np.maximum(x1, x0 + 1)
        y1 = np.maximum(y1, y0 + 1)
        sharpness = _box_means(_gradient_summed_area(image), x0, y0, x1, y1)

        # Gradient energy over the projection: sharpness carried over the area
        # the face actually covers, not the box used to measure it.
        quality[seen, view_index] = projected_area * sharpness
        # A face degenerate in this view contributes nothing, whatever its
        # gradient: it has no pixels to take colour from.
        quality[seen, view_index] *= face_area[seen] > 0

    return quality


NO_VIEW = -1


def select_views_mrf(
    quality: np.ndarray,
    adjacency: np.ndarray,
    smoothness: float = 1.0,
    max_iterations: int = 20,
    max_seeds: int = 8,
):
    """Choose one view per face, trading per-face quality against seam count.

    Blending every view is a low-pass filter: two views of a point are never
    registered to sub-pixel accuracy after real SfM, so averaging them destroys
    exactly the high-frequency detail the photographs contain. Texturing each
    face from a *single* view keeps that detail. The cost is a colour
    discontinuity wherever neighbouring faces choose differently, so the choice
    is posed as an energy that also counts those seams:

    .. code-block:: text

        E(l) = sum_f  D_f(l_f)  +  smoothness * sum_(f,g adjacent) [ l_f != l_g ]
        D_f(v) = -log(quality[f, v] + eps),   D_f(NO_VIEW) = a large constant

    The Potts smoothness term penalises the *number* of seams, which is both
    what we want minimised and what keeps the levelling step afterwards
    tractable.

    **Optimiser: ICM (iterated conditional modes).** Start from the per-face
    best view, then sweep faces repeatedly, moving each to the label that
    minimises its local energy given its neighbours, until a sweep changes
    nothing.

    A single ICM run is badly seed-dependent, and not in a subtle way: swept
    from the per-face best view with a strong smoothness term, the first face
    to move can cascade every other face onto its neighbour's label. On a
    three-face example that lands on total energy 4.14 where 4.39 *below zero*
    was available. So ICM is run from several seeds -- the per-face best, and
    "every face takes view alpha" for each of the strongest few views -- and
    the lowest-energy result wins. Each seed is a fixed point of the same
    sweep, so this is deterministic, and it can only improve on any one seed.

    *Honest tradeoff:* even so, ICM is a greedy local optimiser with no
    optimality bound, where alpha-expansion via graph cut is within a known
    factor of the global optimum. ICM is used because alpha-expansion needs a
    max-flow solver, and this package's optional ``mesh`` extra is deliberately
    just open3d and imageio. For a Potts model with a strong data term the
    residual gap is mostly a few extra seams -- which is what seam levelling
    removes anyway. The signature is kept optimiser-agnostic so a graph cut can
    be dropped in behind it.

    Args:
        quality: ``(F, V)`` from :func:`face_view_quality`. Zero = unusable.
        adjacency: ``(E, 2)`` face pairs from :func:`_face_adjacency`.
        smoothness: Weight on the seam count. Raising it yields fewer, larger
            single-view regions at the cost of using worse views inside them.
        max_iterations: Cap on ICM sweeps per seed.
        max_seeds: Cap on the "every face takes view alpha" seeds tried, taken
            from the views with the most total quality. Bounds the cost on
            captures with hundreds of images.

    Returns:
        ``(labels, stats)``. ``labels`` is ``(F,)`` of view indices, with
        :data:`NO_VIEW` where no view can texture the face at all. ``stats``
        reports ``num_faces``, ``num_unlabelled``, ``num_seams``,
        ``num_views_used``, ``energy``, ``iterations`` and ``num_seeds``.
    """
    quality = np.asarray(quality, dtype=np.float64)
    num_faces, num_views = quality.shape

    # Data term. An unusable (zero-quality) view must never be chosen, so give
    # it +inf rather than a merely-large cost -- a large finite cost can still
    # win against enough smoothness pressure, which would texture a face from a
    # camera that cannot see it.
    with np.errstate(divide="ignore"):
        data = -np.log(quality)
    data[quality <= 0] = np.inf

    usable = np.isfinite(data)
    has_any = usable.any(axis=1)
    # Cost of giving up on a face: worse than any real option, so NO_VIEW is
    # only ever chosen where nothing else is available.
    finite = data[usable]
    no_view_cost = (finite.max() + 1.0) if finite.size else 0.0

    # Neighbours per face, so each sweep can vectorise over candidate labels.
    neighbours = [[] for _ in range(num_faces)]
    for face_a, face_b in adjacency:
        neighbours[face_a].append(face_b)
        neighbours[face_b].append(face_a)
    neighbours = [np.asarray(n, dtype=np.int64) for n in neighbours]

    def total_energy(labels):
        cost = 0.0
        for face in range(num_faces):
            label = int(labels[face])
            cost += no_view_cost if label == NO_VIEW else data[face, label]
        if len(adjacency):
            cost += smoothness * float(
                np.sum(labels[adjacency[:, 0]] != labels[adjacency[:, 1]])
            )
        return cost

    def sweep(labels):
        """Run ICM from `labels` until a pass changes nothing."""
        labels = labels.copy()
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            changed = 0
            for face in range(num_faces):
                if not has_any[face]:
                    continue
                candidates = np.nonzero(usable[face])[0]
                costs = data[face, candidates].copy()
                neighbour_labels = labels[neighbours[face]]
                if neighbour_labels.size:
                    costs = costs + smoothness * (
                        candidates[:, None] != neighbour_labels[None, :]
                    ).sum(axis=1)
                best = candidates[int(np.argmin(costs))]
                if best != labels[face]:
                    labels[face] = best
                    changed += 1
            if changed == 0:
                break
        return labels, iterations

    # Seed 1: each face's own best view, ignoring seams.
    greedy = np.full(num_faces, NO_VIEW, dtype=np.int64)
    greedy[has_any] = np.argmin(np.where(usable, data, np.inf), axis=1)[has_any]
    seeds = [greedy]

    # Seeds 2..n: "every face that can, takes view alpha", for the strongest
    # few views. These are the labellings a seam-dominated energy actually
    # wants, and no per-face sweep from the greedy seed can reach them.
    strength = np.where(usable, quality, 0.0).sum(axis=0)
    for alpha in np.argsort(strength)[::-1][: max(max_seeds, 0)]:
        if strength[alpha] <= 0:
            break
        seed = greedy.copy()
        takeable = usable[:, alpha]
        seed[takeable] = alpha
        seeds.append(seed)

    best_labels, best_energy, best_iterations = None, np.inf, 0
    for seed in seeds:
        candidate, iterations = sweep(seed)
        energy = total_energy(candidate)
        if energy < best_energy:
            best_labels, best_energy, best_iterations = candidate, energy, iterations
    labels = best_labels

    seams = (
        int(np.sum(labels[adjacency[:, 0]] != labels[adjacency[:, 1]]))
        if len(adjacency)
        else 0
    )
    stats = {
        "num_faces": int(num_faces),
        "num_unlabelled": int(np.sum(labels == NO_VIEW)),
        "num_seams": seams,
        "num_views_used": int(len(np.unique(labels[labels != NO_VIEW]))),
        "energy": float(best_energy),
        "iterations": int(best_iterations),
        "num_seeds": int(len(seeds)),
    }
    return labels, stats


def _texel_face_ids(o3d, mesh, positions, normals):
    """Which triangle each atlas texel's surface point belongs to.

    The atlas rasterizer interpolates positions and normals but does not record
    the source triangle, and view selection is *per face*, so it has to be
    recovered: nudge each texel out along its normal and cast straight back
    down, then read ``primitive_ids``. Measured 100% recovery on a sphere.

    Texels rasterized in the margin around a chart are extrapolated off the
    surface and may hit nothing (or a neighbour); those come back as -1 so the
    caller can fall back rather than texture them from a wrong face's view.

    Returns:
        ``(face_ids, hit, barycentric)``: the ``(N,)`` face index per texel (-1
        where none), the ``(N,)`` bool mask of texels that hit, and the
        ``(N, 3)`` barycentric weights of the hit within that face, ordered to
        match the face's corners.
    """
    if positions.shape[0] == 0:
        return (
            np.full(0, -1, dtype=np.int64),
            np.zeros(0, dtype=bool),
            np.zeros((0, 3)),
        )

    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    eps = max(1e-4 * float(np.linalg.norm(extent)), 1e-9)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.concatenate([positions + normals * eps, -normals], axis=1)
    result = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    hit_t = result["t_hit"].numpy()
    primitives = result["primitive_ids"].numpy()

    # A texel on the surface hits its own face at t ~ eps. A much longer hit
    # means the ray started off-surface and landed on unrelated geometry.
    hit = np.isfinite(hit_t) & (hit_t <= 4.0 * eps)
    face_ids = np.where(hit, primitives.astype(np.int64), -1)
    # open3d's primitive_uvs weight the corners (1-u-v, u, v) -- pinned
    # empirically elsewhere in this module by reconstructing hit positions.
    uv = result["primitive_uvs"].numpy()
    barycentric = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)
    barycentric[~hit] = 0.0
    return face_ids, hit, barycentric


def bake_texture_atlas_view_selected(
    mesh,
    dataset,
    texture_size: int = 2048,
    max_views: Optional[int] = None,
    mrf_smoothness: float = 1.0,
    outlier_sigma: Optional[float] = None,
    outlier_iterations: int = 3,
    unwrap_size: Optional[int] = None,
    gutter: float = 1.0,
    margin: float = 2.0,
    max_stretch: float = 1.0 / 6.0,
    dilation: int = 4,
    seam_smoothness: Optional[float] = 0.1,
):
    """Bake a UV atlas that samples **one** view per face instead of blending.

    :func:`bake_texture_atlas` averages every view that sees a texel. Two views
    of a surface point are never registered to sub-pixel accuracy after real
    SfM, so that average is a low-pass filter and the atlas comes out blurrier
    than any single source photograph. This instead chooses one view per face
    (:func:`select_views_mrf` over :func:`face_view_quality`, following Waechter
    et al., *Let There Be Color!*, ECCV 2014) and samples each texel from that
    view alone.

    **This is a tradeoff, not a strict improvement, and it is deliberately not
    the default.** Measured on a synthetic sphere with cameras perturbed to
    simulate residual pose error: at 45' of rotation the blended atlas retains
    53% of the ground truth's contrast and this one retains 105% -- but the
    blended atlas is *closer pointwise* (L1 0.152 vs 0.185), because blending
    attenuates detail while single-view sampling displaces it. Pick this for an
    asset that will be looked at; pick blending for one that will be measured.
    :func:`gsplat.photogrammetry.metrics.atlas_sharpness` reports the numbers
    that distinguish them.

    The blended atlas is computed anyway and used wherever a single view will
    not do: faces no view can texture (:data:`NO_VIEW`), texels whose face
    could not be recovered, and texels their face's chosen view cannot actually
    see. Averaging a handful of views is the right answer there, and it is what
    keeps this from punching holes in the atlas.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``. Must be manifold, as for
            :func:`bake_texture_atlas`.
        dataset: An ``examples.datasets.colmap.Dataset``-like object.
        texture_size: Width/height of the (square) texture, in texels.
        max_views: If given, only the first ``max_views`` images are used.
        mrf_smoothness: Seam penalty passed to :func:`select_views_mrf`. Higher
            means fewer, larger single-view regions textured from worse views.
        outlier_sigma: Robust fusion for the *blended fallback* -- see
            :func:`_bake_points_from_views`. The two are complements, not
            alternatives: robust blending still governs the fallback regions.
        unwrap_size: See :func:`bake_texture_atlas`.
        gutter: See :func:`bake_texture_atlas`.
        margin: See :func:`bake_texture_atlas`.
        max_stretch: See :func:`bake_texture_atlas`.
        dilation: See :func:`bake_texture_atlas`.
        seam_smoothness: Levelling strength -- see :func:`level_seams`, which
            removes the exposure/white-balance steps where neighbouring faces
            chose different photographs. ``None`` skips levelling, which is
            useful only for measuring what it did.

    Returns:
        ``(mesh, texture, stats)``. ``mesh`` carries ``triangle_uvs``/
        ``textures``; ``texture`` is the ``(texture_size, texture_size, 3)``
        ``uint8`` atlas. ``stats`` reports the labelling under ``"mrf"``, the
        texel accounting (``num_texels``, ``num_texels_view_selected``,
        ``num_texels_blended``, ``fallback_fraction``, ``num_texels_no_face``),
        ``atlas_sharpness`` for both this atlas and the blended one it replaced
        (measured over the same covered texels so they are comparable), and,
        when levelling ran, ``seam_levelling`` plus ``seam_discontinuity``
        before and after it.

    Raises:
        ValueError: If ``mesh`` has no triangles, ``texture_size`` is not
            positive, or ``mesh`` is non-manifold (see
            :func:`_unwrap_and_rasterize`).
    """
    from .metrics import atlas_sharpness, seam_discontinuity

    o3d = _require_open3d()

    atlas = _unwrap_and_rasterize(
        mesh,
        texture_size,
        unwrap_size=unwrap_size,
        gutter=gutter,
        margin=margin,
        max_stretch=max_stretch,
    )
    rows, cols = atlas.rows, atlas.cols
    positions, normals = atlas.positions, atlas.normals

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    blended_texture = np.zeros_like(texture)
    filled = np.zeros((texture_size, texture_size), dtype=bool)
    stats = {
        "num_texels": int(rows.size),
        "num_texels_view_selected": 0,
        "num_texels_blended": 0,
        "fallback_fraction": 0.0,
        "num_texels_no_face": 0,
        "mrf": {},
        "atlas_sharpness": {},
        "blended_atlas_sharpness": {},
    }
    labels = np.full(len(mesh.triangles), NO_VIEW, dtype=np.int64)

    if rows.size > 0:
        # 1. The blended bake, unchanged -- the fallback everywhere a single
        #    view won't do.
        color_accum, weight_accum = _bake_points_from_views(
            mesh,
            dataset,
            positions,
            normals,
            max_views=max_views,
            outlier_sigma=outlier_sigma,
            outlier_iterations=outlier_iterations,
        )
        observed = weight_accum > 0
        blended = np.zeros_like(color_accum)
        blended[observed] = color_accum[observed] / weight_accum[observed, None]

        # 2. One view per face.
        face_ids, has_face, barycentric = _texel_face_ids(o3d, mesh, positions, normals)
        quality = face_view_quality(mesh, dataset, max_views=max_views)
        adjacency = _face_adjacency(np.asarray(mesh.triangles))
        mrf_labels, mrf_stats = select_views_mrf(
            quality, adjacency, smoothness=mrf_smoothness
        )
        labels = mrf_labels
        texel_label = np.full(positions.shape[0], NO_VIEW, dtype=np.int64)
        texel_label[has_face] = labels[face_ids[has_face]]

        # 3. Sample each texel from its one view. Reusing _view_samples means
        #    the projection, in-frame test and occlusion ray cast are the same
        #    ones the blended path uses, rather than a second implementation
        #    that could disagree about what "visible" means.
        selected = np.zeros_like(blended)
        picked = np.zeros(positions.shape[0], dtype=bool)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        for chunk, colors, _weights, view_index in _view_samples(
            scene, o3d, dataset, positions, normals, max_views, 1 << 20
        ):
            take = texel_label[chunk] == view_index
            if take.any():
                selected[chunk[take]] = colors[take]
                picked[chunk[take]] = True

        # 4. Level the seams: neighbouring faces that chose different
        #    photographs disagree about exposure and white balance, and that
        #    step is the price of not blending. Corrections are solved per
        #    (vertex, label) and interpolated across each face, so they close
        #    the seam without introducing a new edge anywhere else.
        levelled = selected
        if seam_smoothness is not None:
            correction = level_seams(
                mesh,
                dataset,
                labels,
                max_views=max_views,
                smoothness=seam_smoothness,
            )
            stats["seam_levelling"] = correction.stats
            if len(correction.values):
                corner_pairs = correction.corner_pairs[face_ids[has_face]]
                usable = (corner_pairs >= 0).all(axis=1)
                rows_of = np.nonzero(has_face)[0][usable]
                delta = (
                    correction.values[corner_pairs[usable]]
                    * barycentric[rows_of][:, :, None]
                ).sum(axis=1)
                levelled = selected.copy()
                take = picked[rows_of]
                levelled[rows_of[take]] += delta[take]

        final = np.where(picked[:, None], levelled, blended)
        covered = observed | picked
        texture[rows[covered], cols[covered]] = final[covered]
        blended_texture[rows[covered], cols[covered]] = blended[covered]
        filled[rows[covered], cols[covered]] = True

        num_covered = int(covered.sum())
        stats["mrf"] = mrf_stats
        stats["num_texels_view_selected"] = int(picked.sum())
        stats["num_texels_blended"] = num_covered - int(picked.sum())
        stats["fallback_fraction"] = (
            float(1.0 - picked.sum() / num_covered) if num_covered else 0.0
        )
        stats["num_texels_no_face"] = int((~has_face).sum())
        stats["atlas_sharpness"] = atlas_sharpness(texture, filled)
        stats["blended_atlas_sharpness"] = atlas_sharpness(blended_texture, filled)

        if seam_smoothness is not None:
            # Measured on the atlas as shipped, before and after the same
            # correction, so the number says what levelling did rather than
            # what it was asked to do.
            unlevelled = np.zeros_like(texture)
            unlevelled[rows[covered], cols[covered]] = np.where(
                picked[covered, None], selected[covered], blended[covered]
            )
            stats["seam_discontinuity"] = seam_discontinuity(
                mesh, texture, labels, atlas.triangle_uvs
            )
            stats["seam_discontinuity_before"] = seam_discontinuity(
                mesh, unlevelled, labels, atlas.triangle_uvs
            )

    texture = _fill_texture_holes(texture, filled, dilation)
    texture = (np.clip(texture, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(atlas.triangle_uvs)
    mesh.textures = [o3d.geometry.Image(texture)]
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(mesh.triangles), dtype=np.int32)
    )
    return mesh, texture, stats


def _conjugate_gradient(matvec, rhs, max_iterations=200, tol=1e-8, project_mean=False):
    """Solve ``A x = rhs`` for symmetric positive semi-definite ``A``.

    ``A`` is given only as ``matvec``, so the matrix is never materialised --
    the seam system has one row per seam vertex and per mesh edge, which is
    dense in neither shape nor storage but is large. Hand-rolled because
    ``gsplat[mesh]`` is deliberately just open3d and imageio; scipy's ``cg``
    would do, and is not available as a hard dependency here.

    Args:
        matvec: Callable applying ``A`` to an ``(N, C)`` array.
        rhs: ``(N, C)`` right-hand side.
        max_iterations: Cap on iterations.
        tol: Stop when the residual norm falls below ``tol`` times the initial.
        project_mean: Subtract the mean of ``x`` (over N) each iteration, to
            anchor the gauge of a system that is singular along the constants
            -- which the seam system is, since its energy only sees
            *differences* of corrections. Cheaper and better conditioned than
            pinning one unknown, which would bias the correction toward
            whichever vertex was pinned.

            For a *consistent* right-hand side this is belt and braces rather
            than load-bearing: every row of the seam system is a difference of
            two unknowns, so its ``rhs`` is already orthogonal to the constants
            and CG started from zero stays in the range space by itself
            (removing this projection changes that solve by less than 1e-9).
            It earns its keep against accumulated rounding on a large system,
            and against a caller whose ``rhs`` is not exactly consistent.

    Returns:
        ``(x, stats)`` with ``iterations``, ``residual`` (relative) and
        ``converged``.
    """
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    if project_mean and residual.shape[0]:
        residual -= residual.mean(axis=0, keepdims=True)
    direction = residual.copy()
    rs_old = float((residual * residual).sum())
    rs_initial = rs_old
    if rs_initial == 0.0:
        return x, {"iterations": 0, "residual": 0.0, "converged": True}

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        a_dir = matvec(direction)
        denominator = float((direction * a_dir).sum())
        if denominator <= 0.0:
            # A is only positive *semi*-definite: a direction in its null space
            # gives no descent, and dividing by it would produce inf.
            break
        alpha = rs_old / denominator
        x += alpha * direction
        residual -= alpha * a_dir
        if project_mean:
            x -= x.mean(axis=0, keepdims=True)
            residual -= residual.mean(axis=0, keepdims=True)
        rs_new = float((residual * residual).sum())
        if rs_new <= tol * tol * rs_initial:
            rs_old = rs_new
            break
        direction = residual + (rs_new / rs_old) * direction
        rs_old = rs_new

    relative = float(np.sqrt(rs_old / rs_initial))
    return x, {
        "iterations": int(iterations),
        "residual": relative,
        "converged": bool(relative <= tol),
    }


def _shared_edge_vertices(triangles: np.ndarray, adjacency: np.ndarray):
    """The two vertices each adjacent face pair shares, as ``(E, 2)``.

    Returns ``(pairs, shared)`` -- the subset of ``adjacency`` that shares
    exactly two vertices, and those vertices. A pair sharing three (a duplicate
    face) is dropped rather than silently contributing a malformed edge.
    """
    if len(adjacency) == 0:
        return adjacency, np.zeros((0, 2), dtype=np.int64)
    first = triangles[adjacency[:, 0]]
    second = triangles[adjacency[:, 1]]
    is_shared = (first[:, :, None] == second[:, None, :]).any(axis=2)  # (E, 3)
    usable = is_shared.sum(axis=1) == 2
    return adjacency[usable], first[usable][is_shared[usable]].reshape(-1, 2)


@dataclass
class _SeamCorrection:
    """Per-(vertex, label) additive colour corrections, ready to interpolate.

    ``corner_pairs[f, c]`` indexes ``values`` for corner ``c`` of face ``f``, or
    -1 where the face carries no label. A texel's correction is the barycentric
    blend of its face's three corner corrections, which is what makes the
    result continuous *inside* each single-view region while still stepping
    across the seams it was solved to close.
    """

    corner_pairs: np.ndarray
    values: np.ndarray
    stats: dict


def level_seams(
    mesh,
    dataset,
    labels: np.ndarray,
    max_views: Optional[int] = None,
    smoothness: float = 0.1,
    samples_per_edge: int = 4,
    max_iterations: int = 200,
) -> "_SeamCorrection":
    """Solve for the additive colour correction that closes view-selection seams.

    Neighbouring faces textured from different photographs meet at a visible
    step: the two cameras disagree about exposure, white balance and shading.
    Following Waechter et al., the fix is not to re-choose views but to add a
    correction that makes them agree along their shared border, spread smoothly
    over each single-view region so nothing else shifts abruptly.

    The unknown is one correction per **(vertex, label)** pair, not per vertex.
    A single per-vertex offset provably cannot close a discontinuity *at* that
    vertex: both sides would receive it and the step would survive unchanged.

    .. code-block:: text

        E(g) = sum over seam edges, at each endpoint v of edge (l1 | l2):
                   || (g[v,l1] - g[v,l2]) + (mean_l1 - mean_l2) ||^2
             + smoothness * sum over edges of a face labelled l:
                   || g[v,l] - g[w,l] ||^2

    ``mean_l1`` and ``mean_l2`` are each view's **mean colour along the shared
    edge**, over the same set of sample points. Comparing the two views *at the
    vertex* instead does not work, and not marginally: measured on the
    synthetic sphere, two views of one vertex disagree by 0.288 (L2 over RGB)
    from pixel quantisation and silhouette bleed alone, where the exposure
    difference this is meant to remove is 0.26. The solve then fits noise
    larger than the signal and makes the atlas *worse*. Averaging along the
    edge is what separates them -- it is the reason the reference method is
    written that way, and it is worth not undoing.

    This is a linear least squares, solved through its normal equations with
    :func:`_conjugate_gradient`; the three colour channels are independent and
    ride along as columns.

    Args:
        mesh: The ``open3d.geometry.TriangleMesh`` that was labelled.
        dataset: The dataset the labels index into.
        labels: ``(F,)`` from :func:`select_views_mrf`, with :data:`NO_VIEW`
            for faces no view can texture.
        max_views: Must match what the labelling used.
        smoothness: Weight on the smoothness term. Too small and the correction
            is a sharp local patch around each seam; too large and it cannot
            close the seam at all.
        samples_per_edge: Points along each seam edge whose colours are
            averaged. More is steadier and costs only arithmetic -- the
            dataset pass is over all of them at once.
        max_iterations: Cap on conjugate-gradient iterations.

    Returns:
        A :class:`_SeamCorrection`. Its ``stats`` reports ``num_pairs``,
        ``num_seam_terms``, ``num_seam_edges``, ``num_unmeasured_edges``, the
        solver's ``iterations``/``residual``/``converged``, and
        ``mean_correction``/``max_correction``.
    """
    o3d = _require_open3d()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)
    labels = np.asarray(labels, dtype=np.int64)

    # 1. The unknowns: every (vertex, label) a labelled face actually uses.
    labelled = labels != NO_VIEW
    corner_pairs = np.full((len(triangles), 3), -1, dtype=np.int64)
    if labelled.any():
        keys = np.stack(
            [triangles[labelled].reshape(-1), np.repeat(labels[labelled], 3)], axis=1
        )
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        corner_pairs[labelled] = inverse.reshape(-1, 3)
    else:
        unique_keys = np.zeros((0, 2), dtype=np.int64)
    num_pairs = len(unique_keys)

    stats = {
        "num_pairs": int(num_pairs),
        "num_seam_edges": 0,
        "num_seam_terms": 0,
        "num_unmeasured_edges": 0,
        "iterations": 0,
        "residual": 0.0,
        "converged": True,
        "mean_correction": 0.0,
        "max_correction": 0.0,
    }
    empty = _SeamCorrection(corner_pairs, np.zeros((num_pairs, 3)), stats)
    if num_pairs == 0:
        return empty

    # 2. The seam edges: adjacent faces that chose different views.
    pairs, shared = _shared_edge_vertices(triangles, _face_adjacency(triangles))
    if len(pairs) == 0:
        return empty
    face_a, face_b = pairs[:, 0], pairs[:, 1]
    is_seam = (
        (labels[face_a] != labels[face_b])
        & (labels[face_a] != NO_VIEW)
        & (labels[face_b] != NO_VIEW)
    )
    if not is_seam.any():
        return empty
    face_a, face_b, shared = face_a[is_seam], face_b[is_seam], shared[is_seam]
    label_a, label_b = labels[face_a], labels[face_b]
    num_edges = len(face_a)
    stats["num_seam_edges"] = int(num_edges)

    # 3. Sample each seam edge at interior points, in both of its views.
    fractions = (np.arange(samples_per_edge) + 1.0) / (samples_per_edge + 1.0)
    start = vertices[shared[:, 0]]
    end = vertices[shared[:, 1]]
    points = (
        start[:, None, :] * (1.0 - fractions[None, :, None])
        + end[:, None, :] * fractions[None, :, None]
    ).reshape(-1, 3)
    normals = (
        vertex_normals[shared[:, 0]][:, None, :] * (1.0 - fractions[None, :, None])
        + vertex_normals[shared[:, 1]][:, None, :] * fractions[None, :, None]
    ).reshape(-1, 3)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    # Which of the two sides each view is responsible for, per sample point.
    edge_of_point = np.repeat(np.arange(num_edges), samples_per_edge)
    wanted = {}
    for side, label in ((0, label_a), (1, label_b)):
        for view_index in np.unique(label):
            selection = np.nonzero(label[edge_of_point] == view_index)[0]
            if selection.size:
                wanted.setdefault(int(view_index), []).append((side, selection))

    sampled = np.zeros((2, len(points), 3), dtype=np.float64)
    seen = np.zeros((2, len(points)), dtype=bool)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for chunk, colors, _weights, view_index in _view_samples(
        scene, o3d, dataset, points, normals, max_views, 1 << 20
    ):
        for side, selection in wanted.get(view_index, ()):
            keep = np.zeros(len(points), dtype=bool)
            keep[selection] = True
            take = keep[chunk]
            if take.any():
                sampled[side, chunk[take]] = colors[take]
                seen[side, chunk[take]] = True

    # 4. Per edge, the mean over the points *both* views saw -- otherwise the
    #    two means would describe different stretches of the edge and their
    #    difference would be geometry, not exposure.
    both = (seen[0] & seen[1]).reshape(num_edges, samples_per_edge)
    counts = both.sum(axis=1)
    measured = counts > 0
    stats["num_unmeasured_edges"] = int((~measured).sum())
    difference = np.zeros((num_edges, 3))
    if measured.any():
        masked = both[..., None]
        totals = (sampled.reshape(2, num_edges, samples_per_edge, 3) * masked).sum(
            axis=2
        )
        means = totals / np.clip(counts, 1, None)[None, :, None]
        difference[measured] = (means[0] - means[1])[measured]

    # 5. Residual rows. Every row is a difference of two unknowns, which is
    #    what makes the system singular along the constants (see the solver's
    #    `project_mean`).
    row_a, row_b, row_weight, row_target = [], [], [], []

    def pair_at(face, vertex):
        """The (vertex, label) unknown for `vertex` as a corner of `face`."""
        corner = (triangles[face] == vertex[:, None]).argmax(axis=1)
        return corner_pairs[face, corner]

    for edge_end in range(2):
        vertex = shared[measured, edge_end]
        row_a.append(pair_at(face_a[measured], vertex))
        row_b.append(pair_at(face_b[measured], vertex))
        row_weight.append(np.ones(int(measured.sum())))
        # g[a] - g[b] should cancel the two views' colour disagreement.
        row_target.append(-difference[measured])
    stats["num_seam_terms"] = int(sum(len(r) for r in row_a))

    if smoothness > 0:
        weight = np.sqrt(smoothness)
        for corner in range(3):
            first = corner_pairs[labelled, corner]
            second = corner_pairs[labelled, (corner + 1) % 3]
            row_a.append(first)
            row_b.append(second)
            row_weight.append(np.full(len(first), weight))
            row_target.append(np.zeros((len(first), 3)))

    if sum(len(r) for r in row_a) == 0:
        return empty

    row_a = np.concatenate(row_a)
    row_b = np.concatenate(row_b)
    row_weight = np.concatenate(row_weight)
    row_target = np.concatenate(row_target)

    def matvec(g):
        scaled = row_weight[:, None] ** 2 * (g[row_a] - g[row_b])
        out = np.zeros_like(g)
        np.add.at(out, row_a, scaled)
        np.add.at(out, row_b, -scaled)
        return out

    rhs = np.zeros((num_pairs, 3))
    scaled_target = row_weight[:, None] * row_target
    np.add.at(rhs, row_a, scaled_target)
    np.add.at(rhs, row_b, -scaled_target)

    correction, solver_stats = _conjugate_gradient(
        matvec, rhs, max_iterations=max_iterations, project_mean=True
    )
    stats.update(solver_stats)
    magnitude = np.linalg.norm(correction, axis=1)
    stats["mean_correction"] = float(magnitude.mean())
    stats["max_correction"] = float(magnitude.max())
    return _SeamCorrection(corner_pairs, correction, stats)
