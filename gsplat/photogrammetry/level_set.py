"""Extract a surface as a level set of a scalar field, via marching tetrahedra.

TSDF fusion of rendered depth maps is the 2020-era answer to "turn a radiance
field into a mesh": render depth from each camera, fuse, march. It inherits
every weakness of the depth maps and it can only ever see what a camera saw.
Current work extracts the surface from the *field itself* -- Gaussian Opacity
Fields (Yu et al., 2024) takes a level set of the opacity field over a
tetrahedralisation of the Gaussian centres; SuGaR (Guedon & Lepetit, 2024) and
PGSR (Chen et al., 2024) are neighbours in the same family.

## What is and is not verified here

**This module has never run against a real Gaussian field, because that needs a
GPU and the compiled CUDA extension, and this machine has neither.** It is
deliberately split so that almost all of it does not:

- :func:`tetrahedral_grid`, :func:`marching_tetrahedra`,
  :func:`extract_level_set` and every diagnostic below are pure NumPy/open3d
  and are **measured against analytic fields** -- a sphere, a plane, a torus --
  where the answer is known in closed form, including a convergence check that
  halving the cell size halves the error.
- :func:`gaussian_opacity_field` is the only part that needs a GPU. It is a
  thin adapter that answers "what is the field at these points", and it is
  **reviewed, not executed**.

That split is the same seam ``_tsdf_fuse`` has, and for the same reason: it is
what let the fusion half of TSDF be trusted while the rendering half could not
be run. Do not report this module as working end to end until someone has run
:func:`gaussian_opacity_field` on a real checkpoint.

## Debugging a GPU run you cannot step through

Everything here is instrumented, because the expensive failures in this kind of
code are silent: a level outside the field's range extracts nothing, a bounding
box that misses the surface extracts nothing, a sign convention flipped inside
out extracts *everything*. Each of those returns an empty or absurd mesh with no
traceback pointing at the cause.

So, in rough order of what to reach for:

1. :func:`probe_field` -- cheap. Samples the field on a coarse grid and reports
   its distribution, whether the requested level is even inside its range, and
   what fraction of samples fall either side. **Run this before a full
   extraction.** It costs one coarse field evaluation and catches the majority
   of "empty mesh" cases before you have spent the GPU time.
2. :func:`validate_level_set_pipeline` -- free, needs no GPU at all. Runs the
   whole extraction against an analytic sphere and checks the result. If this
   fails, the bug is in this module, not in your field.
3. ``extract_level_set(..., diagnostics=...)`` -- fills a dict with per-stage
   counts, timings, field statistics and warnings. Read ``warnings`` first.
4. ``extract_level_set(..., debug_dir=...)`` -- writes the grid points coloured
   by field value, the raw pre-cleanup mesh, and a field histogram to disk, so
   the intermediate state can be opened in a viewer rather than guessed at.
5. :func:`level_set_residual` -- the self-check that needs **no ground truth**:
   it re-evaluates the field at the extracted mesh's own vertices and reports
   how far they are from the requested level. On a correct extraction this is
   near zero. It is the single most useful number on a real capture, where
   there is no reference mesh to compare against.
"""

import os
import time
from typing import Callable, Optional, Sequence

import numpy as np

from ._open3d import _require_open3d

# Tetrahedron edges, in the order the case table below indexes them.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

# Marching-tetrahedra case table, keyed by the bitmask of vertices strictly
# *inside* the level set. Values are triangles given as edge indices into
# `_TET_EDGES`. Winding is not trusted here -- it is fixed afterwards against
# the field gradient by `_orient_against_field`, which is both simpler to get
# right and directly testable.
_MT_CASES = {
    0: [],
    1: [(0, 1, 2)],
    2: [(0, 4, 3)],
    3: [(1, 3, 4), (1, 4, 2)],
    4: [(1, 5, 3)],
    5: [(0, 5, 3), (0, 2, 5)],
    6: [(0, 4, 5), (0, 5, 1)],
    7: [(2, 4, 5)],
    8: [(2, 5, 4)],
    9: [(0, 1, 5), (0, 5, 4)],
    10: [(0, 3, 5), (0, 5, 2)],
    11: [(1, 3, 5)],
    12: [(1, 2, 4), (1, 4, 3)],
    13: [(0, 3, 4)],
    14: [(0, 2, 1)],
    15: [],
}


def tetrahedral_grid(bounds: np.ndarray, resolution: int):
    """A regular grid over ``bounds``, split into tetrahedra.

    Six tetrahedra per cube, the standard Freudenthal decomposition, which
    tiles space without cracks -- neighbouring cubes agree on their shared
    faces, so the extracted surface is watertight where the field is.

    GOF tetrahedralises the *Gaussian centres* (a Delaunay complex) rather than
    a regular grid, which adapts resolution to where the Gaussians actually
    are. That needs a Delaunay triangulation, and ``scipy`` is deliberately not
    a hard dependency of ``gsplat[mesh]`` -- see
    :func:`tetrahedralize_points`, which uses it when present.

    Args:
        bounds: ``(2, 3)`` -- ``[[min_x, min_y, min_z], [max_x, max_y, max_z]]``.
        resolution: Number of cells along the longest axis. Cells are cubic, so
            the other axes get however many cells fit.

    Returns:
        ``(vertices, tets)`` -- ``(V, 3)`` float64 positions and ``(T, 4)``
        int64 indices.

    Raises:
        ValueError: If the bounds are degenerate or the resolution is below 1.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError(f"bounds must have shape (2, 3), got {bounds.shape}")
    extent = bounds[1] - bounds[0]
    if not np.all(np.isfinite(extent)) or np.any(extent <= 0):
        raise ValueError(
            f"bounds must describe a positive, finite box; got min={bounds[0]} "
            f"max={bounds[1]} (extent {extent}). An empty or inverted box is "
            "usually a sign the caller computed it from an empty point set."
        )
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}")

    cell = float(extent.max()) / resolution
    counts = np.maximum(np.ceil(extent / cell).astype(int), 1)

    axes = [bounds[0][i] + np.arange(counts[i] + 1) * cell for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    vertices = grid.reshape(-1, 3)

    shape = tuple(counts + 1)

    def index(i, j, k):
        return np.ravel_multi_index((i, j, k), shape)

    i, j, k = np.meshgrid(
        np.arange(counts[0]), np.arange(counts[1]), np.arange(counts[2]), indexing="ij"
    )
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    # The eight corners of every cube, in the canonical order the six-tet
    # decomposition below is written against.
    c = [
        index(i, j, k),
        index(i + 1, j, k),
        index(i + 1, j + 1, k),
        index(i, j + 1, k),
        index(i, j, k + 1),
        index(i + 1, j, k + 1),
        index(i + 1, j + 1, k + 1),
        index(i, j + 1, k + 1),
    ]
    # Freudenthal: all six tets share the cube's main diagonal 0-6, which is
    # what makes adjacent cubes agree on their shared faces.
    pattern = [
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ]
    tets = np.stack(
        [np.stack([c[a], c[b], c[d], c[e]], axis=-1) for a, b, d, e in pattern],
        axis=1,
    ).reshape(-1, 4)
    return vertices, tets.astype(np.int64)


def tetrahedralize_points(points: np.ndarray):
    """Delaunay tetrahedralisation of arbitrary points, if scipy is available.

    This is what GOF actually uses -- the Delaunay complex of the Gaussian
    centres -- so the volume is dense where the scene is and empty where it is
    not, instead of paying for a uniform grid over the bounding box.

    ``scipy`` is an *optional accelerator* in this package, never a hard
    import (see ``docs/handoff/SCOPE.md``), so this raises a clear
    ``ImportError`` rather than being unavailable silently. Callers that must
    work without it should use :func:`tetrahedral_grid`.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "tetrahedralize_points needs scipy (an optional dependency here). "
            "Install it, or use tetrahedral_grid for a regular grid instead."
        ) from error

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3), got {points.shape}")
    if len(points) < 4:
        raise ValueError(f"need at least 4 points to tetrahedralise, got {len(points)}")
    return points, Delaunay(points).simplices.astype(np.int64)


def marching_tetrahedra(
    vertices: np.ndarray, tets: np.ndarray, values: np.ndarray, level: float = 0.0
):
    """The ``level`` iso-surface of a scalar field sampled on a tet mesh.

    Pure NumPy and fully vectorised: every tet is classified by the sign
    pattern of its four values, and the surface crosses each edge that joins an
    inside vertex to an outside one, at the point linear interpolation puts it.

    Duplicate vertices are merged by the edge they came from rather than by
    position, so the result is topologically welded exactly -- comparing
    floating-point coordinates would leave hairline cracks.

    Args:
        vertices: ``(V, 3)`` tet-mesh vertex positions.
        tets: ``(T, 4)`` vertex indices.
        values: ``(V,)`` field value at each vertex.
        level: The iso-value to extract.

    Returns:
        ``(surface_vertices, faces)``. Both are empty arrays with the right
        shape when the field does not cross ``level``, which is a legitimate
        answer and not an error -- see :func:`probe_field` for diagnosing it.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"got {values.shape[0]} field values for {vertices.shape[0]} "
            "vertices; they must correspond one-to-one"
        )

    inside = values < level
    codes = (
        inside[tets[:, 0]].astype(np.int64)
        | (inside[tets[:, 1]].astype(np.int64) << 1)
        | (inside[tets[:, 2]].astype(np.int64) << 2)
        | (inside[tets[:, 3]].astype(np.int64) << 3)
    )

    crossing = np.nonzero((codes != 0) & (codes != 15))[0]
    if crossing.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    # One (N, 3, 2) block per (case, triangle): N tets share a case, and each
    # produces one triangle whose three corners are the three edge keys.
    #
    # Laying this out per-edge instead -- appending all N copies of corner 0,
    # then all N of corner 1 -- and reshaping to (-1, 3) at the end silently
    # builds triangles from three *different tets'* first corners. The vertex
    # positions still come out right, because they are interpolated per unique
    # edge and do not depend on the grouping, so a test that measures where the
    # vertices are cannot see it. Only a connectivity check can.
    blocks = []
    for code in np.unique(codes[crossing]):
        cases = _MT_CASES[int(code)]
        if not cases:
            continue
        selected = crossing[codes[crossing] == code]
        for tri in cases:
            corners = []
            for edge in tri:
                a = tets[selected, _TET_EDGES[edge][0]]
                b = tets[selected, _TET_EDGES[edge][1]]
                corners.append(np.stack([np.minimum(a, b), np.maximum(a, b)], -1))
            blocks.append(np.stack(corners, axis=1))  # (N, 3, 2)

    keys = np.concatenate(blocks, axis=0).reshape(-1, 2)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    v0 = values[unique_keys[:, 0]]
    v1 = values[unique_keys[:, 1]]
    denominator = v1 - v0
    # A zero denominator means both ends sit exactly at the level; the midpoint
    # is the only defensible answer and the alternative is a division by zero
    # that silently produces inf coordinates.
    t = np.where(
        np.abs(denominator) > 1e-30,
        (level - v0) / np.where(denominator == 0, 1, denominator),
        0.5,
    )
    t = np.clip(t, 0.0, 1.0)
    p0 = vertices[unique_keys[:, 0]]
    p1 = vertices[unique_keys[:, 1]]
    surface_vertices = p0 + t[:, None] * (p1 - p0)

    faces = inverse.reshape(-1, 3)
    return surface_vertices, faces.astype(np.int64)


def probe_field(
    field_fn: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    level: float = 0.0,
    resolution: int = 16,
) -> dict:
    """Sample the field coarsely and report whether an extraction can work.

    **Run this before a full extraction on a GPU.** The three ways a level-set
    extraction returns nothing useful are all visible from a coarse sample, and
    all of them are silent otherwise:

    - the level sits outside the field's range, so nothing crosses it;
    - the bounding box misses the surface, so the field has one sign throughout;
    - the field is constant, all-NaN, or otherwise degenerate.

    Cost is one field evaluation at ``resolution**3`` points, which is cheap
    next to the extraction it protects.

    Returns:
        A dict with the field's range and quantiles, the fraction of samples
        below ``level``, a ``crossable`` flag, ``suggested_level`` (the median,
        which is the level most likely to produce a surface if the requested
        one cannot), and a list of human-readable ``warnings``.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    vertices, _tets = tetrahedral_grid(bounds, resolution)
    started = time.perf_counter()
    values = np.asarray(field_fn(vertices), dtype=np.float64).reshape(-1)
    elapsed = time.perf_counter() - started
    if values.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"field_fn returned {values.shape[0]} values for "
            f"{vertices.shape[0]} query points; it must return one per point"
        )

    finite = np.isfinite(values)
    messages = []
    report = {
        "num_samples": int(values.size),
        "num_finite": int(finite.sum()),
        "seconds": float(elapsed),
        "level": float(level),
        "bounds": bounds.tolist(),
        "resolution": int(resolution),
    }

    if not finite.any():
        messages.append(
            "every sampled field value is NaN or infinite -- field_fn is "
            "returning nothing usable, so no level can be extracted."
        )
        report.update(
            {"crossable": False, "warnings": messages, "suggested_level": None}
        )
        return report

    usable = values[finite]
    below = float((usable < level).mean())
    report.update(
        {
            "min": float(usable.min()),
            "max": float(usable.max()),
            "mean": float(usable.mean()),
            "median": float(np.median(usable)),
            "p01": float(np.quantile(usable, 0.01)),
            "p99": float(np.quantile(usable, 0.99)),
            "fraction_below_level": below,
            "suggested_level": float(np.median(usable)),
        }
    )

    crossable = bool(usable.min() < level < usable.max())
    report["crossable"] = crossable

    if not finite.all():
        messages.append(
            f"{100 * (1 - finite.mean()):.1f}% of sampled field values are NaN "
            "or infinite; those regions cannot contribute a surface."
        )
    if not crossable:
        messages.append(
            f"level={level} is outside the sampled field range "
            f"[{usable.min():.6g}, {usable.max():.6g}], so the extraction will "
            f"return an empty mesh. Either the bounds miss the surface, or the "
            f"level is wrong -- the field's median is {np.median(usable):.6g}."
        )
    elif below < 1e-3 or below > 1 - 1e-3:
        messages.append(
            f"only {100 * min(below, 1 - below):.3f}% of samples fall on one "
            "side of the level, so the surface is a sliver in these bounds. "
            "Tighten the bounds or reconsider the level."
        )
    if float(usable.max() - usable.min()) < 1e-12:
        messages.append(
            "the field is constant over these bounds; there is no surface to "
            "extract anywhere in them."
        )

    report["warnings"] = messages
    return report


def _orient_against_field(
    mesh, field_fn: Callable[[np.ndarray], np.ndarray], level: float, step: float
) -> dict:
    """Make face normals point toward *increasing* field, and say how surely.

    ``marching_tetrahedra`` does not trust its own winding (see ``_MT_CASES``),
    so orientation is established here against the field itself: step a little
    either side of each face along its normal and check which side is larger.
    With ``inside = value < level`` the outward normal is the direction the
    field grows in.

    A confidence well below 1.0 means the two sides disagree, which usually
    means ``step`` is larger than the field's own features.
    """
    o3d = _require_open3d()

    mesh.orient_triangles()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return {"flipped": False, "orientation_confidence": None}

    vertices = np.asarray(mesh.vertices)
    centroids = vertices[triangles].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)

    # A bounded sample: orientation is a single global bit, so there is no need
    # to pay for every face on a large mesh.
    if len(centroids) > 4096:
        stride = len(centroids) // 4096
        centroids = centroids[::stride]
        normals = normals[::stride]

    outward = np.asarray(field_fn(centroids + normals * step), dtype=np.float64)
    inward = np.asarray(field_fn(centroids - normals * step), dtype=np.float64)
    valid = np.isfinite(outward) & np.isfinite(inward)
    if not valid.any():
        return {"flipped": False, "orientation_confidence": None}

    votes = (outward[valid] > inward[valid]).mean()
    flipped = bool(votes < 0.5)
    if flipped:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    return {
        "flipped": flipped,
        "orientation_confidence": float(max(votes, 1.0 - votes)),
        "orientation_samples": int(valid.sum()),
    }


def level_set_residual(
    mesh, field_fn: Callable[[np.ndarray], np.ndarray], level: float = 0.0
) -> dict:
    """How far the extracted surface actually sits from ``level``.

    **The most useful check on a real capture, because it needs no ground
    truth.** Re-evaluate the field at the mesh's own vertices: a faithful
    extraction puts them all at ``level``, so the residual is near zero. A large
    residual means the extraction does not represent the field it came from --
    a resolution far too coarse for the field's features, a field that is not
    continuous, or a bug here.

    Report it in scene-relevant terms by comparing ``mean_abs`` against the
    grid's cell size: below the cell size is expected, far above is not.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return {"num_vertices": 0, "mean_abs": None, "max_abs": None}
    values = np.asarray(field_fn(vertices), dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return {"num_vertices": int(len(vertices)), "mean_abs": None, "max_abs": None}
    residual = np.abs(values[finite] - level)
    return {
        "num_vertices": int(len(vertices)),
        "num_finite": int(finite.sum()),
        "mean_abs": float(residual.mean()),
        "median_abs": float(np.median(residual)),
        "p95_abs": float(np.quantile(residual, 0.95)),
        "max_abs": float(residual.max()),
    }


def extract_level_set(
    field_fn: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    resolution: int = 128,
    level: float = 0.0,
    chunk_size: int = 1 << 20,
    clean: bool = True,
    orient: bool = True,
    diagnostics: Optional[dict] = None,
    debug_dir: Optional[str] = None,
    probe: bool = True,
):
    """Extract ``level`` of ``field_fn`` over ``bounds`` as a triangle mesh.

    The whole of this is CPU work except the calls to ``field_fn``, which is
    the one place a GPU is needed -- see :func:`gaussian_opacity_field`.

    Args:
        field_fn: Called with an ``(N, 3)`` array of world positions, returns
            ``(N,)`` field values. Values below ``level`` are inside. It is
            called in chunks, so it never sees more than ``chunk_size`` points
            at once and does not have to manage its own batching.
        bounds: ``(2, 3)`` box to extract within.
        resolution: Cells along the longest axis. Cost is cubic in this, and so
            is memory; start at 64 on a new scene and check the residual before
            raising it.
        level: The iso-value.
        chunk_size: Query points per ``field_fn`` call.
        clean: Remove degenerate triangles and duplicate vertices afterwards.
        orient: Point face normals toward increasing field -- see
            :func:`_orient_against_field`. Costs two extra small field
            evaluations.
        diagnostics: If given, a dict filled in place with per-stage counts,
            timings, field statistics and ``warnings``. **Read ``warnings``
            first when something looks wrong.**
        debug_dir: If given, a directory to write intermediate state into: the
            grid points coloured by field value (``field_samples.ply``), the
            raw pre-cleanup surface (``raw_surface.ply``), and the field values
            themselves (``field_values.npy``). Costs disk and a little time;
            intended for a GPU run that produced something surprising.
        probe: Run :func:`probe_field` first and fold its warnings in. Cheap
            relative to the extraction and it explains empty results.

    Returns:
        ``(mesh, stats)``. An empty mesh is a legitimate result when the field
        does not cross ``level``; ``stats["warnings"]`` says why.
    """
    o3d = _require_open3d()

    bounds = np.asarray(bounds, dtype=np.float64)
    stats: dict = {"warnings": [], "level": float(level), "resolution": int(resolution)}
    timings: dict = {}

    if probe:
        started = time.perf_counter()
        # A coarse probe: cheap next to the real thing, and it is what turns
        # "empty mesh, no explanation" into a sentence naming the cause.
        report = probe_field(
            field_fn, bounds, level=level, resolution=min(resolution, 16)
        )
        timings["probe"] = time.perf_counter() - started
        stats["probe"] = report
        stats["warnings"].extend(report.get("warnings", []))

    started = time.perf_counter()
    vertices, tets = tetrahedral_grid(bounds, resolution)
    timings["grid"] = time.perf_counter() - started
    cell_size = float((bounds[1] - bounds[0]).max()) / resolution
    stats.update(
        {
            "num_grid_vertices": int(len(vertices)),
            "num_tetrahedra": int(len(tets)),
            "cell_size": cell_size,
            "bounds": bounds.tolist(),
        }
    )

    started = time.perf_counter()
    values = np.empty(len(vertices), dtype=np.float64)
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        chunk = np.asarray(field_fn(vertices[start:stop]), dtype=np.float64).reshape(-1)
        if chunk.shape[0] != stop - start:
            raise ValueError(
                f"field_fn returned {chunk.shape[0]} values for {stop - start} "
                "query points; it must return exactly one value per point"
            )
        values[start:stop] = chunk
    timings["field"] = time.perf_counter() - started

    finite = np.isfinite(values)
    stats["field"] = {
        "num_finite": int(finite.sum()),
        "min": float(values[finite].min()) if finite.any() else None,
        "max": float(values[finite].max()) if finite.any() else None,
        "fraction_below_level": (
            float((values[finite] < level).mean()) if finite.any() else None
        ),
    }
    if not finite.all():
        # A non-finite value would classify as "outside" and quietly punch a
        # hole, so it is named rather than left to be discovered as a gap.
        stats["warnings"].append(
            f"{int((~finite).sum())} of {len(values)} field samples are NaN or "
            "infinite; they are treated as outside the surface, which can "
            "leave holes."
        )

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, "field_values.npy"), values)
        np.save(os.path.join(debug_dir, "grid_bounds.npy"), bounds)
        cloud = o3d.geometry.PointCloud()
        # Subsample: a full grid at resolution 128 is two million points, which
        # no viewer wants and which is not more informative than a slice of it.
        stride = max(1, len(vertices) // 200_000)
        sample = vertices[::stride]
        sample_values = values[::stride]
        cloud.points = o3d.utility.Vector3dVector(sample)
        span = np.nanmax(sample_values) - np.nanmin(sample_values)
        normalised = (
            np.zeros_like(sample_values)
            if not np.isfinite(span) or span <= 0
            else np.clip((sample_values - np.nanmin(sample_values)) / span, 0, 1)
        )
        normalised = np.nan_to_num(normalised, nan=0.5)
        # Red below the level, blue above -- so the surface is where the colour
        # changes, and a box that misses it is uniformly one colour.
        inside = np.nan_to_num(sample_values, nan=level) < level
        colors = np.zeros((len(sample), 3))
        colors[inside] = np.stack(
            [np.ones(inside.sum()), normalised[inside], normalised[inside]], -1
        )
        colors[~inside] = np.stack(
            [normalised[~inside], normalised[~inside], np.ones((~inside).sum())], -1
        )
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(debug_dir, "field_samples.ply"), cloud)

    started = time.perf_counter()
    surface_vertices, faces = marching_tetrahedra(vertices, tets, values, level)
    timings["marching"] = time.perf_counter() - started

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(surface_vertices),
        o3d.utility.Vector3iVector(faces),
    )
    stats["num_raw_vertices"] = int(len(surface_vertices))
    stats["num_raw_triangles"] = int(len(faces))

    if debug_dir is not None and len(faces) > 0:
        o3d.io.write_triangle_mesh(os.path.join(debug_dir, "raw_surface.ply"), mesh)

    if len(faces) == 0:
        stats["warnings"].append(
            "the field does not cross the level anywhere in these bounds, so "
            "the surface is empty. stats['probe'] says which of the level, the "
            "bounds or the field is responsible."
        )
        # The empty case must report the same keys as the successful one. It is
        # the case a caller is looking at *because* something went wrong, so it
        # is the worst possible one to make them handle a KeyError in.
        stats.update(
            {
                "num_vertices": 0,
                "num_triangles": 0,
                "is_watertight": False,
                "is_edge_manifold": True,
                "flipped": False,
                "orientation_confidence": None,
                "residual": level_set_residual(mesh, field_fn, level),
            }
        )
        stats["timings"] = timings
        if diagnostics is not None:
            diagnostics.update(stats)
        return mesh, stats

    if clean:
        started = time.perf_counter()
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        timings["clean"] = time.perf_counter() - started

    if orient:
        started = time.perf_counter()
        # Step a fraction of a cell: far enough to see the field change, near
        # enough not to cross to the other side of a thin feature.
        orientation = _orient_against_field(mesh, field_fn, level, 0.25 * cell_size)
        timings["orient"] = time.perf_counter() - started
        stats.update(orientation)
        confidence = orientation.get("orientation_confidence")
        if confidence is not None and confidence < 0.9:
            stats["warnings"].append(
                f"face orientation is only {confidence:.0%} consistent with the "
                "field gradient, so the mesh may be partly inside out. That "
                "usually means the probe step (a quarter cell) is larger than "
                "the field's features -- raise the resolution."
            )
    else:
        mesh.compute_vertex_normals()

    stats["num_vertices"] = int(len(mesh.vertices))
    stats["num_triangles"] = int(len(mesh.triangles))
    stats["is_watertight"] = bool(mesh.is_watertight())
    stats["is_edge_manifold"] = bool(mesh.is_edge_manifold())

    started = time.perf_counter()
    residual = level_set_residual(mesh, field_fn, level)
    timings["residual"] = time.perf_counter() - started
    stats["residual"] = residual
    if residual.get("mean_abs") is not None:
        # Scale-free: the extraction is only as precise as one cell, so the
        # residual is read against the cell and not in raw field units.
        stats["residual"]["mean_abs_over_cell"] = residual["mean_abs"] / cell_size
        if residual["mean_abs"] > cell_size:
            stats["warnings"].append(
                f"the extracted surface sits {residual['mean_abs']:.4g} from "
                f"the level on average, more than one cell ({cell_size:.4g}). "
                "The field is varying faster than the grid can follow; raise "
                "the resolution."
            )

    stats["timings"] = timings
    if diagnostics is not None:
        diagnostics.update(stats)
    return mesh, stats


def gaussian_opacity_field(
    splats,
    device: str = "cuda",
    max_gaussians_per_chunk: int = 1 << 14,
    cutoff_sigmas: float = 3.0,
):
    """Build the field callback for a trained Gaussian scene. **GPU; untested.**

    This is the only part of this module that needs a GPU, and the only part
    that has never been executed -- see the module docstring. It is kept
    deliberately thin so that everything which *can* be checked on CPU is on
    the other side of the callback boundary.

    The field returned is ``cutoff - opacity(x)``, so that it is **negative
    inside** the object, matching the ``value < level`` convention the rest of
    this module uses. Opacity at a point is the standard Gaussian evaluation
    accumulated over the scene:

        opacity(x) = 1 - prod_i (1 - o_i * exp(-0.5 * d_i(x)^T S_i^-1 d_i(x)))

    which is GOF's "opacity field" in the simplest form -- it ignores the
    view-dependent ray-space refinement the paper adds, and a level set of this
    is therefore an approximation of theirs, not a reimplementation.

    Args:
        splats: A dict of tensors as saved by ``simple_trainer*.py`` --
            ``means``, ``quats``, and **unactivated** ``scales`` and
            ``opacities``, matching what :func:`extract_mesh_tsdf` consumes.
        device: Torch device to evaluate on.
        max_gaussians_per_chunk: Gaussians per inner batch. Cost is
            ``points x gaussians``, so this bounds peak memory.
        cutoff_sigmas: Gaussians further than this many standard deviations
            from a query point contribute essentially nothing and are skipped.

    Returns:
        ``(field_fn, info)`` -- a callable for :func:`extract_level_set`, and a
        dict describing the scene (Gaussian count, the bounding box of the
        means, and a suggested ``bounds`` to extract within).

    **Before spending GPU time**, call :func:`validate_level_set_pipeline` to
    confirm this module works in your environment, then :func:`probe_field` on
    the returned callable to confirm the level and bounds are sane. Those two
    calls cost seconds and rule out most of what goes wrong.
    """
    import torch

    required = ("means", "quats", "scales", "opacities")
    missing = [k for k in required if k not in splats]
    if missing:
        raise ValueError(
            f"splats is missing {missing}; this needs the raw "
            f"{list(required)} a trainer saves. Appearance-embedding "
            "checkpoints are out of scope here, as they are for TSDF."
        )

    means = splats["means"].to(device).float()
    # `scales` and `opacities` are stored unactivated, exactly as
    # `extract_mesh_tsdf` assumes; activating twice is a silent and very
    # confusing error, so it happens once, here.
    scales = torch.exp(splats["scales"].to(device).float())
    opacities = torch.sigmoid(splats["opacities"].to(device).float()).reshape(-1)
    quats = splats["quats"].to(device).float()
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    radii = cutoff_sigmas * scales.max(dim=-1).values

    def field_fn(points: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
        transmittance = torch.ones(query.shape[0], device=device)
        for start in range(0, means.shape[0], max_gaussians_per_chunk):
            stop = min(start + max_gaussians_per_chunk, means.shape[0])
            delta = query[:, None, :] - means[None, start:stop, :]
            near = delta.norm(dim=-1) <= radii[None, start:stop]
            if not bool(near.any()):
                continue
            local = torch.einsum("pgi,gij->pgj", delta, rotation[start:stop])
            mahalanobis = ((local / scales[None, start:stop, :]) ** 2).sum(-1)
            contribution = opacities[None, start:stop] * torch.exp(-0.5 * mahalanobis)
            transmittance = transmittance * (1.0 - contribution * near).clamp(
                min=0.0, max=1.0
            ).prod(dim=1)
        opacity = 1.0 - transmittance
        return (0.5 - opacity).detach().cpu().numpy().astype(np.float64)

    with torch.no_grad():
        lower = (means - radii[:, None]).min(dim=0).values
        upper = (means + radii[:, None]).max(dim=0).values
    info = {
        "num_gaussians": int(means.shape[0]),
        "bounds": np.stack([lower.cpu().numpy(), upper.cpu().numpy()]).tolist(),
        "cutoff_sigmas": float(cutoff_sigmas),
        "suggested_level": 0.0,
        "executed": False,
        "note": (
            "This adapter has never been run against a real checkpoint. Treat "
            "any result from it as unverified until it has."
        ),
    }
    return field_fn, info


def validate_level_set_pipeline(resolution: int = 24, verbose: bool = True) -> dict:
    """Run the whole extraction against analytic fields. **Needs no GPU.**

    The first thing to run in a new environment, and the first thing to run
    when a real extraction misbehaves: if this fails, the problem is in this
    module rather than in your field or your checkpoint, and every number below
    has a closed-form answer to compare against.

    Checks a sphere (curved, closed), a plane through the box (open), and that
    halving the cell size roughly quarters the error -- linear interpolation on
    a smooth field is second-order accurate, so a *linear* improvement means
    something is wrong even though the surface may look fine.

    Returns:
        A dict of measurements with an overall ``passed`` flag and, when
        anything failed, ``failures`` naming which check and by how much.
    """
    results: dict = {"checks": {}, "failures": []}
    bounds = np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])

    def sphere(points):
        return np.linalg.norm(points, axis=1) - 1.0

    errors = {}
    for res in (resolution, resolution * 2):
        mesh, stats = extract_level_set(sphere, bounds, resolution=res, probe=False)
        radii = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
        error = float(np.abs(radii - 1.0).mean())
        errors[res] = error
        results["checks"][f"sphere_res_{res}"] = {
            "mean_radial_error": error,
            "num_triangles": stats["num_triangles"],
            "is_watertight": stats["is_watertight"],
            "surface_area_over_analytic": (
                float(mesh.get_surface_area() / (4.0 * np.pi))
            ),
            "residual_over_cell": stats["residual"].get("mean_abs_over_cell"),
        }
        if not stats["is_watertight"]:
            results["failures"].append(
                f"the sphere at resolution {res} came out not watertight, which "
                "means the tetrahedra do not agree on their shared faces"
            )

    ratio = errors[resolution] / max(errors[resolution * 2], 1e-30)
    results["checks"]["convergence_ratio"] = ratio
    if not 2.5 < ratio < 6.0:
        results["failures"].append(
            f"halving the cell size changed the error by {ratio:.2f}x, not the "
            "~4x expected of second-order accuracy; the interpolation along "
            "crossing edges is probably wrong"
        )

    # A plane: the surface is open, so it must *not* claim to be watertight,
    # and its area is known exactly.
    plane_mesh, plane_stats = extract_level_set(
        lambda p: p[:, 2], bounds, resolution=resolution, probe=False
    )
    area = float(plane_mesh.get_surface_area())
    results["checks"]["plane"] = {
        "area": area,
        "analytic_area": 9.0,
        "is_watertight": plane_stats["is_watertight"],
    }
    if abs(area - 9.0) > 0.05:
        results["failures"].append(
            f"the plane z=0 across a 3x3 box came out with area {area:.4f}, " "not 9.0"
        )

    # An empty result must be reported as such, with a reason, rather than
    # raising or returning something malformed.
    _empty, empty_stats = extract_level_set(
        sphere, bounds, resolution=8, level=99.0, probe=True
    )
    results["checks"]["empty_is_explained"] = {
        "num_triangles": empty_stats["num_triangles"],
        "has_warning": bool(empty_stats["warnings"]),
    }
    if empty_stats["num_triangles"] != 0 or not empty_stats["warnings"]:
        results["failures"].append(
            "an unreachable level did not produce an empty mesh with an "
            "explanatory warning"
        )

    results["passed"] = not results["failures"]
    if verbose:
        status = "PASSED" if results["passed"] else "FAILED"
        print(f"[level_set] self-test {status}")
        for name, value in results["checks"].items():
            print(f"[level_set]   {name}: {value}")
        for failure in results["failures"]:
            print(f"[level_set]   FAILURE: {failure}")
    return results
