"""Move mesh vertices to where the photographs agree the surface is.

Every geometry lever in this package is either *upstream* of the mesh (bundle
adjustment, dense MVS, the 2DGS depth used for TSDF) or *subtractive*
(:func:`cull_unobserved_faces`, :func:`simplify_mesh`). Nothing moves a vertex
to fit the images, so a TSDF surface is only ever as accurate as the voxel grid
it was fused on.

This is the refinement stage classical MVS pipelines end with -- Vu, Labatut,
Pons & Keriven, *High Accuracy and Visibility-Consistent Dense Multiview
Stereo* (TPAMI 2012), the same step OpenMVS ships as ``RefineMesh``. Each
vertex slides along its own normal to maximise multi-view photoconsistency,
regularised by a Laplacian term so the surface stays smooth where the images
say nothing.

**Why the objective is patchwise and normalised, and not a colour difference.**
`gsplat/photogrammetry/photometric_alignment.py` is a retained negative result:
an objective built on a single fused colour per surface point has its minimum
in the wrong place, because one colour cannot express how views legitimately
differ (pixel footprint, obliquity, exposure). This uses the classical answer
instead -- normalised cross-correlation over a small oriented patch. Because
each view's patch is z-normalised before comparison, a per-view gain and offset
cancel exactly, so exposure differences are *explained* rather than blamed on
geometry.

That objective was verified before any optimiser was written, which is the
lesson the alignment work paid for. On an analytic sphere with a 3.6-pixel
patch, cross-view disagreement is minimised exactly at the true surface:

    radial offset   -0.04    -0.02    -0.01     0.00    +0.01    +0.02
    disagreement   0.1499   0.0579   0.0176   0.0032   0.0697   0.1744

**It is worth running above about a third of a source pixel of error, and not
below.** The correction is proportional to the error; the cost is a roughly
fixed ~0.15 pixel of added noise. Measured, input against output error, both in
source pixels:

    input     0.00    0.24    0.48    0.95    1.91
    output    0.15    0.28    0.39    0.70    1.50

so it helps from about 0.4 px upward and hurts below. That is not a limitation
in practice: :func:`derive_tsdf_parameters` now sizes a voxel at one source
pixel, so a TSDF surface starts out around a pixel from the truth -- comfortably
inside the regime where this pays. It is opt-in regardless, and the CLI reports
the displacement it applied so the decision is checkable after the fact.

**It needs enough views, and the cliff is sharp.** Recovery of the same 0.95
pixel error, all else equal: 8 views recovers 4-12%, 10 views 26.7%, 12 views
24.3%. Below about ten views the cross-view constraint is too weak to separate
surface error from image noise, and the method spends its time moving vertices
without improving the fit -- which the CLI warns about, since the
photoconsistency it reports before and after says so directly.

**The patch size is load-bearing, and too large is not merely slower.** A flat
tangent patch is a chord of a curved surface, so an over-large patch fits best
slightly *inside* a convex object and refinement would shrink it. Measured, the
minimum moves off the truth as the patch grows: 3.6 px -> minimum at 0.000,
9 px -> -0.010, 18 px -> -0.020. The default spaces patch samples about one
source pixel apart, derived from the same ``depth / focal`` quantity
:func:`derive_tsdf_parameters` sizes a voxel with.
"""

from typing import Optional

import numpy as np
import torch

from ._open3d import _require_open3d


def _tangent_frame(normals: np.ndarray):
    """Two unit vectors spanning the plane perpendicular to each normal."""
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))
    degenerate = np.abs(normals[:, 2]) > 0.9
    reference[degenerate] = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normals, reference)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    return tangent, bitangent


def _mesh_edges(triangles: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle mesh, as an (E, 2) array."""
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _bilinear_gray(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear sample of a single-channel image.

    Same half-pixel convention as `texturing._bilinear`, so "where a point
    projects" means one thing across the package.
    """
    height, width = image.shape
    x = torch.clamp(uv[..., 0] - 0.5, 0.0, width - 1.0)
    y = torch.clamp(uv[..., 1] - 0.5, 0.0, height - 1.0)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    fx = x - x0.to(x.dtype)
    fy = y - y0.to(y.dtype)
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def refine_mesh_photometric(
    mesh,
    dataset,
    max_views: Optional[int] = None,
    iterations: int = 30,
    outer_rounds: int = 6,
    smoothness: float = 1.0,
    patch_radius: int = 2,
    patch_spacing: Optional[float] = None,
    max_displacement_pixels: float = 3.0,
    learning_rate: float = 1.0,
    min_views: int = 2,
    stats_out: Optional[dict] = None,
):
    """Slide vertices along their normals onto the photoconsistent surface.

    Args:
        mesh: An ``open3d.geometry.TriangleMesh``. Not modified; a refined copy
            is returned.
        dataset: An ``examples.datasets.colmap.Dataset``-like object -- see
            :func:`gsplat.photogrammetry.bake_texture`.
        max_views: If given, only the first ``max_views`` images are used.
        iterations: Optimiser steps per round.
        outer_rounds: How many times to recompute visibility and the tangent
            frames from the moved surface. Visibility is not differentiable, so
            it is held fixed within a round. Measured, 6 recovers meaningfully
            more than 3 (26.7% of the error against 22.2%) for twice the time.
        smoothness: Weight on the Laplacian term, which keeps the surface sane
            where the images carry no texture to fit. Measured, the result is
            largely insensitive to it (0.0 to 1.0 recover 23-27%), which says
            the binding constraint is the photometric objective's own precision
            rather than the regulariser.
        patch_radius: Patch half-width in samples; 2 gives a 5x5 patch.
        patch_spacing: World distance between patch samples. ``None`` derives
            it as one source pixel at the median viewing distance -- see the
            module docstring for why too large is a bias and not just a cost.
        max_displacement_pixels: Vertices are not allowed to move further than
            this many source pixels' worth of world distance, so a vertex in a
            textureless region cannot be dragged away by noise.
        learning_rate: Initial L-BFGS step; the line search rescales it.
        min_views: A vertex needs at least this many views to be refined at
            all. Two is the minimum that can disagree, but see the module
            docstring: the *capture* wants ten or more for this to pay.
        stats_out: If given, a dict updated in place with what was measured.

    Returns:
        ``(mesh, stats)`` -- a new mesh, and the displacement statistics.
    """
    o3d = _require_open3d()

    num_views = len(dataset) if max_views is None else min(max_views, len(dataset))
    if num_views < min_views:
        raise ValueError(
            f"Photometric mesh refinement needs at least {min_views} views to "
            f"compare, got {num_views}."
        )

    refined = o3d.geometry.TriangleMesh(mesh)
    refined.compute_vertex_normals()
    triangles = np.asarray(refined.triangles)
    base = np.asarray(refined.vertices, dtype=np.float64).copy()
    if len(base) == 0 or len(triangles) == 0:
        raise ValueError("Photometric mesh refinement needs a non-empty mesh.")

    grays, Ks, camtoworlds = [], [], []
    for i in range(num_views):
        item = dataset[i]
        image = np.asarray(item["image"], dtype=np.float64) / 255.0
        if image.ndim == 3:
            image = image.mean(axis=-1)
        grays.append(torch.tensor(image, dtype=torch.float64))
        Ks.append(np.asarray(item["K"], dtype=np.float64))
        camtoworlds.append(np.asarray(item["camtoworld"], dtype=np.float64))

    # One source pixel at the median viewing distance -- the same quantity
    # `derive_tsdf_parameters` sizes a voxel with, for the same reason.
    centre = base.mean(axis=0)
    distances = [np.linalg.norm(c[:3, 3] - centre) for c in camtoworlds]
    focal = float(np.mean([0.5 * (K[0, 0] + K[1, 1]) for K in Ks]))
    pixel_world = float(np.median(distances)) / max(focal, 1e-9)
    if patch_spacing is None:
        patch_spacing = pixel_world
    max_displacement = max_displacement_pixels * pixel_world

    offsets = np.stack(
        np.meshgrid(
            np.arange(-patch_radius, patch_radius + 1),
            np.arange(-patch_radius, patch_radius + 1),
        ),
        axis=-1,
    ).reshape(-1, 2)
    offsets_t = torch.tensor(offsets, dtype=torch.float64)

    edges = _mesh_edges(triangles)
    edge_a = torch.tensor(edges[:, 0], dtype=torch.long)
    edge_b = torch.tensor(edges[:, 1], dtype=torch.long)

    displacement = torch.zeros(len(base), dtype=torch.float64, requires_grad=True)
    base_t = torch.tensor(base, dtype=torch.float64)

    history = []
    for _round in range(max(outer_rounds, 1)):
        with torch.no_grad():
            current = base + displacement.numpy()[:, None] * np.asarray(
                refined.vertex_normals, dtype=np.float64
            )
        moved = o3d.geometry.TriangleMesh(refined)
        moved.vertices = o3d.utility.Vector3dVector(current)
        moved.compute_vertex_normals()
        normals = np.asarray(moved.vertex_normals, dtype=np.float64)
        tangent, bitangent = _tangent_frame(normals)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(moved))

        # Visibility is not differentiable, so it is fixed for the round. The
        # test is strict -- the vertex must *be* the first hit, and face the
        # camera -- because a loose tolerance lets a point just under the
        # surface count as visible, which silently flattens the objective.
        view_masks = []
        for i in range(num_views):
            camera = camtoworlds[i][:3, 3]
            delta = current - camera
            distance = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(distance, 1e-12)[:, None]
            rays = np.concatenate(
                [np.broadcast_to(camera, direction.shape), direction], axis=-1
            ).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            facing = (normals * -direction).sum(axis=1) > 0.2
            view_masks.append(
                np.isfinite(hit) & (np.abs(hit - distance) < 2e-3 * distance) & facing
            )

        seen_count = np.sum(view_masks, axis=0)
        active = seen_count >= min_views
        if not active.any():
            raise ValueError(
                "No vertex is visible from two or more views, so there is "
                "nothing to compare. The mesh and the cameras may be in "
                "different coordinate frames."
            )

        normals_t = torch.tensor(normals, dtype=torch.float64)
        tangent_t = torch.tensor(tangent, dtype=torch.float64)
        bitangent_t = torch.tensor(bitangent, dtype=torch.float64)
        active_t = torch.tensor(np.nonzero(active)[0], dtype=torch.long)
        masks_t = [torch.tensor(m[active], dtype=torch.bool) for m in view_masks]

        def objective():
            positions = base_t + displacement.unsqueeze(-1) * normals_t
            centres = positions[active_t]
            patch = (
                centres.unsqueeze(1)
                + patch_spacing
                * offsets_t[:, 0].reshape(1, -1, 1)
                * tangent_t[active_t].unsqueeze(1)
                + patch_spacing
                * offsets_t[:, 1].reshape(1, -1, 1)
                * bitangent_t[active_t].unsqueeze(1)
            )
            flat = patch.reshape(-1, 3)

            normalised = []
            valid = []
            for i in range(num_views):
                c2w = torch.tensor(camtoworlds[i], dtype=torch.float64)
                K = torch.tensor(Ks[i], dtype=torch.float64)
                local = (flat - c2w[:3, 3]) @ c2w[:3, :3]
                depth = local[:, 2].clamp_min(1e-6)
                uv = (local @ K.transpose(0, 1))[:, :2] / depth.unsqueeze(-1)
                sampled = _bilinear_gray(grays[i], uv).reshape(centres.shape[0], -1)
                # z-normalise each patch: this is what makes the comparison a
                # correlation, so a per-view gain and offset cancel exactly.
                mean = sampled.mean(dim=1, keepdim=True)
                std = sampled.std(dim=1, keepdim=True).clamp_min(1e-3)
                normalised.append((sampled - mean) / std)
                valid.append(masks_t[i])

            stack = torch.stack(normalised, dim=0)
            weight = torch.stack(valid, dim=0).to(stack.dtype).unsqueeze(-1)
            count = weight.sum(dim=0).clamp_min(1.0)
            mean_patch = (stack * weight).sum(dim=0) / count
            spread = (((stack - mean_patch) ** 2) * weight).sum(dim=0) / count
            enough = (weight.squeeze(-1).sum(dim=0) >= min_views).to(stack.dtype)
            photo = (spread.mean(dim=1) * enough).sum() / enough.sum().clamp_min(1.0)

            # Laplacian over mesh edges, in displacement rather than position:
            # it resists *changes* in shape, so it does not fight the geometry
            # the mesh already has.
            lap = ((displacement[edge_a] - displacement[edge_b]) ** 2).mean()
            return photo + smoothness * lap

        optimizer = torch.optim.LBFGS(
            [displacement],
            lr=learning_rate,
            max_iter=iterations,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        with torch.no_grad():
            before = float(objective().item())
        optimizer.step(closure)
        with torch.no_grad():
            displacement.clamp_(-max_displacement, max_displacement)
            after = float(objective().item())
        history.append({"before": before, "after": after})

    with torch.no_grad():
        final = displacement.numpy()
    result = o3d.geometry.TriangleMesh(refined)
    result.vertices = o3d.utility.Vector3dVector(
        base + final[:, None] * np.asarray(refined.vertex_normals, dtype=np.float64)
    )
    result.compute_vertex_normals()

    stats = {
        "num_vertices": int(len(base)),
        "num_views": int(num_views),
        "patch_spacing": float(patch_spacing),
        "pixel_world_size": float(pixel_world),
        "max_displacement": float(max_displacement),
        "mean_abs_displacement": float(np.abs(final).mean()),
        "max_abs_displacement": float(np.abs(final).max()),
        "rounds": history,
        "photoconsistency_before": history[0]["before"],
        "photoconsistency_after": history[-1]["after"],
    }
    if stats_out is not None:
        stats_out.update(stats)
    return result, stats
