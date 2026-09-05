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
"""End-to-end tests that actually run ``examples/extract_mesh.py``'s ``main()``.

**This file exists because nothing ever called it.** ``extract_mesh.py`` opened
with a blanket ``assert cfg.ckpt`` that ran *before* the method dispatch, so
reaching any of its ~ten guards, or the delivery path they protect, needed a
trained GPU checkpoint. The existing tests could only get close:
``tests/test_photogrammetry_pipeline.py`` compares forwarded flag *strings*
against ``Config``'s field names, and ``tests/test_extract_mesh_io.py`` imports
a single helper. So a call passing ``seam_smoothness=`` to a
``bake_mesh_texture()`` that had no such parameter sat on the default path,
crashing **every** texture-baking run of the script with ``TypeError``, and the
whole suite stayed green.

That is the fifth instance of the failure mode ``docs/handoff/ISSUES.md`` § 5
documents -- a mechanism proved to work while its call site went unpinned -- and
the first that was a hard crash. These tests drive the script itself.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _examples_on_path():
    """Import the example scripts the way the CLI runs them.

    They import `datasets.colmap` relative to `examples/`, so that directory
    has to be importable, not just the repo root.
    """
    pytest.importorskip("open3d", reason="open3d not installed")
    pytest.importorskip("pycolmap", reason="pycolmap not installed")
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("tyro", reason="tyro not installed")
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """One small on-disk capture, shared by every test in this file.

    Deliberately tiny -- 6 views at 64x64 over a few hundred triangles -- so the
    whole delivery path runs inside the suite's existing per-test time budget
    (``tests/test_texturing.py``'s slowest test is 4.2s). Module-scoped because
    building it is the expensive part and nothing here mutates it.
    """
    _examples_on_path()
    import make_synthetic_capture

    data_dir = tmp_path_factory.mktemp("capture")
    summary = make_synthetic_capture.main(
        make_synthetic_capture.Config(
            data_dir=str(data_dir),
            shape="sphere_on_plane",
            resolution=8,
            num_views=6,
            width=64,
            height=64,
            # Exposure drift is what seam levelling exists to remove; without
            # it the levelling stage runs against nothing and the delivery
            # path's most fragile step is exercised vacuously.
            exposure=0.15,
            num_dense_points=4000,
            num_sparse_points=300,
        )
    )
    return summary


@pytest.fixture(scope="module")
def misregistered(tmp_path_factory):
    """The same capture, with the reported poses perturbed by 45'.

    `pose_error_arcmin` moves the pose that is *written to the COLMAP model*
    and leaves the rendered image alone, which is exactly residual SfM error --
    the pose you have does not quite match the photograph it belongs to.
    """
    _examples_on_path()
    import make_synthetic_capture

    data_dir = tmp_path_factory.mktemp("misregistered")
    return make_synthetic_capture.main(
        make_synthetic_capture.Config(
            data_dir=str(data_dir),
            shape="sphere",
            resolution=12,
            num_views=8,
            width=96,
            height=96,
            pose_error_arcmin=45.0,
            num_dense_points=2000,
            num_sparse_points=300,
        )
    )


def _config(capture, tmp_path, **overrides):
    """An `extract_mesh.Config` pointed at the capture, on the mesh entry."""
    import extract_mesh

    kwargs = dict(
        data_dir=capture["data_dir"],
        data_factor=1,
        # Keep every rendered view in the "train" split the bake reads.
        test_every=10_000,
        mesh_path=capture["mesh_path"],
        result_dir=str(tmp_path / "out"),
        device="cpu",
    )
    kwargs.update(overrides)
    return extract_mesh.Config(**kwargs)


# ---------------------------------------------------------------------------
# The test that would have caught the crash
# ---------------------------------------------------------------------------


def test_the_whole_delivery_path_runs_and_writes_readable_files(capture, tmp_path):
    """Drive `main()` with the full delivery flag set, then read the result back.

    This is the call-site pin. Every flag here was previously *reviewed only* --
    ``docs/handoff/PROGRESS.md`` listed roughly ten ``extract_mesh.py`` guards
    as never executed, because none was reachable without a checkpoint.

    ``--texture_seam_smoothness`` is the one that mattered: it reached
    ``bake_mesh_texture(seam_smoothness=...)``, which took no such argument, so
    this exact combination raised ``TypeError: bake_mesh_texture() got an
    unexpected keyword argument 'seam_smoothness'`` on the default path.
    """
    _examples_on_path()
    import cv2
    import imageio.v2 as imageio
    import open3d as o3d

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        cull_unobserved=True,
        texture_mode="atlas",
        texture_view_selection=True,
        texture_seam_smoothness=0.1,
        texture_size=128,
        # Loose enough that decimation actually runs. The mesh is the very
        # surface the dense cloud was sampled from, so a ratio of 1.0 is
        # already missed before any decimation -- culling removes the unseen
        # underside while the cloud keeps its points there.
        target_fit_ratio=4.0,
        dense_points=capture["dense_path"],
        normal_map=True,
        normal_map_bits=16,
        ao_map=True,
        ao_samples=8,
    )
    extract_mesh.main(cfg)

    out = Path(cfg.result_dir)
    obj, mtl = out / "mesh.obj", out / "mesh.mtl"
    normal_png, ao_png = out / "mesh_normal.png", out / "mesh_ao.png"
    for path in (obj, mtl, normal_png, ao_png, out / "mesh_metrics.json"):
        assert path.exists(), f"{path.name} was not written"

    # The mesh reads back with its UVs and its texture attached.
    mesh = o3d.io.read_triangle_mesh(str(obj), True)
    assert len(mesh.triangles) > 0
    assert len(mesh.triangle_uvs) == 3 * len(mesh.triangles)
    assert len(mesh.textures) > 0

    # The 16-bit normal map is genuinely 16-bit *on disk*. Pillow cannot write
    # 16-bit RGB PNG at all, so this path goes through OpenCV -- see
    # `tests/test_extract_mesh_io.py`.
    #
    # Read the depth out of the PNG header rather than trusting a decoder:
    # `imageio.imread` silently hands back uint8 for this exact file, so an
    # assertion phrased on its dtype fails on a perfectly good 16-bit map. The
    # header is bytes 24 (bit depth) and 25 (colour type, 2 = RGB) of IHDR.
    header = normal_png.read_bytes()[:26]
    assert header[24] == 16, f"normal map is {header[24]}-bit, expected 16"
    assert header[25] == 2, f"normal map colour type {header[25]}, expected RGB"
    normal = cv2.imread(str(normal_png), cv2.IMREAD_UNCHANGED)
    assert normal.dtype == np.uint16, normal.dtype
    assert normal.shape == (128, 128, 3), normal.shape

    ao = imageio.imread(str(ao_png))
    assert ao.shape[:2] == (128, 128)
    # AO on a sphere resting on a plane must actually occlude something: the
    # contact region is what the map is for. A uniformly white map would mean
    # the rays never hit anything.
    assert ao.min() < 250, f"nothing was occluded (min={ao.min()})"

    # And the .mtl references both extra maps, or they ship as orphan files
    # that every importer ignores.
    mtl_text = mtl.read_text()
    assert "mesh_normal.png" in mtl_text
    assert "mesh_ao.png" in mtl_text

    import json

    stats = json.loads((out / "mesh_metrics.json").read_text())
    for key in (
        "culling",
        "decimation",
        "normal_map",
        "ambient_occlusion",
        "view_selection",
        "point_to_mesh",
    ):
        assert key in stats, f"{key} missing from mesh_metrics.json"
    assert (
        stats["decimation"]["triangles_after"] < stats["decimation"]["triangles_before"]
    )
    assert stats["culling"]["num_culled"] > 0


def test_seam_smoothness_reaches_the_bake_from_the_cli(capture, tmp_path):
    """The specific crash, at the specific call site, with the flag set.

    Narrower than the test above and kept separate on purpose: that one would
    also fail if the AO map or the normal map broke, so it cannot say *this*
    regression came back. This one asserts the levelling stage actually ran,
    which is only observable because the CLI writes its stats out.
    """
    _examples_on_path()
    import json

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        texture_mode="atlas",
        texture_view_selection=True,
        texture_seam_smoothness=0.25,
        texture_size=128,
    )
    extract_mesh.main(cfg)

    stats = json.loads((Path(cfg.result_dir) / "mesh_metrics.json").read_text())
    levelling = stats["view_selection"]["seam_levelling"]
    assert levelling["num_seam_edges"] > 0
    assert "seam_discontinuity_before" in stats["view_selection"]


def test_photometric_alignment_runs_from_the_cli_and_reports_itself(capture, tmp_path):
    """`--photometric_align` end to end, with its numbers in mesh_metrics.json.

    The library function has its own tests
    (`tests/test_photometric_alignment.py`); this pins the *call site*, which is
    the thing that has broken five times on this branch. A flag accepted by the
    CLI and then dropped, mis-spelled into the callee, or run against the
    original dataset instead of the re-posed one would pass every one of those
    tests and do nothing here.
    """
    _examples_on_path()
    import json

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        texture_mode="atlas",
        texture_size=64,
        photometric_align=True,
        # Small: this is a wiring test, not a measurement of the method.
        photometric_align_levels=2,
        photometric_align_iters=15,
        photometric_align_rounds=1,
    )
    extract_mesh.main(cfg)

    stats = json.loads((Path(cfg.result_dir) / "mesh_metrics.json").read_text())
    alignment = stats["photometric_alignment"]
    assert alignment["num_observations"] > 0
    assert alignment["num_views"] == alignment["num_views"]
    for key in (
        "mean_photometric_residual_before",
        "mean_photometric_residual_after",
        "mean_pose_correction_arcmin",
    ):
        assert key in alignment, key
    # The cameras actually moved: a no-op wiring would report exactly zero.
    assert alignment["mean_pose_correction_arcmin"] > 0.0


def test_the_refined_poses_are_the_ones_the_bake_uses(misregistered, tmp_path):
    """Computing the refinement and then ignoring it must not pass.

    It did. The test above asserts the *stats* land in `mesh_metrics.json`, and
    a mutation that solved for the refined poses and then dropped them on the
    floor -- never substituting them into the dataset the bake reads -- kept
    that test green. Stats are an output of the solve, not evidence the solve
    was used: this is the ISSUES.md § 5 failure mode arriving *inside* a test
    written to pre-empt it.

    So this compares the delivered colours. Per-vertex mode deliberately, not an
    atlas: `compute_uvatlas` is non-deterministic, so two runs produce two UV
    layouts and their texture images are not comparable pixel to pixel. Vertex
    colours ride the mesh's own vertex order, which is fixed by the file.
    """
    _examples_on_path()
    import open3d as o3d
    from datasets.colmap import Parser

    import extract_mesh
    from make_synthetic_capture import surface_pattern

    def bake(align: bool):
        cfg = _config(
            misregistered,
            tmp_path / ("aligned" if align else "raw"),
            data_dir=misregistered["data_dir"],
            mesh_path=misregistered["mesh_path"],
            texture_mode="vertex",
            photometric_align=align,
            photometric_align_levels=3,
            photometric_align_iters=40,
            photometric_align_rounds=2,
        )
        extract_mesh.main(cfg)
        mesh = o3d.io.read_triangle_mesh(str(Path(cfg.result_dir) / "mesh.ply"))
        return np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)

    raw_xyz, raw_rgb = bake(False)
    _, aligned_rgb = bake(True)

    # Discarding the refined poses would make these byte-identical.
    assert not np.allclose(raw_rgb, aligned_rgb), (
        "the aligned and unaligned bakes are identical, so the refined poses "
        "never reached the bake"
    )

    # And it is used the right way round: the vertices are written in the
    # dataset's normalized frame, so map them back to world coordinates to
    # evaluate the analytic colour the capture was rendered with.
    parser = Parser(
        data_dir=misregistered["data_dir"],
        factor=1,
        normalize=True,
        test_every=10_000,
    )
    world = (
        np.linalg.inv(parser.transform) @ np.c_[raw_xyz, np.ones(len(raw_xyz))].T
    ).T[:, :3]
    truth = np.clip(surface_pattern(world), 0.0, 1.0)
    # Only vertices some camera actually coloured; an unobserved vertex keeps
    # a default and would compare against the pattern as pure noise.
    seen = raw_rgb.sum(axis=1) > 0
    raw_l1 = float(np.abs(raw_rgb[seen] - truth[seen]).mean())
    aligned_l1 = float(np.abs(aligned_rgb[seen] - truth[seen]).mean())
    assert aligned_l1 < raw_l1, (raw_l1, aligned_l1)


def test_super_resolution_runs_from_the_cli_and_beats_the_blend(capture, tmp_path):
    """`--texture_super_resolve` end to end, pinned on its *effect*.

    Asserting only that the flag is accepted would repeat the mistake this file
    exists to stop: `--texture_seam_smoothness` was accepted by the CLI for
    three commits while crashing inside it. So this checks the solve's own
    reported comparison -- the delivered atlas against the blend it started
    from -- which a dispatcher that quietly ignored the flag could not produce.
    """
    _examples_on_path()
    import json

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        texture_mode="atlas",
        texture_size=128,
        texture_super_resolve=True,
    )
    extract_mesh.main(cfg)

    assert (Path(cfg.result_dir) / "mesh.obj").exists()
    stats = json.loads((Path(cfg.result_dir) / "mesh_metrics.json").read_text())
    solve = stats["super_resolution"]
    # Not asserted: that the solve *converged*. On a capture this small it
    # often does not inside the iteration cap, and that is a legitimate
    # outcome the CLI warns about rather than an error -- the unknown is a
    # correction to the blended atlas, so an early stop degrades toward
    # blending instead of toward nonsense. What matters here is that the flag
    # reached the solve and the solve made progress.
    assert 0.0 < solve["solver"]["residual"] < 1.0, solve["solver"]
    assert solve["solver"]["iterations"] > 0
    assert solve["mean_psf_sigma_texels"] > 0.0
    # The solve changed the atlas: a dispatcher that accepted the flag and then
    # ran the plain blend would report these two as the same number.
    assert (
        solve["atlas_sharpness"]["mean_gradient"]
        != solve["blended_atlas_sharpness"]["mean_gradient"]
    )
    # Not asserted here: that it came out *sharper*. On this capture it does
    # not -- 0.059 against the blend's 0.075 -- and that is a real property of
    # the regime rather than a wiring fault. The fixture is 6 views of 64x64
    # over a scene 4 units across, so a source pixel covers more surface than a
    # texel and there is no sub-texel detail to recover; the gradient prior
    # then dominates the under-determined solve and smooths. The quality claim
    # is asserted in `tests/test_texturing.py`, on a fixture whose resolution
    # makes the premise true. Pinning it here as well would encode "this method
    # always helps", which the measurements do not support.


def test_super_resolution_and_view_selection_are_refused_together(capture, tmp_path):
    """Two answers to the same question, so the CLI must not take both."""
    _examples_on_path()
    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        texture_mode="atlas",
        texture_size=64,
        texture_super_resolve=True,
        texture_view_selection=True,
    )
    with pytest.raises(ValueError, match="same question"):
        extract_mesh.main(cfg)


def test_mesh_refinement_runs_from_the_cli_and_moves_the_geometry(
    misregistered, tmp_path
):
    """`--refine_mesh` end to end, pinned on the geometry it delivers.

    Stats alone are not evidence the flag was used -- that mistake has been
    made three times on this branch and twice in this session. So this compares
    the written mesh against the same run without the flag: a wiring that
    computed a refined mesh and returned the original would produce identical
    vertices.
    """
    _examples_on_path()
    import json

    import open3d as o3d

    import extract_mesh

    def run(refine):
        cfg = _config(
            misregistered,
            tmp_path / ("refined" if refine else "raw"),
            data_dir=misregistered["data_dir"],
            mesh_path=misregistered["mesh_path"],
            texture_mode="vertex",
            bake_texture_=False,
            refine_mesh=refine,
            refine_mesh_iterations=2,
        )
        extract_mesh.main(cfg)
        mesh = o3d.io.read_triangle_mesh(str(Path(cfg.result_dir) / "mesh.ply"))
        return cfg, np.asarray(mesh.vertices)

    _raw_cfg, raw = run(False)
    refined_cfg, refined = run(True)

    assert raw.shape == refined.shape
    assert not np.allclose(raw, refined), (
        "the refined mesh is identical to the unrefined one, so --refine_mesh "
        "never reached the geometry"
    )

    stats = json.loads((Path(refined_cfg.result_dir) / "mesh_metrics.json").read_text())
    refinement = stats["mesh_refinement"]
    assert refinement["num_vertices_moved"] > 0
    assert refinement["mean_vertex_displacement"] > 0.0
    for key in (
        "mean_photoconsistency_before",
        "mean_photoconsistency_after",
        "point_to_mesh_before",
        "point_to_mesh_after",
    ):
        assert key in refinement, key


def test_extra_maps_are_skipped_when_the_atlas_bake_falls_back(capture, tmp_path):
    """Losing the extra maps is right; losing the whole run is not.

    `--normal_map`/`--ao_map` are checked against `--texture_mode` up front, but
    the atlas bake can still *fall back* to per-vertex colours at run time --
    most often because `--cull_unobserved` left the mesh non-manifold, which
    removing faces does whenever it disconnects two patches that met at a
    single vertex. Measured on this fixture: culling alone takes a
    vertex-manifold mesh to a non-manifold one.

    There are then no UVs to bake a map into, and both bakers raised from
    inside `_unwrap_and_rasterize` -- *after* the albedo had already been
    baked, so the run died at the last step with all its work done. Found by
    running the CLI end to end, which is the only way it could have been.
    """
    _examples_on_path()
    import open3d as o3d

    import extract_mesh

    # Two tetrahedra joined at a single shared vertex: edge-manifold, but not
    # vertex-manifold, which is exactly the shape culling produces when it
    # disconnects two patches. Built explicitly rather than by culling, so the
    # test pins the *fallback* rather than depending on this fixture's culling
    # happening to leave a non-manifold mesh.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # the shared apex
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [-0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.0, -0.1],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 4],
            [4, 6, 5],
        ]
    )
    pinched = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    assert pinched.is_edge_manifold()
    assert not pinched.is_vertex_manifold(), "the fixture is not the shape it claims"
    mesh_path = tmp_path / "pinched.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), pinched)

    cfg = _config(
        capture,
        tmp_path,
        mesh_path=str(mesh_path),
        texture_mode="atlas",
        texture_size=64,
        normal_map=True,
        ao_map=True,
        ao_samples=4,
    )
    with pytest.warns(RuntimeWarning, match="Skipping --normal_map"):
        extract_mesh.main(cfg)

    out = Path(cfg.result_dir)
    # A .ply, because there is no atlas to hang off a .obj -- and no orphan
    # map files claiming otherwise.
    assert (out / "mesh.ply").exists()
    assert not (out / "mesh_normal.png").exists()
    assert not (out / "mesh_ao.png").exists()
    mesh = o3d.io.read_triangle_mesh(str(out / "mesh.ply"))
    assert len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()


# ---------------------------------------------------------------------------
# The checkpoint-free entry itself
# ---------------------------------------------------------------------------


def test_no_input_source_names_all_three_alternatives(capture, tmp_path):
    """The replacement for `assert cfg.ckpt`, which said only "--ckpt is required".

    That assert was what made `main()` untestable, so the error replacing it has
    to keep pointing at the two GPU-free ways in -- otherwise the next person
    with no checkpoint concludes, as before, that the script cannot be run.
    """
    _examples_on_path()
    import extract_mesh

    cfg = _config(capture, tmp_path, mesh_path=None)
    with pytest.raises(ValueError) as excinfo:
        extract_mesh.main(cfg)
    message = str(excinfo.value)
    for alternative in ("--ckpt", "--mesh_path", "--dense_points"):
        assert alternative in message, message


def test_two_surface_sources_are_refused(capture, tmp_path):
    """--ckpt and --mesh_path are different surfaces; delivering one is the job."""
    _examples_on_path()
    import extract_mesh

    cfg = _config(capture, tmp_path, ckpt="/nonexistent/ckpt.pt")
    with pytest.raises(ValueError, match="two different surfaces"):
        extract_mesh.main(cfg)


def test_poisson_runs_on_a_dense_cloud_with_no_checkpoint(capture, tmp_path):
    """`--method poisson` never opens a checkpoint, and now never asks for one.

    The old assert demanded `--ckpt` before the method dispatch, so this path --
    which is pure open3d and needs no GPU at all -- was unreachable without one.
    """
    _examples_on_path()
    import open3d as o3d

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        mesh_path=None,
        method="poisson",
        dense_points=capture["dense_path"],
        poisson_depth=6,
        texture_mode="vertex",
    )
    extract_mesh.main(cfg)

    ply = Path(cfg.result_dir) / "mesh.ply"
    assert ply.exists()
    mesh = o3d.io.read_triangle_mesh(str(ply))
    assert len(mesh.triangles) > 0
    # Per-vertex colours, not the grey of a mesh nothing was baked onto.
    assert mesh.has_vertex_colors()
    assert float(np.asarray(mesh.vertex_colors).std()) > 0.02


def test_poisson_reports_the_parameters_it_derived(capture, tmp_path):
    """`--method poisson` must choose its normal radius and say what it chose.

    The CLI half of the derivation: the constants are gone from `Config`'s
    defaults (they are `None` now), so a run that did not derive anything would
    reach open3d with `radius=None` and fail rather than silently using a
    number nobody chose.
    """
    _examples_on_path()
    import json

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        mesh_path=None,
        method="poisson",
        dense_points=capture["dense_path"],
        poisson_depth=6,
        texture_mode="vertex",
        bake_texture_=False,
    )
    extract_mesh.main(cfg)

    stats = json.loads((Path(cfg.result_dir) / "mesh_metrics.json").read_text())
    derived = stats["reconstruction_parameters"]
    assert derived["derived"] == ["normal_radius"]
    assert derived["normal_radius"] == pytest.approx(
        3.0 * derived["point_spacing"], rel=1e-6
    )
    assert derived["point_spacing"] > 0.0

    # An explicit value still overrides, and is reported as not derived.
    cfg = _config(
        capture,
        tmp_path / "explicit",
        mesh_path=None,
        method="poisson",
        dense_points=capture["dense_path"],
        poisson_depth=6,
        poisson_normal_radius=0.25,
        texture_mode="vertex",
        bake_texture_=False,
    )
    extract_mesh.main(cfg)
    stats = json.loads((Path(cfg.result_dir) / "mesh_metrics.json").read_text())
    assert stats["reconstruction_parameters"]["derived"] == []
    assert stats["reconstruction_parameters"]["normal_radius"] == 0.25


def test_geometry_off_disk_lands_in_the_dataset_frame(capture, tmp_path):
    """A mesh read from disk must be put in the frame the cameras live in.

    `Parser(normalize=True)` -- which this script always uses -- rescales and
    reorients the world, and hands out poses in *that* frame. A mesh or a dense
    cloud read straight off disk is still in the COLMAP model's original world
    coordinates. `Parser` applies exactly this transform to its own
    `dense_points_path`; these two paths read the file themselves, so they have
    to as well, and the `--method poisson` path never did.

    Measured on this capture the two frames differ by ~3.4x in scale plus a
    rotation, which puts every camera *inside* the mesh -- so the failure is not
    subtle, it is a bake against geometry nothing can see. Asserting on the
    written mesh's extent catches it without needing to eyeball a texture.
    """
    _examples_on_path()
    import open3d as o3d
    from datasets.colmap import Parser

    import extract_mesh

    cfg = _config(
        capture,
        tmp_path,
        texture_mode="atlas",
        texture_size=64,
        bake_texture_=False,
    )
    extract_mesh.main(cfg)

    parser = Parser(data_dir=cfg.data_dir, factor=1, normalize=True, test_every=10_000)
    assert not np.allclose(parser.transform, np.eye(4)), (
        "the normalization is the identity on this capture, so this test cannot "
        "distinguish a transformed mesh from an untransformed one"
    )

    written = o3d.io.read_triangle_mesh(str(Path(cfg.result_dir) / "mesh.ply"))
    extent = np.asarray(written.vertices).max(0) - np.asarray(written.vertices).min(0)
    camera_extent = parser.camtoworlds[:, :3, 3].max(0) - parser.camtoworlds[
        :, :3, 3
    ].min(0)
    # The subject sits inside the ring of cameras that photographed it. Skip the
    # transform and the mesh keeps its raw ~4-unit span against a camera ring
    # barely 2 units across.
    assert np.max(extent) < np.max(camera_extent), (
        f"mesh extent {extent} is not inside the camera extent {camera_extent}; "
        "the mesh was probably not mapped into the dataset's normalized frame"
    )


# ---------------------------------------------------------------------------
# The one-command path, for real
# ---------------------------------------------------------------------------


def test_run_pipeline_delivers_a_mesh_without_a_checkpoint(capture, tmp_path):
    """`run_pipeline.py`'s extract_mesh stage, non-dry-run, end to end.

    Every previous pipeline test was `--dry_run`: they compared the constructed
    argv against `extract_mesh.Config`'s fields, which is why a flag that was
    *accepted* by the CLI and then crashed inside it passed them all. This one
    actually runs the subprocess and checks the artifact.
    """
    _examples_on_path()
    import json

    result_dir = tmp_path / "pipeline"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_DIR / "run_pipeline.py"),
            "--data_dir",
            capture["data_dir"],
            "--result_dir",
            str(result_dir),
            "--data_factor",
            "1",
            "--stages",
            "extract_mesh",
            "--mesh_path",
            capture["mesh_path"],
            "--texture_mode",
            "atlas",
            "--texture_size",
            "128",
            "--texture_view_selection",
            "--cull_unobserved",
            "--device",
            "cpu",
            "--extract_mesh_extra_args=--test_every",
            "10000",
        ],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLES_DIR),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert (result_dir / "mesh.obj").exists(), proc.stdout
    report = json.loads((result_dir / "pipeline_report.json").read_text())
    stage = next(s for s in report["stages"] if s["name"] == "extract_mesh")
    assert stage["status"] == "ok", stage
