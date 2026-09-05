"""Guards on the seam between `examples/extract_mesh.py` and the library.

`tests/test_photogrammetry_pipeline.py` already guards the *outer* seam --
every `--flag` `run_pipeline.py` emits must be a field `extract_mesh.Config`
accepts. Nothing guarded the *inner* one: the keyword arguments `main()`
passes to the photogrammetry functions it imports.

That gap shipped a hard crash. `fa70683` added `--texture_seam_smoothness` and
passed `seam_smoothness=` to `bake_mesh_texture()`, which had no such parameter
and no `**kwargs`. Since `bake_texture_` defaults to True and the keyword was
passed unconditionally, *every* texture-baking run of `extract_mesh.py` -- and
so `run_pipeline.py`'s whole delivery stage -- raised

    TypeError: bake_mesh_texture() got an unexpected keyword argument
    'seam_smoothness'

`bake_texture_atlas_view_selected` had been tested directly and worked; only
the call site was wrong. That is the fifth time on this branch a mechanism was
proved while its caller went unpinned (docs/handoff/ISSUES.md section 5), so
the first test here is deliberately general -- it checks every such call, not
just the one that broke.
"""

import ast
import inspect
import os
import sys

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="open3d not installed")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))
sys.path.insert(0, os.path.dirname(__file__))

EXTRACT_MESH_PY = os.path.join(REPO_ROOT, "examples", "extract_mesh.py")


def _photogrammetry_imports(tree):
    """Map local name -> the `gsplat.photogrammetry` object it refers to."""
    import importlib

    resolved = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module or "").startswith("gsplat.photogrammetry"):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            target = getattr(module, alias.name, None)
            if callable(target):
                resolved[alias.asname or alias.name] = target
    return resolved


def test_extract_mesh_passes_only_keywords_the_library_accepts():
    """Every kwarg `extract_mesh.py` passes must exist on the real signature.

    A static check rather than a run, so it covers call sites on branches a
    single invocation would not reach (poisson vs tsdf, each texture mode).
    """
    tree = ast.parse(open(EXTRACT_MESH_PY).read())
    resolved = _photogrammetry_imports(tree)
    assert resolved, "found no gsplat.photogrammetry imports -- the parse is wrong"

    problems = []
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = resolved.get(node.func.id)
        if target is None:
            continue
        # `**kwargs` at a call site hides the names; skip rather than guess.
        if any(kw.arg is None for kw in node.keywords):
            continue
        signature = inspect.signature(target)
        accepts_any = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if accepts_any:
            continue
        checked += 1
        for kw in node.keywords:
            if kw.arg not in signature.parameters:
                problems.append(
                    f"line {node.lineno}: {node.func.id}(... {kw.arg}=...) -- "
                    f"{node.func.id}{signature} has no such parameter"
                )

    assert checked, "matched no calls -- the parse is wrong, not the code"
    assert (
        not problems
    ), "extract_mesh.py passes keywords the library rejects:\n" + "\n".join(problems)


def _mesh_with_fixed_uvs(dataset, resolution, texture_size):
    """A sphere whose UV layout is already decided.

    open3d's `compute_uvatlas` is non-deterministic (docs/handoff/ISSUES.md),
    so two independently unwrapped meshes produce atlases that cannot be
    compared texel-by-texel -- the difference would be the layout, not the
    thing under test. `_unwrap_and_rasterize` reuses `triangle_uvs` when they
    are already present, so baking once and reusing the returned mesh pins the
    layout for every later bake.
    """
    import copy

    from gsplat.photogrammetry.texturing import bake_mesh_texture
    from test_mesh_extraction import _unit_sphere_mesh

    seeded, _ = bake_mesh_texture(
        _unit_sphere_mesh(resolution=resolution),
        dataset,
        mode="atlas",
        texture_size=texture_size,
    )
    assert seeded.has_triangle_uvs(), "the seeding bake produced no UV layout"
    return lambda: copy.deepcopy(seeded)


def test_bake_mesh_texture_forwards_seam_smoothness_to_the_levelling():
    """Accepting the keyword is not enough -- it must reach `level_seams`.

    Adding the parameter without forwarding it satisfies the static guard
    above while silently making `--texture_seam_smoothness` a no-op, so this
    pins the behaviour: with per-view exposure offsets, levelling must change
    the atlas.
    """
    from gsplat.photogrammetry.texturing import bake_mesh_texture
    from test_mesh_extraction import _SphereDataset

    dataset = _SphereDataset(num_views=8, width=96, height=96, exposure=0.15)
    fresh = _mesh_with_fixed_uvs(dataset, resolution=6, texture_size=128)

    def bake(seam_smoothness):
        _, texture = bake_mesh_texture(
            fresh(),
            dataset,
            mode="atlas",
            texture_size=128,
            view_selection=True,
            seam_smoothness=seam_smoothness,
            stats_out={},
        )
        return texture.astype(np.float64)

    unlevelled = bake(None)
    levelled = bake(0.1)

    # Premise: the atlas is real and both bakes share one layout, so any
    # difference below is the levelling rather than a fresh unwrap.
    assert unlevelled.shape == levelled.shape
    assert unlevelled.max() > 0
    # Premise: the bake is itself deterministic on a fixed layout, or the
    # comparison measures noise.
    assert np.array_equal(bake(None), unlevelled)

    changed = float(np.abs(levelled - unlevelled).mean())
    assert changed > 0.5, (
        "seam_smoothness did not change the atlas -- bake_mesh_texture is "
        f"accepting the argument without forwarding it (mean delta {changed:.4f})"
    )


def test_bake_mesh_texture_ignores_seam_smoothness_without_view_selection():
    """Blending produces no label boundaries, so there is nothing to level.

    `extract_mesh.py` passes the flag unconditionally, so this must be
    accepted-and-ignored rather than an error -- the same contract
    `mrf_smoothness` already has.
    """
    from gsplat.photogrammetry.texturing import bake_mesh_texture
    from test_mesh_extraction import _SphereDataset

    dataset = _SphereDataset(num_views=6, width=64, height=64)
    fresh = _mesh_with_fixed_uvs(dataset, resolution=4, texture_size=64)

    def bake(**kwargs):
        _, texture = bake_mesh_texture(
            fresh(), dataset, mode="atlas", texture_size=64, **kwargs
        )
        return texture

    baseline = bake(seam_smoothness=0.1)
    assert baseline.max() > 0, "the blended atlas came back empty"
    assert np.array_equal(baseline, bake(seam_smoothness=None))
    assert np.array_equal(baseline, bake())


def _capture(tmp_path, **overrides):
    """Write a small synthetic capture and return its build() result."""
    from make_synthetic_capture import Config, build

    settings = dict(
        out_dir=str(tmp_path / "capture"),
        num_views=10,
        width=96,
        height=96,
        focal=130.0,
        num_points=250,
        num_dense_points=6000,
        mesh_resolution=12,
    )
    settings.update(overrides)
    return build(Config(**settings))


def _run(tmp_path, capture, **overrides):
    import extract_mesh

    settings = dict(
        method="mesh",
        mesh_path=capture["mesh_path"],
        data_dir=capture["data_dir"],
        data_factor=1,
        test_every=10_000,
        result_dir=str(tmp_path / "out"),
        device="cpu",
        texture_mode="atlas",
        texture_size=128,
    )
    settings.update(overrides)
    cfg = extract_mesh.Config(**settings)
    extract_mesh.main(cfg)
    return settings["result_dir"]


def test_the_whole_delivery_path_runs_from_the_cli(tmp_path):
    """Run `extract_mesh.main()` itself, with every delivery option on.

    This is the test the crash in `fa70683` needed and did not have. Until a
    checkpoint-free entry existed, `main()` could not be reached without a GPU,
    so roughly ten CLI guards were 'reviewed, never executed' -- and one of them
    raised `TypeError` on every run.

    It asserts the artifacts rather than any quality number: the point is that
    the wiring executes end to end and the files a DCC tool needs land on disk.
    """
    import json

    cv2 = pytest.importorskip("cv2", reason="opencv not installed")

    capture = _capture(tmp_path)
    result_dir = _run(
        tmp_path,
        capture,
        cull_unobserved=True,
        texture_view_selection=True,
        texture_seam_smoothness=0.1,
        texture_outlier_sigma=2.0,
        target_fit_ratio=0.25,
        normal_map=True,
        normal_map_bits=16,
        ao_map=True,
        ao_samples=8,
    )

    # A UV atlas means .obj (a .ply cannot carry UVs), plus its material and maps.
    for name in (
        "mesh.obj",
        "mesh.mtl",
        "mesh_0.png",
        "mesh_normal.png",
        "mesh_ao.png",
    ):
        path = os.path.join(result_dir, name)
        assert os.path.exists(path), f"{name} was not written"
        assert os.path.getsize(path) > 0, f"{name} is empty"

    # The 16-bit normal map must be 16-bit *on disk*. It has to be read with
    # OpenCV to check that: imageio's Pillow backend cannot write a 16-bit RGB
    # PNG and silently down-converts one to uint8 on *read* too, so reading it
    # back through imageio reports uint8 for a file that is genuinely uint16.
    normal = cv2.imread(
        os.path.join(result_dir, "mesh_normal.png"), cv2.IMREAD_UNCHANGED
    )
    assert normal is not None, "OpenCV could not read the normal map back"
    assert (
        normal.dtype == np.uint16
    ), f"normal map is {normal.dtype} on disk, not uint16"

    mesh = o3d.io.read_triangle_mesh(os.path.join(result_dir, "mesh.obj"))
    assert len(mesh.triangles) > 0
    assert mesh.has_triangle_uvs()

    # Every stage that was asked for must have recorded its own stats. This is
    # what makes the test a check that each stage *ran*, rather than only that
    # the script exited zero.
    metrics = json.load(open(os.path.join(result_dir, "mesh_metrics.json")))
    for key in (
        "culling",
        "decimation",
        "view_selection",
        "normal_map",
        "ambient_occlusion",
        "point_to_mesh",
    ):
        assert key in metrics, (
            f"mesh_metrics.json is missing {key!r}, so that stage did not run: "
            f"{sorted(metrics)}"
        )
    assert metrics["is_watertight"] is True
    assert metrics["num_triangles"] > 0
    assert metrics["point_to_mesh"]["mean"] is not None


def test_a_colmap_frame_mesh_is_transformed_into_the_camera_frame(tmp_path):
    """`Parser(normalize=True)` moves the cameras; the mesh must follow.

    A mesh or dense cloud read straight off disk is in the sparse model's raw
    frame, but `extract_mesh.py` builds its `Parser` with `normalize=True`, so
    `dataset`'s cameras are not. Nothing raises when they disagree -- the mesh
    is simply textured from cameras that do not line up with it, which is the
    worst kind of bug this pipeline can have, since it ships a plausible-looking
    asset. `Parser` applies the same transform to its own `dense_points_path`
    for exactly this reason.

    Driven through `main()` rather than by re-deriving the transform here: a
    test that reimplements the mechanism proves the mechanism and leaves the
    call site unpinned, which is how the five bugs in docs/handoff/ISSUES.md
    section 5 got in. `--mesh_frame normalized` is the switch that skips the
    transform, so it reproduces the old behaviour exactly.
    """
    import json

    capture = _capture(tmp_path)

    def culled_fraction(mesh_frame):
        result_dir = _run(
            tmp_path / mesh_frame,
            capture,
            mesh_frame=mesh_frame,
            cull_unobserved=True,
            result_dir=str(tmp_path / mesh_frame / "out"),
        )
        stats = json.load(open(os.path.join(result_dir, "mesh_metrics.json")))
        culling = stats["culling"]
        return culling["num_culled"] / culling["num_faces_before"]

    # Premise: this capture surrounds the subject, so with the frames agreeing
    # essentially nothing is unobserved. Without that the comparison is empty.
    correct = culled_fraction("colmap")
    assert correct < 0.05, (
        f"{correct:.1%} of faces unobserved even in the right frame -- the "
        "capture does not surround the subject, so this test proves nothing"
    )

    untransformed = culled_fraction("normalized")
    assert untransformed > 0.5, (
        "skipping the normalization transform left the mesh visible anyway "
        f"({untransformed:.1%} culled), so this scene cannot detect the bug"
    )


def test_an_unreadable_mesh_path_says_so(tmp_path):
    """A mesh that loads no triangles must name the likely cause.

    open3d's readers do not raise on a path it cannot parse -- they return an
    empty mesh -- so without this the run continues and fails much later,
    somewhere inside texturing, with nothing pointing at the input.
    """
    import extract_mesh

    capture = _capture(tmp_path, num_views=4, num_points=60, write_dense=False)
    empty = tmp_path / "empty.ply"
    empty.write_text("not a mesh at all\n")

    with pytest.raises(ValueError, match="no triangles"):
        extract_mesh.main(
            extract_mesh.Config(
                method="mesh",
                mesh_path=str(empty),
                data_dir=capture["data_dir"],
                data_factor=1,
                test_every=10_000,
                result_dir=str(tmp_path / "out"),
                device="cpu",
            )
        )


def test_a_dense_cloud_is_transformed_into_the_camera_frame(tmp_path):
    """The same frame guard, on the Poisson path -- where the bug was found.

    A dense MVS cloud comes out of `colmap` in the sparse model's raw frame,
    and `extract_mesh.py` read it straight off disk while normalizing the
    cameras. Kept as its own test because the mesh path's guard does not cover
    it: dropping the transform here alone left the whole suite green.
    """
    import json

    capture = _capture(tmp_path)
    result_dir = _run(
        tmp_path,
        capture,
        method="poisson",
        mesh_path=None,
        dense_points=capture["dense_path"],
        poisson_depth=6,
        cull_unobserved=True,
        bake_texture_=True,
    )
    stats = json.load(open(os.path.join(result_dir, "mesh_metrics.json")))
    culling = stats["culling"]
    culled = culling["num_culled"] / culling["num_faces_before"]

    # The capture surrounds the subject, so a correctly-framed Poisson surface
    # is almost entirely observed. Untransformed it is mostly invisible.
    assert culled < 0.2, (
        f"{culled:.1%} of the Poisson mesh's faces were unobserved -- the "
        "cloud and the cameras are probably in different frames"
    )


def test_each_method_requires_its_own_input(tmp_path):
    """The input check must be per-method, not a blanket `assert cfg.ckpt`.

    The blanket assert made `--method poisson` demand a checkpoint it never
    opens, and made `main()` unreachable without a GPU at all -- which is how
    the `TypeError` above went unnoticed for five commits.
    """
    import extract_mesh

    capture = _capture(tmp_path, num_views=4, num_points=60, write_dense=False)
    common = dict(
        data_dir=capture["data_dir"],
        data_factor=1,
        test_every=10_000,
        result_dir=str(tmp_path / "out"),
        device="cpu",
    )

    for method, flag in (
        ("tsdf", "--ckpt"),
        ("poisson", "--dense_points"),
        ("mesh", "--mesh_path"),
    ):
        with pytest.raises(ValueError, match=flag.lstrip("-")):
            extract_mesh.main(extract_mesh.Config(method=method, **common))

    # And a checkpoint is *not* required once another source is given.
    extract_mesh.main(
        extract_mesh.Config(
            method="mesh",
            mesh_path=capture["mesh_path"],
            bake_texture_=False,
            **common,
        )
    )
    assert os.path.exists(os.path.join(common["result_dir"], "mesh.ply"))
