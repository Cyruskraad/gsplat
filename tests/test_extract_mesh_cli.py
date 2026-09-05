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
