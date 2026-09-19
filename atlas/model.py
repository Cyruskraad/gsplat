# SPDX-FileCopyrightText: Copyright 2026 Cyrus Kraad and contributors. All rights reserved.
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

"""The relightable Gaussian model, and Path A rendering.

A :class:`RelightSplats` is an ordinary Gaussian scene -- means, rotations,
scales, opacities, exactly what ``gsplat.rasterization`` expects -- with one
parameter added: a per-primitive **transport** ``[N, 3, B]``. Colour is not
stored. It is produced, per frame, by contracting the transport against the
illumination's projection onto the atom basis.

Path A: contract, then splat
----------------------------

::

    colors = contract(transport, ell)      # [N, 3], one matvec over primitives
    rasterization(..., colors=colors, sh_degree=None)

The contraction does not involve the camera, so **one contraction serves any
number of views**. That is the whole reason this path exists, and it is what
makes stereo, a light-field display or a multi-camera rig cost one contraction
plus N cheap three-channel splats.

Path B -- splat the ``3B`` transport channels once and contract per pixel --
renders the identical image and suits the opposite case, a fixed camera under
changing light. It is not implemented yet; the exactness of the two paths is
already proven in ``atlas.functional.transport`` and pinned in the tests.

``gsplat`` is imported lazily
-----------------------------

Only :meth:`RelightSplats.render` needs it, and it needs CUDA. Everything else
here -- construction, PLY initialisation, saving, loading, parameter counting --
works in a bare interpreter, so the model is testable on a laptop and the import
error, when it comes, names the extra to install rather than surfacing as a
mysterious ``ModuleNotFoundError`` at the top of the file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .functional.atoms import make_sg_atoms, project_environment
from .functional.transport import contract_chunked
from .ply import SH_C0, read_ply

__all__ = [
    "RelightSplats",
    "DEFAULT_NUM_ATOMS",
    "PRIMITIVE_PARAMETERS",
    "ATOM_PARAMETERS",
    "MIN_SHARPNESS",
]

#: The per-primitive tensors, in the order gsplat's densification expects to
#: find them. Every one is ``[N, ...]``, so concatenating along dimension zero
#: is always the right thing.
PRIMITIVE_PARAMETERS = ("means", "quats", "scales", "opacities", "transport")

#: The basis. ``[B, ...]``, so these must never go through densification.
ATOM_PARAMETERS = ("atom_axes", "atom_sharpness")

#: Floor on a learned sharpness. Below this an atom is so broad that it is
#: nearly constant over the sphere and contributes only a bias, and the fit
#: becomes ill-conditioned rather than wrong -- which is harder to notice.
MIN_SHARPNESS = 1e-2

# Near-field training needs a per-primitive [N, B] atom evaluation. At 600k
# primitives that is 77 MB at B=32 and 307 MB at B=128, before autograd doubles
# it. The design's "train over-complete at 128, then compress" is right, but it
# is a second run: measure the memory on real hardware first.
DEFAULT_NUM_ATOMS = 32


class RelightSplats:
    """Geometry plus per-primitive light transport.

    Attributes:
        means: ``[N, 3]`` primitive centres, world space.
        quats: ``[N, 4]`` rotations, ``(w, x, y, z)`` as gsplat expects.
        scales: ``[N, 3]`` **log** scales; the rasteriser exponentiates.
        opacities: ``[N]`` **logit** opacities; the rasteriser applies sigmoid.
        transport: ``[N, 3, B]`` the transport, and the only new parameter.
        atom_axes: ``[B, 3]`` and ``atom_sharpness`` ``[B]`` -- the basis the
            transport is expressed in. Stored with the model because a
            checkpoint is meaningless without the basis its coefficients refer
            to.
    """

    def __init__(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        transport: Tensor,
        atom_axes: Tensor,
        atom_sharpness: Tensor,
    ):
        num = means.shape[0]
        if means.shape != (num, 3):
            raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
        if quats.shape != (num, 4):
            raise ValueError(f"quats must be [{num}, 4], got {tuple(quats.shape)}")
        if scales.shape != (num, 3):
            raise ValueError(f"scales must be [{num}, 3], got {tuple(scales.shape)}")
        if opacities.shape != (num,):
            raise ValueError(f"opacities must be [{num}], got {tuple(opacities.shape)}")
        if transport.ndim != 3 or transport.shape[:2] != (num, 3):
            raise ValueError(
                f"transport must be [{num}, 3, B], got {tuple(transport.shape)}"
            )
        num_atoms = transport.shape[-1]
        if atom_axes.shape != (num_atoms, 3):
            raise ValueError(
                f"atom_axes must be [{num_atoms}, 3] to match the transport, got "
                f"{tuple(atom_axes.shape)}"
            )
        if atom_sharpness.shape != (num_atoms,):
            raise ValueError(
                f"atom_sharpness must be [{num_atoms}], got "
                f"{tuple(atom_sharpness.shape)}"
            )
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.transport = transport
        self.atom_axes = atom_axes
        self.atom_sharpness = atom_sharpness

    # --- shape and bookkeeping ---------------------------------------------

    @property
    def num_primitives(self) -> int:
        return self.means.shape[0]

    @property
    def num_atoms(self) -> int:
        return self.transport.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def to(self, *args, **kwargs) -> "RelightSplats":
        moved = {
            name: getattr(self, name).to(*args, **kwargs)
            for name in (
                "means",
                "quats",
                "scales",
                "opacities",
                "transport",
                "atom_axes",
                "atom_sharpness",
            )
        }
        return RelightSplats(**moved)

    def requires_grad_(
        self, flag: bool = True, *, atoms: bool = False
    ) -> "RelightSplats":
        """Turn gradients on. ``atoms`` is the L in ATLAS and is opt-in.

        The basis is held out by default so that a first run has a fixed-basis
        baseline to attribute any later gain to. Learning it is a change to the
        method, not a tuning knob, and it should be measured as one.
        """
        for name in PRIMITIVE_PARAMETERS:
            getattr(self, name).requires_grad_(flag)
        for name in ATOM_PARAMETERS:
            getattr(self, name).requires_grad_(flag and atoms)
        return self

    @torch.no_grad()
    def normalise_atoms_(
        self, *, min_sharpness: float = MIN_SHARPNESS
    ) -> "RelightSplats":
        """Project the basis back onto its constraints, in place.

        An atom is a spherical Gaussian ``exp(lambda (w . xi - 1))``: ``xi``
        must be a unit vector or the lobe is both rotated and rescaled by the
        same number, and ``lambda`` must stay positive or the lobe inverts into
        a trough that grows without bound away from its axis.

        This is projected gradient descent -- the step is unconstrained and the
        projection follows it -- rather than an ``exp`` or softplus
        reparameterisation. The reason is legibility: ``atom_sharpness`` in a
        checkpoint is then the ``lambda`` the mathematics uses, and
        ``atom_axes`` really are unit vectors, so both invariants can be checked
        on the file rather than inferred through a transform.

        Call it after every ``optimizer.step()`` that touched the atoms.
        """
        self.atom_axes.div_(self.atom_axes.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        self.atom_sharpness.clamp_(min=min_sharpness)
        return self

    # --- the densification view ---------------------------------------------

    def as_parameter_dict(self) -> "torch.nn.ParameterDict":
        """The per-primitive tensors, as gsplat's strategies want them.

        ``gsplat.strategy.ops`` special-cases ``means``, ``scales`` and
        ``opacities`` by name and treats every other entry generically --
        ``p[sel].repeat([2] + [1] * (p.dim() - 1))`` -- so ``transport``
        ``[N, 3, B]`` is carried through a split or a clone with no code at all.
        That is exactly the reuse the design counts on.

        The atoms are **not** in here. They are ``[B, ...]``, not ``[N, ...]``,
        and densification concatenates along dimension zero: including them
        would silently grow the basis every time a primitive split.
        """
        return torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    getattr(self, name), requires_grad=getattr(self, name).requires_grad
                )
                for name in PRIMITIVE_PARAMETERS
            }
        )

    def atom_parameters(self) -> Dict[str, Tensor]:
        """The basis, which lives in its own optimiser group."""
        return {name: getattr(self, name) for name in ATOM_PARAMETERS}

    @classmethod
    def from_parameter_dict(
        cls,
        params: "torch.nn.ParameterDict",
        atom_axes: Tensor,
        atom_sharpness: Tensor,
    ) -> "RelightSplats":
        """Rebuild a model around a dict that densification may have replaced.

        No copy: the parameters are taken as they are, so the model and the
        optimiser keep referring to the same storage.
        """
        missing = [name for name in PRIMITIVE_PARAMETERS if name not in params]
        if missing:
            raise ValueError(f"the parameter dict is missing {missing}")
        return cls(
            **{name: params[name] for name in PRIMITIVE_PARAMETERS},
            atom_axes=atom_axes,
            atom_sharpness=atom_sharpness,
        )

    def parameter_bytes(self) -> Dict[str, int]:
        """Bytes per parameter block, so a memory surprise arrives early."""
        return {
            name: getattr(self, name).numel() * getattr(self, name).element_size()
            for name in ("means", "quats", "scales", "opacities", "transport")
        }

    # --- construction -------------------------------------------------------

    @classmethod
    def from_ply(
        cls,
        path: Path | str,
        *,
        num_atoms: int = DEFAULT_NUM_ATOMS,
        sharpness: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "RelightSplats":
        """Initialise from a fixed-light gsplat reconstruction of the same object.

        Geometry is taken as-is. The transport is initialised so that **under a
        uniform white environment the model reproduces the colour the
        fixed-light reconstruction had**, which means the very first render is
        recognisably the object rather than noise -- the difference between a
        smoke test that tells you something and one that only tells you the code
        ran.

        The arithmetic: under a uniform environment of unit radiance every atom
        integrates to the same constant ``I``, so the rendered colour is
        ``I * sum_k transport[c, k]``. Setting every coefficient to
        ``albedo[c] / (I * B)`` makes that sum the albedo.

        Args:
            path: A PLY written by ``gsplat.export_splats``.
            num_atoms: Size of the light basis, ``B``.
            sharpness: Atom sharpness; defaults to the touching-lobes value.
            device, dtype: Where and in what precision to build the tensors.

        Returns:
            A :class:`RelightSplats`.
        """
        data = read_ply(path)
        required = ["x", "y", "z", "opacity"]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                f"{Path(path).name}: PLY is missing {missing}; expected the "
                f"layout gsplat.export_splats writes. Found: {data.names()[:12]}..."
            )

        def column(name: str) -> Tensor:
            return torch.tensor(data[name], dtype=dtype, device=device)

        means = torch.stack([column("x"), column("y"), column("z")], dim=-1)

        scale_names = data.prefixed("scale_")
        if len(scale_names) != 3:
            raise ValueError(f"expected scale_0..2 in the PLY, found {scale_names}")
        scales = torch.stack([column(n) for n in scale_names], dim=-1)

        rot_names = data.prefixed("rot_")
        if len(rot_names) != 4:
            raise ValueError(f"expected rot_0..3 in the PLY, found {rot_names}")
        quats = torch.stack([column(n) for n in rot_names], dim=-1)

        opacities = column("opacity")

        dc_names = data.prefixed("f_dc_")
        if len(dc_names) == 3:
            albedo = torch.stack([column(n) for n in dc_names], dim=-1) * SH_C0 + 0.5
        else:
            # No colour in the PLY: start grey rather than refuse. The transport
            # is going to be fitted anyway; this only affects iteration zero.
            albedo = torch.full((means.shape[0], 3), 0.5, dtype=dtype, device=device)
        albedo = albedo.clamp(min=1e-4)

        axes, sharpnesses = make_sg_atoms(
            num_atoms, sharpness=sharpness, device=device, dtype=dtype
        )
        uniform = _uniform_environment_coefficient(axes, sharpnesses)
        transport = (
            (albedo / (uniform * num_atoms)).unsqueeze(-1).expand(-1, -1, num_atoms)
        )

        return cls(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            transport=transport.contiguous(),
            atom_axes=axes,
            atom_sharpness=sharpnesses,
        )

    # --- rendering ----------------------------------------------------------

    def render(
        self,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        ell,
        *,
        chunk_size: int = 0,
        validate: bool = False,
        **rasterization_kwargs: Any,
    ):
        """Path A: contract the transport against the light, then splat.

        Args:
            viewmats: ``[C, 4, 4]`` world-to-camera.
            Ks: ``[C, 3, 3]`` pinhole intrinsics.
            width, height: Output size.
            ell: ``[3, B]`` shared light coefficients, ``[N, 3, B]`` when every
                primitive sees its own light -- which is what a near-field point
                light produces -- or a callable ``(start, stop)`` producing one
                chunk at a time, which is how a near-field light avoids being
                stored at all.
            chunk_size: Primitives per contraction chunk. ``0`` picks one from a
                memory budget. Only the per-primitive and callable light forms
                have a temporary to save; see ``atlas.functional.transport``.
            validate: Check each chunk for non-finite values, so a diverged
                model names the primitive instead of rendering black.
            **rasterization_kwargs: Passed through to ``gsplat.rasterization``.

        Returns:
            Whatever ``gsplat.rasterization`` returns: ``(colors, alphas, meta)``.
        """
        try:
            from gsplat import rasterization
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "rendering needs gsplat and a CUDA device. "
                'Install it with: pip install -e ".[gpu]"'
            ) from exc

        colors = contract_chunked(
            self.transport, ell, chunk_size=chunk_size, validate=validate
        )  # [N, 3]
        return rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
            **rasterization_kwargs,
        )

    def render_image(
        self,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        ell,
        *,
        backend: str = "auto",
        chunk_size: int = 0,
        validate: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """One view, through whichever renderer this machine can run.

        This is the single call site that makes "works on the GPU when there is
        one" true rather than aspirational. ``"auto"`` uses the CUDA rasteriser
        when CUDA and gsplat are both present and the reference renderer
        otherwise, and it never pretends the two are interchangeable in speed:
        the reference path is roughly a thousand times slower and is for smoke
        tests and oracles.

        Args:
            viewmat: ``[4, 4]`` world-to-camera.
            K: ``[3, 3]`` intrinsics.
            width, height: Frame size.
            ell: Light coefficients, in any form
                :func:`~atlas.functional.transport.contract_chunked` accepts.
            backend: ``"auto"``, ``"gsplat"`` or ``"reference"``.
            chunk_size: Primitives per contraction chunk; ``0`` auto-sizes.
            validate: Check each chunk for non-finite values.

        Returns:
            ``(image [H, W, 3], alpha [H, W])``.
        """
        backend = self.resolve_backend(backend)
        colors = contract_chunked(
            self.transport, ell, chunk_size=chunk_size, validate=validate
        )

        if backend == "reference":
            from .reference import render_reference

            return render_reference(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                colors,
                viewmat,
                K,
                width,
                height,
                **kwargs,
            )

        try:
            from gsplat import rasterization
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "the gsplat backend was asked for but gsplat is not installed. "
                'Install it with: pip install -e ".[gpu]", or pass '
                'backend="reference" to use the CPU renderer.'
            ) from exc

        rendered, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities),
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            **kwargs,
        )
        return rendered[0], alphas[0, ..., 0]

    @staticmethod
    def resolve_backend(backend: str = "auto") -> str:
        """Which renderer ``backend`` names on this machine, and why not.

        Raises rather than falling back when a backend is named explicitly, for
        the same reason ``atlas.device.select_device`` does: a CPU run that
        silently replaces a GPU run still produces numbers.
        """
        backend = (backend or "auto").strip().lower()
        if backend not in ("auto", "gsplat", "reference"):
            raise ValueError(
                f"backend must be 'auto', 'gsplat' or 'reference', got {backend!r}"
            )
        if backend == "reference":
            return "reference"

        try:
            import gsplat  # noqa: F401

            has_gsplat = True
        except ImportError:
            has_gsplat = False
        usable = has_gsplat and torch.cuda.is_available()

        if backend == "gsplat":
            if not usable:
                missing = []
                if not has_gsplat:
                    missing.append("gsplat is not installed")
                if not torch.cuda.is_available():
                    missing.append("torch reports no CUDA device")
                raise RuntimeError(
                    f"the gsplat backend was requested but {' and '.join(missing)}. "
                    f"Refusing to fall back to the reference renderer, which is "
                    f"about a thousand times slower."
                )
            return "gsplat"
        return "gsplat" if usable else "reference"

    def project_environment(self, envmap: Tensor) -> Tensor:
        """Project an equirectangular environment onto this model's atom basis.

        The basis travels with the model precisely so that this cannot be done
        against the wrong one.
        """
        return project_environment(envmap, self.atom_axes, self.atom_sharpness)

    # --- persistence --------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save to ``.pt``. gsplat's PLY exporter is SH-only and cannot hold this."""
        torch.save(
            {
                "format": "atlas-relight-splats",
                "version": 1,
                "means": self.means.detach().cpu(),
                "quats": self.quats.detach().cpu(),
                "scales": self.scales.detach().cpu(),
                "opacities": self.opacities.detach().cpu(),
                "transport": self.transport.detach().cpu(),
                "atom_axes": self.atom_axes.detach().cpu(),
                "atom_sharpness": self.atom_sharpness.detach().cpu(),
            },
            Path(path),
        )

    @classmethod
    def load(
        cls, path: Path | str, *, device: Optional[torch.device] = None
    ) -> "RelightSplats":
        payload = torch.load(
            Path(path), map_location=device or "cpu", weights_only=True
        )
        if payload.get("format") != "atlas-relight-splats":
            raise ValueError(
                f"{Path(path).name}: not an ATLAS checkpoint "
                f"(format={payload.get('format')!r})"
            )
        return cls(
            means=payload["means"],
            quats=payload["quats"],
            scales=payload["scales"],
            opacities=payload["opacities"],
            transport=payload["transport"],
            atom_axes=payload["atom_axes"],
            atom_sharpness=payload["atom_sharpness"],
        )


def _uniform_environment_coefficient(axes: Tensor, sharpnesses: Tensor) -> Tensor:
    """What one atom integrates to under a uniform environment of unit radiance.

    Closed form for a peak-normalised spherical Gaussian:
    ``2*pi*(1 - exp(-2*lambda)) / lambda``. Used only by the initialiser, and
    checked against the numerical projection in the tests so the two cannot
    drift apart.
    """
    lam = sharpnesses.to(torch.float64)
    value = 2.0 * math.pi * (1.0 - torch.exp(-2.0 * lam)) / lam
    return value.to(sharpnesses.dtype).mean()
