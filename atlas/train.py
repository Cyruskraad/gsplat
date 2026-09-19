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

"""The trainer.

Built on top of ``config.py``, ``run.py``, ``eval.py`` and ``device.py`` rather
than beside them, so that provenance, the results ledger and the held-out-light
gate are properties of a run instead of things someone remembers to do.

Four decisions are worth naming.

**The loss is computed in the log1p domain**, not on linear radiance. The same
reason the metrics are: on a relighting capture the specular highlight is two
orders of magnitude above the diffuse surface, and an L1 on linear radiance is
very nearly a loss on the highlight alone. ``log1p`` is monotone, cheap, and
leaves the model free to represent the highlight -- it just stops the highlight
from being the only thing the gradient sees.

**It dies informatively.** A NaN is detected at the step that produced it and
reported with the frame, the parameter block and the gradient norms, because a
diverged relighting model renders black and every metric afterwards is a
measurement of nothing. ``check_finite`` and ``contract_chunked(validate=True)``
already exist for this.

**Evaluation runs the gate, not just the metrics.** Every evaluation reports
held-out-view and held-out-light side by side and appends a ledger row, so the
question that decides the project is answered continuously rather than once at
the end.

**Checkpoints carry the optimiser.** A resumed run continues the loss curve; a
resumed run that restarts it is not a resume, and the difference is invisible
unless something checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import Config, config_hash, load_config, to_dict
from .data.loader import SET_NAMES, Capture, FrameSplit, load_capture
from .device import (
    DeviceReport,
    autotune_chunks,
    configure_precision,
    preflight,
    seed_everything,
    select_device,
)
from .eval import (
    RelightingReport,
    constant_baseline_psnr,
    SheetEntry,
    SplitMetrics,
    append_evaluation,
    comparison_sheet,
    evaluate_set,
)
from .functional.atoms import make_sg_atoms, project_point_light
from .functional.nearfield import incident_radiance
from .functional.transport import check_finite
from .model import ATOM_PARAMETERS, PRIMITIVE_PARAMETERS, RelightSplats
from .run import RunDirectory, append_ledger, hash_directory

__all__ = ["Trainer", "TrainingDiverged", "build_model", "train", "main"]


class TrainingDiverged(RuntimeError):
    """Non-finite values reached the parameters. Named with where and when."""


# --- the objective ----------------------------------------------------------


def photometric_loss(
    prediction: Tensor,
    reference: Tensor,
    *,
    mask: Optional[Tensor] = None,
    ssim_weight: float = 0.2,
) -> Tuple[Tensor, Dict[str, float]]:
    """L1 plus a structural term, both in the ``log1p`` domain.

    ``log1p`` rather than linear radiance for the reason the metrics use a
    tonemap: a specular highlight two orders of magnitude above the diffuse
    surface would otherwise be nearly the whole gradient. It is monotone, so
    nothing is clipped and the model can still represent the highlight.
    """
    predicted = torch.log1p(prediction.clamp_min(0.0))
    target = torch.log1p(reference.clamp_min(0.0))
    difference = (predicted - target).abs()

    if mask is not None:
        weight = mask.unsqueeze(-1)
        total = weight.sum().clamp_min(1e-8) * predicted.shape[-1]
        l1 = (difference * weight).sum() / total
    else:
        l1 = difference.mean()

    parts = {"l1": float(l1.detach())}
    loss = l1
    if ssim_weight > 0.0:
        from .eval import ssim_map

        window = min(11, predicted.shape[0], predicted.shape[1])
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            similarity = ssim_map(
                predicted, target, data_range=1.0, window_size=window
            ).mean()
            loss = loss + ssim_weight * (1.0 - similarity)
            parts["ssim"] = float(similarity.detach())
    parts["loss"] = float(loss.detach())
    return loss, parts


def cosine_schedule(
    step: int, *, max_steps: int, warmup: int, min_scale: float
) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if max_steps <= warmup:
        return 1.0
    progress = (step - warmup) / max(max_steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_scale + (1.0 - min_scale) * cosine


# --- building the thing to train -------------------------------------------


def build_model(
    config: Config,
    capture: Capture,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> RelightSplats:
    """A model to start from: a fixed-light reconstruction if there is one.

    Initialising from a PLY is the single biggest accelerator available for a
    first run, which is why ``ModelConfig.init_ply`` exists. Without one the
    primitives are scattered on a shell sized to the cameras, which is a poor
    start and an honest one.
    """
    num_atoms = config.atoms.count
    if config.model.init_ply:
        model = RelightSplats.from_ply(
            config.model.init_ply,
            num_atoms=num_atoms,
            sharpness=config.atoms.sharpness,
            device=device,
            dtype=dtype,
        )
        return model

    centres = capture.camera_positions().to(device=device, dtype=dtype)
    radius = float(centres.norm(dim=-1).mean()) * 0.25
    count = config.model.init_count
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    directions = torch.nn.functional.normalize(
        torch.randn(count, 3, generator=generator), dim=-1
    ).to(device=device, dtype=dtype)
    radii = radius * torch.rand(count, 1, generator=generator).to(
        device=device, dtype=dtype
    ).add(0.25)

    axes, sharpnesses = make_sg_atoms(
        num_atoms, sharpness=config.atoms.sharpness, device=device, dtype=dtype
    )
    quats = torch.zeros(count, 4, device=device, dtype=dtype)
    quats[:, 0] = 1.0
    return RelightSplats(
        means=directions * radii,
        quats=quats,
        scales=torch.full(
            (count, 3), math.log(radius * 0.1), device=device, dtype=dtype
        ),
        opacities=torch.full((count,), -2.0, device=device, dtype=dtype),
        transport=torch.full(
            (count, 3, num_atoms), 0.1 / num_atoms, device=device, dtype=dtype
        ),
        atom_axes=axes,
        atom_sharpness=sharpnesses,
    )


# --- the trainer ------------------------------------------------------------


@dataclass
class TrainState:
    """Everything a resume needs that is not a tensor."""

    step: int = 0
    best_metric: float = float("-inf")
    best_step: int = -1
    evaluations_without_improvement: int = 0


class Trainer:
    """One run: a capture, a model, an optimiser, and the gate."""

    def __init__(
        self,
        config: Config,
        capture: Capture,
        run: RunDirectory,
        *,
        report: Optional[DeviceReport] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.capture = capture
        self.run = run
        self.dtype = dtype
        self.report = report or select_device(config.runtime.device)
        self.device = self.report.device
        self.backend = RelightSplats.resolve_backend(config.runtime.backend)

        self.split: FrameSplit = capture.split(
            num_val_views=config.data.num_val_views,
            num_test_views=config.data.num_test_views,
            num_val_lights=config.data.num_val_lights,
            num_test_lights=config.data.num_test_lights,
        )
        self.model = build_model(config, capture, device=self.device, dtype=dtype)
        self.model.requires_grad_(True, atoms=config.optim.learn_atoms)
        self.params = self.model.as_parameter_dict()
        # Rebuild the model *around* the dict. `nn.Parameter(t)` shares storage
        # with `t` but is a new leaf in the graph, so without this the model
        # renders from one set of tensors while the optimiser holds another:
        # every gradient lands on the model's copies, `params[...].grad` stays
        # None, and `optimizer.step()` is a no-op. The loss curve is flat and
        # nothing says why.
        self.model = RelightSplats.from_parameter_dict(
            self.params, self.model.atom_axes, self.model.atom_sharpness
        )
        self.optimizers = self._build_optimizers()
        self.state = TrainState()
        self.chunk_size = config.runtime.chunk_size
        self._baseline: Optional[float] = None

        run.log(
            event="trainer_ready",
            device=str(self.device),
            backend=self.backend,
            primitives=self.model.num_primitives,
            atoms=self.model.num_atoms,
            **self.split.counts(),
        )

    # -- setup --

    def _build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """One optimiser per parameter, which is what gsplat's densification
        expects: it rewrites a parameter and its optimiser state together."""
        optim = self.config.optim
        rates = {
            "means": optim.means_lr,
            "quats": optim.quats_lr,
            "scales": optim.scales_lr,
            "opacities": optim.opacities_lr,
            "transport": optim.transport_lr,
        }
        optimizers = {
            name: torch.optim.Adam(
                [{"params": [self.params[name]], "lr": rates[name], "name": name}],
                eps=1e-15,
            )
            for name in PRIMITIVE_PARAMETERS
        }
        if optim.learn_atoms:
            optimizers["atoms"] = torch.optim.Adam(
                [
                    {
                        "params": [
                            getattr(self.model, name) for name in ATOM_PARAMETERS
                        ],
                        "lr": optim.atoms_lr,
                        "name": "atoms",
                    }
                ],
                eps=1e-15,
            )
        return optimizers

    def _light_for(self, frame) -> Tensor:
        """Per-primitive light coefficients for one shot.

        Near-field by default: the flash is a point light about a metre away, so
        every primitive sees a different direction and a different falloff.
        Dividing that out analytically is what lets the learned atoms be
        functions of direction alone, which is the whole near-field-to-far-field
        argument in `docs/relighting-atlas.md`.
        """
        model = self.model
        position = frame.light_position.to(device=self.device, dtype=self.dtype)
        intensity = frame.light_intensity.to(device=self.device, dtype=self.dtype)
        if not self.config.model.near_field:
            direction = torch.nn.functional.normalize(position, dim=-1)
            return project_point_light(
                direction, intensity, model.atom_axes, model.atom_sharpness
            )
        directions, radiance = incident_radiance(
            model.means.detach(),
            position,
            intensity,
            profile_exponent=self.config.model.profile_exponent,
            reference_distance=frame.reference_distance,
        )
        return project_point_light(
            directions, radiance, model.atom_axes, model.atom_sharpness
        )

    def _render(self, index: int) -> Tuple[Tensor, Tensor]:
        frame = self.capture.frames[index]
        image, alpha = self.model.render_image(
            frame.viewmat.to(device=self.device, dtype=self.dtype),
            self.capture.intrinsics.to(device=self.device, dtype=self.dtype),
            self.capture.width,
            self.capture.height,
            self._light_for(frame),
            backend=self.backend,
            chunk_size=self.chunk_size,
            validate=self.config.runtime.validate_chunks,
        )
        return image.to(self.dtype), alpha.to(self.dtype)

    def _reference(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:
        image = self.capture.image(index).to(device=self.device, dtype=self.dtype)
        mask = self.capture.mask(index)
        if mask is not None:
            mask = mask.to(device=self.device, dtype=self.dtype)
        return image, mask

    # -- one step --

    def step(self, indices: Sequence[int]) -> Dict[str, float]:
        """One optimiser step over ``indices``, with gradient accumulation."""
        optim = self.config.optim
        scale = (
            cosine_schedule(
                self.state.step,
                max_steps=optim.max_steps,
                warmup=optim.warmup_steps,
                min_scale=optim.min_lr_scale,
            )
            if optim.schedule == "cosine"
            else 1.0
        )
        for name, optimizer in self.optimizers.items():
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", group["lr"]) * scale
                group.setdefault("initial_lr", group["lr"] / max(scale, 1e-12))
            optimizer.zero_grad(set_to_none=True)

        totals: Dict[str, float] = {}
        for index in indices:
            prediction, _ = self._render(index)
            reference, mask = self._reference(index)
            loss, parts = photometric_loss(
                prediction, reference, mask=mask, ssim_weight=optim.ssim_weight
            )
            (loss / len(indices)).backward()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value / len(indices)

        grad_norm = self._grad_norm()
        if not math.isfinite(grad_norm):
            raise TrainingDiverged(
                f"the gradient norm is {grad_norm} at step {self.state.step}, on "
                f"frames {list(indices)}. The model has diverged; every metric "
                f"after this point would be a measurement of nothing."
            )
        if optim.grad_clip > 0.0:
            for name in PRIMITIVE_PARAMETERS:
                torch.nn.utils.clip_grad_norm_([self.params[name]], optim.grad_clip)

        for optimizer in self.optimizers.values():
            optimizer.step()
        if optim.learn_atoms:
            self.model.normalise_atoms_()

        self._assert_finite(indices)
        self.state.step += 1
        totals["grad_norm"] = grad_norm
        totals["lr_scale"] = scale
        return totals

    def _grad_norm(self) -> float:
        total = 0.0
        for name in PRIMITIVE_PARAMETERS:
            grad = self.params[name].grad
            if grad is not None:
                total += float(grad.detach().pow(2).sum())
        return math.sqrt(total)

    def _assert_finite(self, indices: Sequence[int]) -> None:
        """Name the parameter block and the frames, not just the fact."""
        for name in PRIMITIVE_PARAMETERS:
            try:
                check_finite(self.params[name].detach(), name)
            except ValueError as error:
                raise TrainingDiverged(
                    f"step {self.state.step} on frames {list(indices)}: {error}"
                ) from error

    # -- evaluation, which is the gate --

    @torch.no_grad()
    def evaluate(self, *, sheet: bool = False) -> RelightingReport:
        """Score every held-out set and run the gate.

        Reports the pair, always. One of these numbers on its own cannot tell a
        model that learned transport from a model that memorised illuminations,
        and this is the measurement the project rests on.
        """
        splits: Dict[str, SplitMetrics] = {}
        entries: List[SheetEntry] = []
        for name in SET_NAMES:
            indices = self.split[name]
            if not indices:
                splits[name] = SplitMetrics(name=name, count=0, metrics={})
                continue
            pairs = []
            for index in indices:
                prediction, _ = self._render(index)
                reference, mask = self._reference(index)
                pairs.append(
                    (
                        prediction.cpu(),
                        reference.cpu(),
                        None if mask is None else mask.cpu(),
                    )
                )
                if sheet and len(entries) < 12 and name != "train":
                    frame = self.capture.frames[index]
                    entries.append(
                        SheetEntry(
                            f"{name.replace('_', ' ').upper()} "
                            f"V{frame.view_index:02d} L{frame.light_index:02d}",
                            reference.cpu(),
                            prediction.cpu(),
                            None if mask is None else mask.cpu(),
                        )
                    )
            splits[name] = evaluate_set(pairs, name=name, background=0.0)

        if sheet and entries:
            comparison_sheet(
                self.run.renders / f"sheet-{self.state.step:06d}.png", entries
            )
        return RelightingReport(
            held_out_view=splits["held_out_view"],
            held_out_light=splits["held_out_light"],
            train=splits["train"],
            baseline_psnr=self.baseline_psnr(),
        )

    def baseline_psnr(self) -> Optional[float]:
        """What a constant image already scores on the held-out views.

        Computed once and cached: it depends only on the data. Without it the
        gate is a difference, and a difference is satisfied by a model that is
        equally hopeless on both splits -- which is exactly what an untrained
        one is.
        """
        if self._baseline is None and self.split.held_out_view:
            references, masks = [], []
            for index in self.split.held_out_view:
                reference, mask = self._reference(index)
                references.append(reference.cpu())
                masks.append(None if mask is None else mask.cpu())
            self._baseline = constant_baseline_psnr(
                references, masks=masks if any(m is not None for m in masks) else None
            )
        return self._baseline

    # -- checkpoints --

    def save_checkpoint(self, name: str) -> Path:
        """Model, optimiser state and the step. All three, or it is not a resume."""
        path = self.run.ckpts / f"{name}.pt"
        torch.save(
            {
                "step": self.state.step,
                "best_metric": self.state.best_metric,
                "best_step": self.state.best_step,
                "params": {k: v.detach().cpu() for k, v in self.params.items()},
                "atoms": {
                    k: getattr(self.model, k).detach().cpu() for k in ATOM_PARAMETERS
                },
                "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
                "config_hash": config_hash(self.config),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore so the loss curve continues rather than restarting."""
        blob = torch.load(Path(path), map_location=self.device, weights_only=False)
        if blob.get("config_hash") != config_hash(self.config):
            raise ValueError(
                f"{path} was written by a different configuration "
                f"({blob.get('config_hash')} vs {config_hash(self.config)}). "
                f"Resuming across a config change would produce a run that "
                f"matches neither."
            )
        with torch.no_grad():
            for name, tensor in blob["params"].items():
                self.params[name].data = tensor.to(device=self.device, dtype=self.dtype)
            for name, tensor in blob["atoms"].items():
                getattr(self.model, name).data = tensor.to(
                    device=self.device, dtype=self.dtype
                )
        self.model = RelightSplats.from_parameter_dict(
            self.params, self.model.atom_axes, self.model.atom_sharpness
        )
        for name, optimizer in self.optimizers.items():
            if name in blob["optimizers"]:
                optimizer.load_state_dict(blob["optimizers"][name])
        self.state = TrainState(
            step=int(blob["step"]),
            best_metric=float(blob["best_metric"]),
            best_step=int(blob["best_step"]),
        )

    # -- the loop --

    def fit(
        self, *, max_steps: Optional[int] = None, sheet: bool = True
    ) -> RelightingReport:
        optim = self.config.optim
        steps = optim.max_steps if max_steps is None else max_steps
        order = list(self.split.train)
        if not order:
            raise ValueError("the split leaves no training frames")
        generator = torch.Generator().manual_seed(self.config.seed)
        started = time.time()

        while self.state.step < steps:
            picks = torch.randint(
                len(order), (max(optim.batch_size, 1),), generator=generator
            )
            metrics = self.step([order[int(i)] for i in picks])

            every = max(1, min(50, steps // 20))
            if self.state.step % every == 0 or self.state.step == 1:
                self.run.record_metrics(self.state.step, **metrics)
            if optim.eval_every and self.state.step % optim.eval_every == 0:
                if self._evaluate_and_maybe_stop(sheet=sheet):
                    break
            if optim.save_every and self.state.step % optim.save_every == 0:
                self.save_checkpoint(f"step-{self.state.step:06d}")

        report = self.evaluate(sheet=sheet)
        self.save_checkpoint("final")
        self.run.record_metrics(self.state.step, **report.as_row())
        self.run.log(
            event="fit_complete",
            steps=self.state.step,
            seconds=time.time() - started,
            gate_passed=report.gate().passed,
        )
        return report

    def _evaluate_and_maybe_stop(self, *, sheet: bool) -> bool:
        """Evaluate, record, keep the best, and say whether to stop."""
        report = self.evaluate(sheet=sheet)
        row = report.as_row()
        self.run.record_metrics(self.state.step, **row)
        print(report.format_table())

        # Best under a declared ordering: held-out-light PSNR, because that is
        # the number the project is about. A checkpoint chosen on training loss
        # would be chosen on the one number that cannot generalise.
        metric = report.held_out_light.get("psnr/mu")
        if metric is None:
            metric = report.held_out_view.get("psnr/mu", float("-inf"))
        if metric > self.state.best_metric:
            self.state.best_metric = float(metric)
            self.state.best_step = self.state.step
            self.save_checkpoint("best")
            self.state.evaluations_without_improvement = 0
        else:
            self.state.evaluations_without_improvement += 1

        patience = self.config.optim.early_stop_patience
        if patience and self.state.evaluations_without_improvement >= patience:
            self.run.log(
                event="early_stop",
                step=self.state.step,
                best_step=self.state.best_step,
                best_metric=self.state.best_metric,
            )
            return True
        return False


# --- the entry point --------------------------------------------------------


def train(
    config: Config,
    *,
    smoke: bool = False,
    resume: Optional[Path | str] = None,
    ledger: Optional[Path | str] = None,
    run_name: Optional[str] = None,
) -> Tuple[RunDirectory, RelightingReport]:
    """Set up, preflight, train, and put the answer in the ledger.

    The order matters. Device and precision are settled before anything is
    allocated, the preflight runs before the run directory exists so a refused
    run leaves no directory behind, and the report is appended to the ledger
    whether or not the gate passed -- a failure that is not recorded is a
    failure that gets repeated.
    """
    report = select_device(config.runtime.device)
    precision = configure_precision(allow_tf32=config.runtime.allow_tf32)
    seeding = seed_everything(config.seed, deterministic=config.runtime.deterministic)

    capture = load_capture(config.data.capture_dir)
    split = capture.split(
        num_val_views=config.data.num_val_views,
        num_test_views=config.data.num_test_views,
        num_val_lights=config.data.num_val_lights,
        num_test_lights=config.data.num_test_lights,
    )

    plan, warnings = preflight(
        num_primitives=4096 if not config.model.init_ply else 1_000_000,
        num_atoms=config.atoms.count,
        report=report,
        width=capture.width,
        height=capture.height,
        batch_size=max(config.optim.batch_size, 1),
        capture_frames=len(capture),
        trainable_frames=len(split.train),
        splits_are_independent=capture.splits_are_independent,
        run_root=config.run_root,
    )
    print(report)
    for warning in warnings:
        print(f"  warning: {warning}")

    if config.runtime.chunk_size == 0 and report.is_cuda:
        autotune_chunks(report)  # recorded below; the contraction sizes itself

    run = RunDirectory.create(
        config,
        name=run_name,
        dataset_hash=hash_directory(config.data.capture_dir, suffixes={".png"}),
    )
    with run:
        run.log(
            event="environment",
            device=report.to_dict(),
            precision=precision,
            seeding=seeding,
            memory_plan=plan.to_dict(),
            warnings=warnings,
        )
        trainer = Trainer(config, capture, run, report=report)
        if resume is not None:
            trainer.load_checkpoint(resume)
            run.log(event="resumed", step=trainer.state.step, path=str(resume))

        steps = 20 if smoke else None
        result = trainer.fit(max_steps=steps, sheet=not smoke)

    append_evaluation(
        ledger or Path(config.run_root) / "ledger.jsonl",
        result,
        run=run.path.name,
        config_hash=config_hash(config),
        step=trainer.state.step,
        provenance=json.loads((run.path / "provenance.json").read_text()),
        device=str(report.device),
        backend=trainer.backend,
        smoke=smoke,
    )
    return run, result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atlas.train", description="Train a relightable model."
    )
    parser.add_argument("config", nargs="*", type=Path, help="YAML layers, in order")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--capture", type=Path, help="shorthand for data.capture_dir")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:N")
    parser.add_argument("--backend", help="auto, gsplat, reference")
    parser.add_argument("--smoke", action="store_true", help="20 steps, then exit")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--name", help="run directory prefix")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = (
            json.loads(value) if value[:1] in '[{"0123456789-tfn' else value
        )
    if args.capture:
        overrides["data.capture_dir"] = str(args.capture)
    if args.device:
        overrides["runtime.device"] = args.device
    if args.backend:
        overrides["runtime.backend"] = args.backend

    config = load_config(args.config, overrides=overrides)
    if not config.data.capture_dir:
        parser.error("no capture: pass --capture PATH or set data.capture_dir")

    run, result = train(
        config,
        smoke=args.smoke,
        resume=args.resume,
        ledger=args.ledger,
        run_name=args.name,
    )
    print(f"\n{run.path}")
    return 0 if result.gate().passed or args.smoke else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
