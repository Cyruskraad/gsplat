"""Small, dependency-free checkpoint iteration semantics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletedUpdateCheckpoint:
    loop_step: int
    completed_updates: int
    filename: str


def completed_update_checkpoint(
    *, loop_step: int, world_rank: int
) -> CompletedUpdateCheckpoint:
    """Name a post-update checkpoint by the number of completed updates."""
    if type(loop_step) is not int or loop_step < 0:
        raise ValueError("loop step must be a non-negative integer")
    if type(world_rank) is not int or world_rank < 0:
        raise ValueError("world rank must be a non-negative integer")
    completed_updates = loop_step + 1
    return CompletedUpdateCheckpoint(
        loop_step=loop_step,
        completed_updates=completed_updates,
        filename=f"ckpt_{completed_updates}_rank{world_rank}.pt",
    )
