import ast
from pathlib import Path

import pytest


def _trainer_source() -> tuple[str, ast.FunctionDef]:
    source = Path("examples/simple_trainer.py").read_text()
    module = ast.parse(source)
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Runner"
    )
    train = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    return source, train


def test_completed_update_checkpoint_identity_is_explicit() -> None:
    from examples.checkpointing import completed_update_checkpoint

    identity = completed_update_checkpoint(loop_step=2999, world_rank=0)
    assert identity.loop_step == 2999
    assert identity.completed_updates == 3000
    assert identity.filename == "ckpt_3000_rank0.pt"
    with pytest.raises(ValueError, match="loop step"):
        completed_update_checkpoint(loop_step=-1, world_rank=0)
    with pytest.raises(ValueError, match="world rank"):
        completed_update_checkpoint(loop_step=0, world_rank=-1)


def test_scheduled_checkpoint_is_after_optimizer_and_structural_update() -> None:
    source, train = _trainer_source()
    body = ast.get_source_segment(source, train)
    assert body is not None
    optimizer = body.index("# optimize")
    structural = body.index("# Run post-backward steps after backward and optimizer.")
    completed_checkpoint = body.index("# save completed-update checkpoint")
    evaluation = body.index("# eval the full set")
    assert optimizer < structural < completed_checkpoint < evaluation
    assert "checkpoint.filename" in body


def test_checkpoint_payload_records_both_loop_and_completed_update_counts() -> None:
    source = Path("examples/simple_trainer.py").read_text()
    assert '"step": step' in source
    assert '"completed_updates": completed_updates' in source
