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

"""Run directories: never reused, honest about how they ended, hard to corrupt.

These are the properties that make a result quotable months later. A run that
overwrote another's artifacts, or that looks complete because nothing recorded
that it was killed, is worse than no run at all.
"""

import json
import os

import pytest

from atlas.config import Config, config_hash, from_dict
from atlas.run import (
    COMPLETED,
    FAILED,
    RUNNING,
    RunDirectory,
    append_ledger,
    collect_provenance,
    hash_directory,
    ledger_to_csv,
    read_ledger,
)

yaml = pytest.importorskip("yaml")


def _config(**payload):
    return from_dict(Config, payload)


# --- shape and identity ------------------------------------------------------


def test_a_new_run_has_the_documented_shape(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path, name="angel")
    for name in ("config.yaml", "config_hash.txt", "provenance.json", RUNNING):
        assert (run.path / name).is_file(), name
    assert run.ckpts.is_dir() and run.renders.is_dir()


def test_the_directory_name_carries_the_config_hash(tmp_path):
    config = _config(atoms={"count": 64})
    run = RunDirectory.create(config, root=tmp_path, name="angel")
    assert run.path.name.startswith("angel-")
    assert run.path.name.endswith(config_hash(config)[:8])
    assert (run.path / "config_hash.txt").read_text().strip() == config_hash(config)


def test_two_configs_land_in_differently_named_directories(tmp_path):
    a = RunDirectory.create(_config(atoms={"count": 16}), root=tmp_path, name="x")
    b = RunDirectory.create(_config(atoms={"count": 32}), root=tmp_path, name="x")
    assert a.path != b.path


def test_a_run_directory_is_never_reused(tmp_path):
    """Overwriting a run destroys the evidence for a result already quoted."""
    from datetime import datetime, timezone

    frozen = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
    config = _config()
    RunDirectory.create(config, root=tmp_path, name="x", now=frozen)
    with pytest.raises(FileExistsError):
        RunDirectory.create(config, root=tmp_path, name="x", now=frozen)


def test_the_resolved_config_is_what_gets_written(tmp_path):
    """Not the layered files it came from -- nobody should replay a merge by hand."""
    config = _config(atoms={"count": 64, "sharpness": 3.5}, seed=11)
    run = RunDirectory.create(config, root=tmp_path)
    written = yaml.safe_load((run.path / "config.yaml").read_text())
    assert written["atoms"] == {"count": 64, "sharpness": 3.5}
    assert written["seed"] == 11
    assert config_hash(from_dict(Config, written)) == config_hash(config)


def test_an_existing_run_can_be_reopened(tmp_path):
    created = RunDirectory.create(_config(), root=tmp_path)
    reopened = RunDirectory.open(created.path)
    assert reopened.path == created.path
    assert reopened.status == RUNNING


def test_reopening_something_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(NotADirectoryError):
        RunDirectory.open(tmp_path / "absent")


# --- how a run ended ---------------------------------------------------------


def test_status_is_a_file_so_a_killed_process_still_tells_the_truth(tmp_path):
    """A process that is OOM-killed writes nothing on its way out.

    A directory still marked RUNNING hours later is the honest record of that,
    where a log that simply stops could mean anything.
    """
    run = RunDirectory.create(_config(), root=tmp_path)
    assert run.status == RUNNING
    assert (run.path / RUNNING).is_file()


def test_the_context_manager_marks_completed(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    with run:
        pass
    assert run.status == COMPLETED
    assert not (run.path / RUNNING).exists()


def test_the_context_manager_marks_failed_and_re_raises(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with run:
            raise RuntimeError("boom")
    assert run.status == FAILED
    logged = [
        json.loads(line) for line in (run.path / "log.jsonl").read_text().splitlines()
    ]
    failure = [r for r in logged if r.get("event") == "run_failed"]
    assert failure and failure[0]["error_type"] == "RuntimeError"
    assert "boom" in failure[0]["error"]


def test_only_one_status_marker_exists_at_a_time(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    run.mark(COMPLETED)
    present = [s for s in (RUNNING, COMPLETED, FAILED) if (run.path / s).exists()]
    assert present == [COMPLETED]


def test_an_unknown_status_is_refused(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    with pytest.raises(ValueError, match="status must be one of"):
        run.mark("MOSTLY_FINE")


# --- metrics -----------------------------------------------------------------


def test_metrics_accumulate_and_the_latest_is_available(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(100, psnr=20.0)
    run.record_metrics(200, psnr=22.5)
    rows = list(run.metrics())
    assert [r["step"] for r in rows] == [100, 200]
    latest = json.loads((run.path / "metrics.json").read_text())
    assert latest["psnr"] == 22.5 and latest["step"] == 200


def test_the_csv_grows_a_column_when_a_later_step_reports_a_new_metric(tmp_path):
    """A fixed header would either drop the column or need rewriting anyway."""
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(1, psnr=20.0)
    run.record_metrics(2, psnr=21.0, lpips=0.13)
    lines = (run.path / "metrics.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "lpips", "psnr"]
    assert lines[1].split(",") == ["1", "", "20.0"]
    assert lines[2].split(",") == ["2", "0.13", "21.0"]


def test_a_value_containing_a_comma_is_quoted(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(1, note="held-out lights, second pass")
    text = (run.path / "metrics.csv").read_text()
    assert '"held-out lights, second pass"' in text


def test_a_corrupt_metrics_json_does_not_stop_the_run(tmp_path):
    """Losing the last update is acceptable; losing the run is not."""
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(1, psnr=20.0)
    (run.path / "metrics.json").write_text("{ this is not json")
    run.record_metrics(2, psnr=21.0)
    assert json.loads((run.path / "metrics.json").read_text())["psnr"] == 21.0


def test_a_line_torn_by_a_crash_is_skipped_not_fatal(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(1, psnr=20.0)
    with (run.path / "metrics.jsonl").open("a") as handle:
        handle.write('{"step": 2, "psnr": 21.')  # killed mid-write
    run.record_metrics(3, psnr=22.0)
    assert [r["step"] for r in run.metrics()] == [1, 3]


def test_writes_are_atomic_leaving_no_temporary_files_behind(tmp_path):
    run = RunDirectory.create(_config(), root=tmp_path)
    run.record_metrics(1, psnr=20.0)
    leftovers = [p.name for p in run.path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# --- provenance --------------------------------------------------------------


def test_provenance_records_the_commit_and_whether_the_tree_was_dirty():
    """The dirty flag matters as much as the commit.

    A result from a working tree with uncommitted changes is not reproducible
    from that commit, and recording the commit alone would imply it is.
    """
    provenance = collect_provenance()
    assert "git_commit" in provenance and "git_dirty" in provenance
    assert provenance["python"] and provenance["hostname"]
    assert "timestamp_utc" in provenance


def test_provenance_survives_not_being_in_a_git_repository(tmp_path):
    provenance = collect_provenance(source_root=tmp_path)
    assert provenance["git_commit"] is None
    assert provenance["git_dirty"] is None
    assert provenance["python"]  # the rest still populated


def test_the_dataset_hash_is_carried_into_provenance(tmp_path):
    run = RunDirectory.create(
        _config(), root=tmp_path, dataset_hash="sha256-manifest:3:abc"
    )
    written = json.loads((run.path / "provenance.json").read_text())
    assert written["dataset_hash"] == "sha256-manifest:3:abc"


# --- dataset hashing ---------------------------------------------------------


def _capture(root, names=("a.jpg", "b.jpg"), size=16):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"x" * size)
    return root


def test_the_manifest_hash_is_stable_and_names_its_method(tmp_path):
    root = _capture(tmp_path / "cap")
    first = hash_directory(root)
    assert first == hash_directory(root)
    assert first.startswith("sha256-manifest:2:")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: (r / "c.jpg").write_bytes(b"x" * 16), id="file added"),
        pytest.param(lambda r: (r / "a.jpg").unlink(), id="file removed"),
        pytest.param(lambda r: (r / "a.jpg").rename(r / "z.jpg"), id="file renamed"),
        pytest.param(lambda r: (r / "a.jpg").write_bytes(b"x" * 99), id="size changed"),
    ],
)
def test_the_manifest_hash_notices_what_actually_goes_wrong(tmp_path, mutate):
    root = _capture(tmp_path / "cap")
    before = hash_directory(root)
    mutate(root)
    assert hash_directory(root) != before


def test_the_manifest_hash_does_not_read_file_contents(tmp_path):
    """Deliberate: reading hundreds of gigabytes at the start of every run is
    not a reasonable thing to do. Same size, same name, same hash."""
    root = _capture(tmp_path / "cap")
    before = hash_directory(root)
    (root / "a.jpg").write_bytes(b"y" * 16)  # different bytes, same length
    assert hash_directory(root) == before
    # ...which is exactly why the content mode exists, and differs.
    assert hash_directory(root, include_contents=True) != before
    assert hash_directory(root, include_contents=True).startswith("sha256-content:")


def test_hashing_can_be_restricted_to_the_formats_that_matter(tmp_path):
    root = _capture(tmp_path / "cap")
    only_images = hash_directory(root, suffixes={".jpg"})
    (root / "notes.txt").write_text("scratch")
    assert hash_directory(root, suffixes={".jpg"}) == only_images
    assert hash_directory(root) != only_images


def test_hashing_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(NotADirectoryError):
        hash_directory(tmp_path / "absent")


# --- the ledger --------------------------------------------------------------


def test_the_ledger_accepts_rows_with_different_keys(tmp_path):
    """Later runs report metrics earlier ones did not, and must not disturb them."""
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, {"run": "a", "psnr": 20.0})
    append_ledger(path, {"run": "b", "psnr": 21.0, "lpips": 0.1})
    rows = read_ledger(path)
    assert [r["run"] for r in rows] == ["a", "b"]
    assert "lpips" not in rows[0] and rows[1]["lpips"] == 0.1


def test_the_ledger_is_append_only(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, {"run": "a"})
    first = path.read_text()
    append_ledger(path, {"run": "b"})
    assert path.read_text().startswith(first)


def test_reading_an_absent_ledger_is_empty_not_an_error(tmp_path):
    assert read_ledger(tmp_path / "nothing.jsonl") == []


def test_the_ledger_renders_to_a_table_with_the_union_of_columns(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, {"run": "a", "config_hash": "aaa", "psnr": 20.0})
    append_ledger(path, {"run": "b", "config_hash": "bbb", "lpips": 0.1})
    destination = ledger_to_csv(path, tmp_path / "ledger.csv")
    lines = destination.read_text().strip().splitlines()
    assert lines[0].startswith("run,config_hash")
    assert set(lines[0].split(",")) == {"run", "config_hash", "psnr", "lpips"}
    assert len(lines) == 3


def test_an_empty_ledger_renders_to_an_empty_table(tmp_path):
    destination = ledger_to_csv(tmp_path / "none.jsonl", tmp_path / "out.csv")
    assert destination.read_text() == ""


def test_concurrent_appends_do_not_lose_rows(tmp_path):
    """Two runs finishing at once must not corrupt each other's lines."""
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "ledger.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: append_ledger(path, {"run": i}), range(64)))
    rows = read_ledger(path)
    assert sorted(r["run"] for r in rows) == list(range(64))


def test_a_torn_ledger_line_does_not_swallow_the_next_run(tmp_path):
    """Where this bug would have hurt most.

    Without healing the missing newline, the next run's row fuses onto the torn
    one and both are lost — so a run that finished cleanly would simply not
    appear in the ledger, with nothing to indicate it was dropped.
    """
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, {"run": "a", "psnr": 20.0})
    with path.open("a") as handle:
        handle.write('{"run": "b", "psnr": 21.')  # killed mid-write
    append_ledger(path, {"run": "c", "psnr": 22.0})
    rows = read_ledger(path)
    assert [r["run"] for r in rows] == ["a", "c"]
