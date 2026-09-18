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

"""The two properties the configuration system exists for.

A typo must fail in the first second rather than the first hour, and the same
configuration must always produce the same hash — because that hash names the
run directory, joins the results ledger, and is what says whether two
experiments differ.
"""

import json

import pytest

from atlas.config import (
    AtomConfig,
    Config,
    ConfigError,
    apply_overrides,
    config_hash,
    deep_merge,
    from_dict,
    load_config,
    to_dict,
)

yaml = pytest.importorskip("yaml")


def _write(path, payload):
    path.write_text(yaml.safe_dump(payload))
    return path


# --- a typo fails immediately, and says what was meant ----------------------


def test_an_unknown_key_is_refused_and_suggests_the_right_one():
    """The failure this whole module exists to prevent.

    ``num_atms: 64`` in an unvalidated loader gives a six-hour run at the
    default 32 and no indication anything was wrong.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(AtomConfig, {"cont": 64})
    message = str(info.value)
    assert "unknown config key" in message
    assert "'cont'" in message
    assert "Did you mean 'count'?" in message


def test_an_unknown_nested_key_names_its_full_path():
    with pytest.raises(ConfigError, match=r"'atoms\.nonsense'"):
        from_dict(Config, {"atoms": {"nonsense": 1}})


def test_an_unknown_key_with_no_near_match_still_lists_the_valid_ones():
    with pytest.raises(ConfigError) as info:
        from_dict(AtomConfig, {"zzzzzz": 1})
    message = str(info.value)
    assert "Did you mean" not in message
    assert "'count'" in message and "'sharpness'" in message


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"atoms": {"count": "thirty"}}, "expected an integer"),
        ({"atoms": {"count": 3.5}}, "expected an integer"),
        ({"atoms": {"count": True}}, "expected an integer"),
        ({"seed": None}, "null is not allowed"),
        ({"model": {"near_field": "yes"}}, "expected true or false"),
        ({"optim": {"transport_lr": "fast"}}, "expected a number"),
        ({"run_root": 7}, "expected a string"),
        ({"atoms": 32}, "expected a section"),
    ],
)
def test_a_wrong_type_is_refused_with_the_reason(payload, message):
    with pytest.raises(ConfigError, match=message):
        from_dict(Config, payload)


def test_a_boolean_is_not_silently_accepted_as_an_integer():
    """``true`` is an ``int`` in Python. Letting it through would make
    ``count: true`` mean ``count: 1``."""
    with pytest.raises(ConfigError):
        from_dict(AtomConfig, {"count": True})


def test_a_non_finite_number_is_refused():
    """NaN has no canonical JSON form, so it would make the hash unstable."""
    with pytest.raises(ConfigError, match="finite"):
        from_dict(Config, {"optim": {"transport_lr": float("nan")}})


def test_optional_fields_accept_null():
    config = from_dict(Config, {"atoms": {"sharpness": None}})
    assert config.atoms.sharpness is None


# --- the hash ---------------------------------------------------------------


def test_identical_configurations_hash_identically_whatever_the_key_order():
    first = from_dict(Config, {"atoms": {"count": 64, "sharpness": 3.0}, "seed": 1})
    second = from_dict(Config, {"seed": 1, "atoms": {"sharpness": 3.0, "count": 64}})
    assert config_hash(first) == config_hash(second)


@pytest.mark.parametrize("written,equivalent", [(1.5, 1.50), (2, 2.0), (0.1, 0.10)])
def test_float_spelling_does_not_change_the_hash(written, equivalent):
    """``1.50`` and ``1.5`` are the same number and must be the same run."""
    a = from_dict(Config, {"optim": {"transport_lr": written}})
    b = from_dict(Config, {"optim": {"transport_lr": equivalent}})
    assert config_hash(a) == config_hash(b)


def test_an_integer_field_and_a_float_field_are_not_conflated():
    """Coercion happens per field, so ``count`` stays an int and cannot collide
    with a float that happens to have the same value."""
    config = from_dict(Config, {"atoms": {"count": 32}})
    assert isinstance(to_dict(config)["atoms"]["count"], int)


def test_a_different_value_changes_the_hash():
    base = from_dict(Config, {})
    for payload in (
        {"atoms": {"count": 33}},
        {"seed": 1},
        {"optim": {"max_steps": 30_001}},
        {"model": {"near_field": False}},
        {"data": {"num_test_lights": 7}},
    ):
        assert config_hash(from_dict(Config, payload)) != config_hash(base), payload


def test_the_hash_is_stable_across_processes():
    """It names directories and joins the ledger, so it cannot depend on a
    per-process seed such as ``PYTHONHASHSEED``."""
    import subprocess
    import sys

    code = (
        "from atlas.config import Config, config_hash, from_dict;"
        "print(config_hash(from_dict(Config, {'atoms': {'count': 17}})))"
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": "."},
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"hash varied across processes: {digests}"


def test_the_hash_is_a_full_sha256():
    assert len(config_hash(Config())) == 64
    assert all(c in "0123456789abcdef" for c in config_hash(Config()))


# --- layering ---------------------------------------------------------------


def test_layers_merge_deepest_last(tmp_path):
    base = _write(tmp_path / "base.yaml", {"atoms": {"count": 16}, "seed": 1})
    over = _write(tmp_path / "over.yaml", {"atoms": {"sharpness": 2.0}, "seed": 9})
    config = load_config([base, over])
    assert config.atoms.count == 16  # survived from the base layer
    assert config.atoms.sharpness == 2.0  # added by the overlay
    assert config.seed == 9  # replaced by the overlay


def test_a_scalar_in_the_overlay_replaces_rather_than_merges():
    merged = deep_merge({"a": {"b": 1}}, {"a": 2})
    assert merged == {"a": 2}


def test_dotted_overrides_apply_on_top(tmp_path):
    base = _write(tmp_path / "base.yaml", {"atoms": {"count": 16}})
    config = load_config([base], overrides={"atoms.count": 64, "seed": 3})
    assert config.atoms.count == 64
    assert config.seed == 3


def test_an_override_cannot_invent_a_section():
    """Otherwise ``--set atomz.count=64`` silently adds a branch nothing reads."""
    with pytest.raises(ConfigError, match="no section 'atomz'"):
        apply_overrides({"atoms": {"count": 1}}, {"atomz.count": 64})


def test_an_override_into_a_scalar_is_refused():
    with pytest.raises(ConfigError, match="is not a section"):
        apply_overrides({"seed": 1}, {"seed.deeper": 2})


def test_a_missing_config_file_names_itself(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config([tmp_path / "absent.yaml"])


def test_a_yaml_file_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config([path])


def test_an_empty_yaml_file_is_an_empty_layer(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert config_hash(load_config([path])) == config_hash(Config())


def test_loading_nothing_gives_the_defaults():
    assert config_hash(load_config([])) == config_hash(Config())


# --- round trip -------------------------------------------------------------


def test_a_config_round_trips_through_its_own_dict():
    original = from_dict(
        Config,
        {
            "atoms": {"count": 64, "sharpness": 12.5},
            "data": {"capture_dir": "/captures/angel", "num_test_lights": 9},
            "model": {"init_ply": "hq.ply", "near_field": False},
            "optim": {"max_steps": 1234},
            "seed": 7,
        },
    )
    restored = from_dict(Config, to_dict(original))
    assert config_hash(restored) == config_hash(original)
    assert restored == original


def test_the_dict_form_is_json_serialisable():
    """It is written into the run directory and the ledger."""
    json.dumps(to_dict(Config()))
