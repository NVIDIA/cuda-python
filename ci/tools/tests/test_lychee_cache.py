# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import lychee_cache


@pytest.fixture
def policy_checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    for relative in lychee_cache.POLICY_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "lychee-bin==0.24.2\n" if relative == ".pre-commit-config.yaml" else f"policy: {relative}\n"
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def invoke_cache_helper(monkeypatch, policy_checkout):
    cache_prefix = lychee_cache.cache_prefix
    monkeypatch.setattr(lychee_cache, "ROOT", policy_checkout)
    monkeypatch.setattr(lychee_cache, "cache_prefix", lambda kind: cache_prefix(kind, policy_checkout))
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    monkeypatch.delenv("LYCHEE_CACHE_DIR", raising=False)

    def invoke(kind, output):
        monkeypatch.setattr(sys, "argv", ["lychee_cache.py", kind, "--github-output", str(output)])
        lychee_cache.main()
        return dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())

    return invoke


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cache_prefix_is_shared_across_checkout_locations(policy_checkout, tmp_path):
    other_checkout = tmp_path / "different runner" / "cuda-python"
    for relative in lychee_cache.POLICY_FILES:
        path = other_checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((policy_checkout / relative).read_bytes())

    assert lychee_cache.cache_prefix("authored", policy_checkout) == lychee_cache.cache_prefix(
        "authored", other_checkout
    )


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_fragment_policy_has_a_separate_cache_namespace(policy_checkout):
    assert lychee_cache.cache_prefix("authored", policy_checkout) != lychee_cache.cache_prefix(
        "rendered", policy_checkout
    )


@pytest.mark.parametrize("relative", lychee_cache.POLICY_FILES)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_policy_changes_invalidate_existing_cache(policy_checkout, relative):
    before = lychee_cache.cache_prefix("authored", policy_checkout)
    path = policy_checkout / relative
    path.write_bytes(path.read_bytes() + b"changed policy\n")

    assert lychee_cache.cache_prefix("authored", policy_checkout) != before


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_native_wheel_upgrade_invalidates_existing_cache(policy_checkout):
    before = lychee_cache.cache_prefix("authored", policy_checkout)
    config = policy_checkout / ".pre-commit-config.yaml"
    config.write_text("lychee-bin==0.24.3\n", encoding="utf-8")

    assert lychee_cache.cache_prefix("authored", policy_checkout) != before


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_document_changes_keep_reusable_external_url_results(policy_checkout):
    before = lychee_cache.cache_prefix("authored", policy_checkout)
    (policy_checkout / "README.md").write_text("[New document](docs/new.md)\n", encoding="utf-8")
    (policy_checkout / ".lycheecache").write_text("successful URL checks\n", encoding="utf-8")

    assert lychee_cache.cache_prefix("authored", policy_checkout) == before


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_authored_output_is_relative_and_preserves_existing_step_outputs(tmp_path, monkeypatch, invoke_cache_helper):
    output = tmp_path / "github-output"
    output.write_text("previous-step=preserved\n", encoding="utf-8")
    monkeypatch.setenv("LYCHEE_CACHE_DIR", str(tmp_path / "rendered cache"))

    values = invoke_cache_helper("authored", output)

    assert values["previous-step"] == "preserved"
    assert values["path"] == ".lycheecache"
    assert values["key"] == f"{values['prefix']}12345-1"


@pytest.mark.parametrize("custom_directory", [False, True])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_rendered_output_uses_its_own_cache_directory(
    tmp_path, monkeypatch, policy_checkout, invoke_cache_helper, custom_directory
):
    directory = tmp_path / "rendered cache with spaces" if custom_directory else policy_checkout / ".lychee-rendered"
    if custom_directory:
        monkeypatch.setenv("LYCHEE_CACHE_DIR", str(directory))

    values = invoke_cache_helper("rendered", tmp_path / "github-output")

    assert values["path"] == str(directory / ".lycheecache")
    assert values["prefix"].startswith("lychee-precommit-v1-rendered-")


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_concurrent_runs_and_retries_get_distinct_immutable_snapshot_keys(tmp_path, monkeypatch, invoke_cache_helper):
    keys = set()
    prefixes = set()
    for run_id, attempt in [(12345, 1), (12345, 2), (12346, 1)]:
        monkeypatch.setenv("GITHUB_RUN_ID", str(run_id))
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", str(attempt))
        values = invoke_cache_helper("authored", tmp_path / f"output-{run_id}-{attempt}")
        assert values["key"] == f"{values['prefix']}{run_id}-{attempt}"
        prefixes.add(values["prefix"])
        keys.add(values["key"])

    assert len(keys) == 3
    assert len(prefixes) == 1


@pytest.mark.parametrize("variable", ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "LYCHEE_CACHE_DIR"])
@pytest.mark.parametrize("separator", ["\n", "\r"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_github_outputs_reject_values_that_could_inject_another_output(
    tmp_path, monkeypatch, invoke_cache_helper, variable, separator
):
    output = tmp_path / "github-output"
    monkeypatch.setenv(variable, f"valid{separator}injected=value")

    with pytest.raises(ValueError, match="contains a newline"):
        invoke_cache_helper("rendered", output)

    assert "injected=value" not in output.read_text(encoding="utf-8")
