# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bindings_config
import check_release_notes as release_notes_module
from check_release_notes import (
    check_release_notes,
    is_post_release,
    main,
    notes_path,
    parse_version_from_tag,
)


class TestParseVersionFromTag:
    def test_plain_tag_bindings(self):
        assert parse_version_from_tag("v13.3.0", "cuda-bindings") == "13.3.0"

    def test_plain_tag_python(self):
        assert parse_version_from_tag("v13.1.0", "cuda-python") == "13.1.0"

    def test_component_prefix_core(self):
        assert parse_version_from_tag("cuda-core-v0.7.0", "cuda-core") == "0.7.0"

    def test_component_prefix_pathfinder(self):
        assert parse_version_from_tag("cuda-pathfinder-v1.5.2", "cuda-pathfinder") == "1.5.2"

    def test_post_release(self):
        assert parse_version_from_tag("v12.9.8.post1", "cuda-bindings") == "12.9.8.post1"

    def test_invalid_tag(self):
        assert parse_version_from_tag("not-a-tag", "cuda-core") is None

    def test_no_v_prefix(self):
        assert parse_version_from_tag("13.1.0", "cuda-bindings") is None

    def test_component_prefix_mismatch(self):
        # cuda-core-v* must not be accepted for component=cuda-pathfinder
        assert parse_version_from_tag("cuda-core-v0.7.0", "cuda-pathfinder") is None

    def test_bare_v_rejected_for_core(self):
        # bare v* belongs to cuda-bindings/cuda-python, not cuda-core
        assert parse_version_from_tag("v0.7.0", "cuda-core") is None

    def test_unknown_component(self):
        assert parse_version_from_tag("v13.1.0", "bogus") is None

    def test_path_traversal_rejected(self):
        assert parse_version_from_tag("v1.0.0/../evil", "cuda-bindings") is None

    def test_path_separator_rejected(self):
        assert parse_version_from_tag("v1/2/3", "cuda-bindings") is None

    def test_leading_dot_rejected(self):
        assert parse_version_from_tag("v.1.0", "cuda-bindings") is None

    def test_whitespace_rejected(self):
        assert parse_version_from_tag("v1.0.0 ", "cuda-bindings") is None

    def test_trailing_suffix_rejected(self):
        # \w permits alphanumerics + underscore only; hyphens and shell meta-chars are out
        assert parse_version_from_tag("v1.0.0-extra", "cuda-bindings") is None


class TestIsPostRelease:
    def test_normal(self):
        assert not is_post_release("13.1.0")

    def test_post(self):
        assert is_post_release("12.6.2.post1")

    def test_post_no_number(self):
        assert is_post_release("1.0.0.post")


class TestCheckReleaseNotes:
    def _make_notes(self, tmp_path, pkg, version, content="Release notes."):
        d = tmp_path / pkg / "docs" / "source" / "release"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{version}-notes.rst"
        f.write_text(content)
        return f

    def test_present_and_nonempty(self, tmp_path):
        self._make_notes(tmp_path, "cuda_core", "0.7.0")
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert problems == []

    def test_missing(self, tmp_path):
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert problems[0][1] == "missing"

    def test_empty(self, tmp_path):
        self._make_notes(tmp_path, "cuda_core", "0.7.0", content="")
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert problems[0][1] == "empty"

    def test_post_release_skipped(self, tmp_path):
        problems = check_release_notes("v12.9.8.post1", "cuda-bindings", tmp_path)
        assert problems == []

    def test_invalid_tag(self, tmp_path):
        problems = check_release_notes("not-a-tag", "cuda-core", tmp_path)
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_component_prefix_mismatch(self, tmp_path):
        # Pass a cuda-core tag with component=cuda-pathfinder; must be rejected.
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-pathfinder", tmp_path)
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_unknown_component(self, tmp_path):
        problems = check_release_notes("v13.1.0", "bogus", tmp_path)
        assert len(problems) == 1
        assert "unknown component" in problems[0][1]

    def test_plain_v_tag(self, tmp_path):
        self._make_notes(tmp_path, "cuda_python", "13.1.0")
        problems = check_release_notes("v13.1.0", "cuda-python", tmp_path)
        assert problems == []

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_v12_bindings_notes_use_imported_tree(self, tmp_path):
        self._make_notes(tmp_path, "cuda_bindings_12", "12.9.8")
        problems = check_release_notes("v12.9.8", "cuda-bindings", tmp_path)
        assert problems == []

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_bindings_notes_use_registry_source_dir(self, tmp_path, monkeypatch):
        self._make_notes(tmp_path, "alternate_bindings", "13.2.0")
        config = Mock()
        config.match_tag.return_value = SimpleNamespace(source_dir="alternate_bindings")
        monkeypatch.setattr(release_notes_module.bindings_config, "load_config", lambda: config)

        problems = check_release_notes("v13.2.0", "cuda-bindings", tmp_path)

        assert problems == []
        config.match_tag.assert_called_once_with("v13.2.0")

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_resolved_legacy_bindings_source_dir(self, tmp_path):
        self._make_notes(tmp_path, "cuda_bindings", "12.9.8")
        line = bindings_config.load_config().line_to_dict(bindings_config.load_config().get_line("released-12"))
        line["release_source_dir"] = "cuda_bindings"

        problems = check_release_notes("v12.9.8", "cuda-bindings", tmp_path, line)

        assert problems == []

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    @pytest.mark.parametrize(
        ("component", "control_package"),
        (("cuda-bindings", "cuda_bindings_12"), ("cuda-python", "cuda_python")),
    )
    def test_control_registry_release_can_use_imported_notes(
        self,
        tmp_path,
        component,
        control_package,
    ):
        release_root = tmp_path / "release"
        release_root.mkdir()
        control_root = tmp_path / "control"
        self._make_notes(control_root, control_package, "12.9.7")
        line = bindings_config.load_config().line_to_dict(bindings_config.load_config().get_line("released-12"))
        line.update(
            {
                "release_source_dir": "cuda_bindings",
                "release_registry_origin": "control",
            }
        )

        problems = check_release_notes(
            "v12.9.7",
            component,
            release_root,
            line,
            control_root,
        )

        assert problems == []

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_tag_registry_release_does_not_fall_back_to_control_notes(self, tmp_path):
        release_root = tmp_path / "release"
        release_root.mkdir()
        control_root = tmp_path / "control"
        self._make_notes(control_root, "cuda_bindings_12", "12.9.7")
        line = bindings_config.load_config().line_to_dict(bindings_config.load_config().get_line("released-12"))
        line.update(
            {
                "release_source_dir": "cuda_bindings",
                "release_registry_origin": "tag",
            }
        )

        problems = check_release_notes(
            "v12.9.7",
            "cuda-bindings",
            release_root,
            line,
            control_root,
        )

        assert problems == [(notes_path("cuda_bindings", "12.9.7"), "missing")]

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_resolved_tag_line_does_not_depend_on_current_registry(self, tmp_path):
        self._make_notes(tmp_path, "alternate_bindings", "13.2.0")
        line = {
            "line_id": "alternate-13",
            "source_dir": "alternate_bindings",
            "release_source_dir": "alternate_bindings",
            "ctk_target": "13.2",
            "toolkit_version": "13.2.0",
            "toolkit_channel": "stable",
            "tag_series": "v13.2.",
            "allow_alpha_beta_tags": True,
        }

        problems = check_release_notes("v13.2.0", "cuda-bindings", tmp_path, line)

        assert problems == []


class TestMain:
    def _make_notes(self, tmp_path, pkg, version, content="Release notes."):
        d = tmp_path / pkg / "docs" / "source" / "release"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{version}-notes.rst").write_text(content)

    def test_success(self, tmp_path):
        d = tmp_path / "cuda_core" / "docs" / "source" / "release"
        d.mkdir(parents=True)
        (d / "0.7.0-notes.rst").write_text("Notes here.")
        rc = main(["--git-tag", "cuda-core-v0.7.0", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 0

    def test_failure(self, tmp_path):
        rc = main(["--git-tag", "cuda-core-v0.7.0", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 1

    def test_post_skip(self, tmp_path):
        rc = main(["--git-tag", "v12.9.8.post1", "--component", "cuda-bindings", "--repo-root", str(tmp_path)])
        assert rc == 0

    def test_unparsable_tag_returns_2(self, tmp_path):
        rc = main(["--git-tag", "not-a-tag", "--component", "cuda-core", "--repo-root", str(tmp_path)])
        assert rc == 2

    def test_path_traversal_returns_2(self, tmp_path):
        rc = main(["--git-tag", "v1.0.0/../evil", "--component", "cuda-bindings", "--repo-root", str(tmp_path)])
        assert rc == 2

    def test_component_prefix_mismatch_returns_2(self, tmp_path):
        rc = main(
            [
                "--git-tag",
                "cuda-core-v0.7.0",
                "--component",
                "cuda-pathfinder",
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert rc == 2

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_invalid_resolved_source_dir_returns_2(self, tmp_path):
        line = bindings_config.load_config().line_to_dict(bindings_config.load_config().get_line("released-12"))
        line["release_source_dir"] = "../cuda_bindings"

        rc = main(
            [
                "--git-tag",
                "v12.9.8",
                "--component",
                "cuda-bindings",
                "--repo-root",
                str(tmp_path),
                "--bindings-line",
                json.dumps(line),
            ]
        )

        assert rc == 2
