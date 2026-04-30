# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_release_notes import (
    check_release_notes,
    is_post_release,
    main,
    parse_version_from_tag,
)


class TestParseVersionFromTag:
    def test_plain_tag_bindings(self):
        assert parse_version_from_tag("v13.1.0", "cuda-bindings") == "13.1.0"

    def test_plain_tag_python(self):
        assert parse_version_from_tag("v13.1.0", "cuda-python") == "13.1.0"

    def test_component_prefix_core(self):
        assert parse_version_from_tag("cuda-core-v0.7.0", "cuda-core") == "0.7.0"

    def test_component_prefix_pathfinder(self):
        assert parse_version_from_tag("cuda-pathfinder-v1.5.2", "cuda-pathfinder") == "1.5.2"

    def test_post_release(self):
        assert parse_version_from_tag("v12.6.2.post1", "cuda-bindings") == "12.6.2.post1"

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
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", str(tmp_path))
        assert problems == []

    def test_missing(self, tmp_path):
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", str(tmp_path))
        assert len(problems) == 1
        assert problems[0][1] == "missing"

    def test_empty(self, tmp_path):
        self._make_notes(tmp_path, "cuda_core", "0.7.0", content="")
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-core", str(tmp_path))
        assert len(problems) == 1
        assert problems[0][1] == "empty"

    def test_post_release_skipped(self, tmp_path):
        problems = check_release_notes("v12.6.2.post1", "cuda-bindings", str(tmp_path))
        assert problems == []

    def test_invalid_tag(self, tmp_path):
        problems = check_release_notes("not-a-tag", "cuda-core", str(tmp_path))
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_component_prefix_mismatch(self, tmp_path):
        # Pass a cuda-core tag with component=cuda-pathfinder; must be rejected.
        problems = check_release_notes("cuda-core-v0.7.0", "cuda-pathfinder", str(tmp_path))
        assert len(problems) == 1
        assert "cannot parse" in problems[0][1]

    def test_unknown_component(self, tmp_path):
        problems = check_release_notes("v13.1.0", "bogus", str(tmp_path))
        assert len(problems) == 1
        assert "unknown component" in problems[0][1]

    def test_plain_v_tag(self, tmp_path):
        self._make_notes(tmp_path, "cuda_python", "13.1.0")
        problems = check_release_notes("v13.1.0", "cuda-python", str(tmp_path))
        assert problems == []


class TestMain:
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
        rc = main(["--git-tag", "v12.6.2.post1", "--component", "cuda-bindings", "--repo-root", str(tmp_path)])
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
