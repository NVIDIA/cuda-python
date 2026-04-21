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
    def test_plain_tag(self):
        assert parse_version_from_tag("v13.1.0") == "13.1.0"

    def test_component_prefix_core(self):
        assert parse_version_from_tag("cuda-core-v0.7.0") == "0.7.0"

    def test_component_prefix_pathfinder(self):
        assert parse_version_from_tag("cuda-pathfinder-v1.5.2") == "1.5.2"

    def test_post_release(self):
        assert parse_version_from_tag("v12.6.2.post1") == "12.6.2.post1"

    def test_invalid_tag(self):
        assert parse_version_from_tag("not-a-tag") is None

    def test_no_v_prefix(self):
        assert parse_version_from_tag("13.1.0") is None


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
