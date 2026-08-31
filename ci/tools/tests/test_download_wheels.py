# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

DOWNLOAD_WHEELS = Path(__file__).parent.parent / "download-wheels"

FAKE_GH = r"""#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_GH_LOG"]).open("a", encoding="utf-8") as stream:
    print(json.dumps(args), file=stream)

if args[:1] == ["api"]:
    if "--paginate" not in args or "--jq" not in args:
        print("artifact lookup must be paginated and filtered", file=sys.stderr)
        raise SystemExit(2)
    for artifact in json.loads(os.environ["FAKE_ARTIFACTS"]):
        if not artifact.get("expired", False):
            print(artifact["name"])
    raise SystemExit(0)

if args[:2] == ["run", "download"]:
    if "--name" in args:
        artifact_name = args[args.index("--name") + 1]
    elif "-p" in args:
        pattern = args[args.index("-p") + 1]
        artifact_name = pattern.replace("*", "wheel")
    else:
        print("download requires --name or -p", file=sys.stderr)
        raise SystemExit(2)
    if "--dir" not in args:
        print("download requires --dir", file=sys.stderr)
        raise SystemExit(2)
    artifact_dir = Path(args[args.index("--dir") + 1]) / artifact_name
    artifact_dir.mkdir()
    wheel_stem = re.sub(r"[^A-Za-z0-9]+", "_", artifact_name).strip("_")
    (artifact_dir / f"{wheel_stem}-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    raise SystemExit(0)

print(f"unexpected gh arguments: {args!r}", file=sys.stderr)
raise SystemExit(2)
"""


@pytest.fixture
def fake_gh(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    return fake_bin


def resolved_line(*, toolkit_version="13.3.0", origin="tag"):
    return json.dumps(
        {
            "toolkit_version": toolkit_version,
            "release_registry_origin": origin,
        },
        separators=(",", ":"),
    )


def run_download(
    tmp_path,
    fake_gh,
    artifacts,
    component,
    *,
    git_tag="",
    bindings_line_json=None,
):
    log = tmp_path / "gh.log"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ARTIFACTS": json.dumps([{"name": name, "expired": False} for name in artifacts]),
            "FAKE_GH_LOG": str(log),
            "GH_TOKEN": "test-token",
            "PATH": f"{fake_gh}{os.pathsep}{env['PATH']}",
        }
    )
    args = [
        str(DOWNLOAD_WHEELS),
        "123",
        component,
        "NVIDIA/cuda-python",
        str(tmp_path / "dist"),
    ]
    if git_tag or bindings_line_json is not None:
        args.append(git_tag)
    if bindings_line_json is not None:
        args.append(bindings_line_json)
    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        args,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    commands = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, commands


def downloaded_names(commands):
    return [command[command.index("--name") + 1] for command in commands if "--name" in command]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
class TestDownloadWheelsReleaseRouting:
    def test_modern_python_prefers_exact_line_artifact(self, tmp_path, fake_gh):
        exact = "cuda-python-wheel-cuda13.3.0"
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel", "cuda-python-wheel-cuda13.2.0", exact],
            "cuda-python",
            git_tag="v13.3.1",
            bindings_line_json=resolved_line(),
        )

        assert result.returncode == 0, result.stderr
        assert downloaded_names(commands) == [exact]

    def test_control_python_falls_back_to_exact_legacy_artifact(self, tmp_path, fake_gh):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel-old", "cuda-python-wheel"],
            "cuda-python",
            git_tag="v12.9.8",
            bindings_line_json=resolved_line(toolkit_version="12.9.1", origin="control"),
        )

        assert result.returncode == 0, result.stderr
        assert downloaded_names(commands) == ["cuda-python-wheel"]

    def test_tag_registry_rejects_legacy_python_fallback(self, tmp_path, fake_gh):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel"],
            "cuda-python",
            git_tag="v13.3.1",
            bindings_line_json=resolved_line(),
        )

        assert result.returncode != 0
        assert downloaded_names(commands) == []
        assert "legacy cuda-python-wheel fallback is not allowed" in result.stderr

    def test_bindings_artifacts_use_the_exact_toolkit_pin(self, tmp_path, fake_gh):
        exact = [
            "cuda-bindings-python310-cuda13.3.0-linux-64-sha",
            "cuda-bindings-python314-cuda13.3.0-win-64-sha",
        ]
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-bindings-python310-cuda13.3.00-linux-64-sha", *exact],
            "cuda-bindings",
            git_tag="v13.3.1",
            bindings_line_json=resolved_line(),
        )

        assert result.returncode == 0, result.stderr
        assert downloaded_names(commands) == exact

    def test_release_routing_requires_resolved_line_json(self, tmp_path, fake_gh):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel-cuda13.3.0"],
            "cuda-python",
            git_tag="v13.3.1",
        )

        assert result.returncode != 0
        assert commands == []
        assert "resolved bindings-line JSON is required" in result.stderr

    def test_nonbindings_five_argument_call_keeps_pattern_download(self, tmp_path, fake_gh):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            [],
            "cuda-core",
            git_tag="cuda-core-v1.1.1",
        )

        assert result.returncode == 0, result.stderr
        assert any("-p" in command and "cuda-core*" in command for command in commands)
