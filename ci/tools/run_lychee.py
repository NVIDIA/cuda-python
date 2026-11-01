# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pre-commit's pinned native Lychee executable.

The upstream Lychee hook uses a shell launcher, which requires a compatible
shell on Windows. We configure a Python hook so pre-commit installs the pinned
lychee-bin wheel in an isolated environment. This wrapper runs that
environment's native executable, passing through arguments and returning
its exit status. The same hook works on Linux and in Windows shells such as
PowerShell, cmd, and Git Bash.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path

from checkout_source_links import repo_link_remaps

ROOT = Path(__file__).resolve().parents[2]


def rendered_inputs(docs_root: Path) -> list[Path]:
    """Select built HTML, excluding theme assets under _static directories."""
    return sorted(
        path
        for path in docs_root.rglob("*.html")
        if path.is_file() and "_static" not in path.relative_to(docs_root).parts
    )


def write_inputs(paths: list[Path], output: Path) -> None:
    """Reject an empty or ambiguous file list before invoking lychee."""
    if not paths:
        raise ValueError("No documentation inputs found for lychee")
    lines = [str(path.absolute()) for path in paths]
    if any("\n" in line or "\r" in line for line in lines):
        raise ValueError("Lychee input paths must not contain newlines")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    # Resolve the executable from this hook's environment to honor the pin.
    executable = Path(sysconfig.get_path("scripts")) / ("lychee.exe" if sys.platform == "win32" else "lychee")
    if not executable.is_file():
        raise SystemExit(
            f"lychee was not found at {executable}; run this script through the configured pre-commit hook"
        )
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    rendered = bool(arguments and arguments[0] == "--rendered")
    options = {}
    if rendered:
        arguments.pop(0)
        # Lychee writes .lycheecache into its working directory; keep the
        # fragment-aware rendered results apart from the authored cache in ROOT.
        cache = ROOT / ".lychee-rendered"
        cache.mkdir(parents=True, exist_ok=True)
        inputs = cache / "inputs.txt"
        write_inputs(rendered_inputs(ROOT / "artifacts/docs"), inputs)
        arguments = [
            "--files-from",
            str(inputs),
            "--include-fragments=full",
            "--config",
            str(ROOT / "lychee.toml"),
            *arguments,
        ]
        options["cwd"] = cache

    report = None
    if report_dir := os.environ.get("LYCHEE_REPORT_DIR"):
        directory = Path(report_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # pre-commit can split long file lists even with require_serial enabled.
        # Give each native invocation a unique report so one chunk's report
        # cannot overwrite another.
        report = directory / f"{uuid.uuid4().hex}.json"
        arguments = ["--format", "json", "--output", str(report), *arguments]

    code = subprocess.run(  # noqa: S603
        [str(executable), *repo_link_remaps(ROOT, rendered=rendered), *arguments], check=False, **options
    ).returncode
    if report is not None:
        report.with_suffix(".status.json").write_text(json.dumps({"exit_code": code}), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
