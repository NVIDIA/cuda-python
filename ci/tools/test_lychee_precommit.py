# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the configured Lychee pre-commit hook without scanning external links."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]


def run_hook(sources: dict[Path, bytes], missing: Path | None = None) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pre_commit",
            "run",
            "lychee",
            "--verbose",
            "--color",
            "never",
            "--files",
            *(source.relative_to(ROOT).as_posix() for source in sources),
        ],
        cwd=ROOT,
        env=dict(os.environ, SKIP=""),
        capture_output=True,
        encoding="utf-8",
        timeout=120,
    )
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    for source, original in sources.items():
        if source.read_bytes() != original:
            raise RuntimeError(f"The Lychee hook changed {source}")
    if "- hook id: lychee" not in output or "Skipped" in output:
        raise RuntimeError("The configured Lychee hook did not run")
    if missing is None:
        if result.returncode != 0:
            raise RuntimeError("The Lychee hook rejected existing checkout targets")
    elif (
        result.returncode == 0
        or "File not found" not in output
        or missing.as_uri() not in output
        or "- exit code: 2" not in output
    ):
        raise RuntimeError(f"Expected Lychee to report the missing local target {missing}")


def main() -> None:
    # Generate sources inside the checkout so pre-commit applies the real file filters.
    with TemporaryDirectory(prefix="lychee hook smoke ", dir=ROOT) as temporary:
        fixture = Path(temporary)
        directory = fixture / "new directory"
        directory.mkdir()
        target = fixture / "new source #$.py"
        target.write_text("value = 1\n", encoding="utf-8")
        relative = fixture / "relative guide.rst"
        relative.write_text("Relative target\n", encoding="utf-8")
        prefix = "https://github.com/NVIDIA/cuda-python"
        file_url = f"{prefix}/blob/main/{quote(target.relative_to(ROOT).as_posix())}"
        directory_url = f"{prefix}/tree/main/{quote(directory.relative_to(ROOT).as_posix())}/"
        sources = {
            fixture / "source guide.md": (
                f"[File]({file_url})\r\n[Directory]({directory_url})\r\n[Relative](relative%20guide.rst)\r\n"
            ).encode(),
            fixture / "source guide.rst": (
                f"`File <{file_url}>`_\n`Directory <{directory_url}>`_\n`Relative <relative%20guide.rst>`_\n"
            ).encode(),
        }
        for source, content in sources.items():
            source.write_bytes(content)

        print("Checking existing local file, directory, and relative targets", flush=True)
        run_hook(sources)
        target.unlink()
        print("Checking that a previously valid file is rejected after deletion", flush=True)
        run_hook(sources, missing=target)
        target.write_text("value = 1\n", encoding="utf-8")
        directory.rmdir()
        print("Checking that a previously valid directory is rejected after deletion", flush=True)
        run_hook(sources, missing=directory)

    print("Configured Lychee hook checks passed; documentation sources were unchanged.")


if __name__ == "__main__":
    main()
