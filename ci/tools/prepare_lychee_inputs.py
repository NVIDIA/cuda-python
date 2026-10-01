# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write deterministic input lists for authored or rendered documentation links."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def authored_inputs(root: Path) -> list[Path]:
    """Select existing tracked Markdown and reStructuredText outside qa/."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "*.rst"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if not name or name.startswith("qa/"):
            continue
        path = root / name
        if path.is_file():
            paths.append(path)
    return sorted(paths)


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
    print(f"Wrote {len(lines)} lychee inputs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("authored", "rendered"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--docs-root", type=Path, default=Path("artifacts/docs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.kind == "authored":
        paths = authored_inputs(root)
    else:
        paths = rendered_inputs((root / args.docs_root).absolute())
    write_inputs(paths, args.output)


if __name__ == "__main__":
    main()
