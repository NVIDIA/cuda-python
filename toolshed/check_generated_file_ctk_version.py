# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from pathlib import Path

import yaml

_GENERATED_MARKER = "CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE:"
_VERSION_RE = re.compile(r"generated across versions from \S+ to (\d+\.\d+)")


def major_minor(version_str):
    major, minor = version_str.split(".")[:2]
    return (int(major), int(minor))


def main(argv):
    data = yaml.safe_load(Path("ci/versions.yml").read_text())
    latest_mm = major_minor(data["cuda"]["build"]["version"])

    failed = False
    for path in map(Path, argv):
        content = path.read_text(encoding="utf-8", errors="replace")
        if _GENERATED_MARKER not in content:
            continue
        m = _VERSION_RE.search(content)
        if m and major_minor(m.group(1)) > latest_mm:
            print(
                f"ERROR: {path}: generated up to CTK {m.group(1)}, which exceeds "
                f"ci/versions.yml cuda.build.version {'.'.join(str(x) for x in latest_mm)}. "
                f"Update ci/versions.yml if a new CTK release is intended."
            )
            failed = True

    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
