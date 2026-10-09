#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run through cuda_core's docs environment, which builds all three sibling
# packages from this checkout rather than installing their published releases.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

if [[ "${PIXI_ENVIRONMENT_NAME:-}" != "docs" ]]; then
    echo 'Run with: pixi run --manifest-path cuda_core/pixi.toml -e docs docs-build-all-latest' >&2
    exit 1
fi

export CUDA_PYTHON_DOCS_GITHUB_REF="${CUDA_PYTHON_DOCS_GITHUB_REF:-$(git rev-parse HEAD)}"
export BUILD_LATEST=1
export BUILD_PREVIEW=0

# Check canonical URLs against the final local output, including pages that
# have not been published yet. This override is only for the checking build.
CUDA_PYTHON_DOCS_DOMAIN=$(python -c 'from pathlib import Path; print(Path("artifacts/docs").resolve().as_uri())')
export CUDA_PYTHON_DOCS_DOMAIN

# The metapackage pins released cuda-core versions. Its dependencies are already
# installed from local Pixi source packages, so never resolve those pins on PyPI.
python -m pip install --no-deps "${REPO_ROOT}/cuda_python"
python - <<'PY'
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

for distribution, module_name in (
    ("cuda-bindings", "cuda.bindings"),
    ("cuda-core", "cuda.core"),
    ("cuda-pathfinder", "cuda.pathfinder"),
):
    module = import_module(module_name)
    source = Path.cwd() / distribution.replace("-", "_")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"{distribution} must be imported from the checkout: {module.__file__}")
    installed_version = version(distribution)
    if module.__version__ != installed_version:
        raise RuntimeError(
            f"{distribution} import version {module.__version__} differs from metadata {installed_version}"
        )
    print(f"{distribution}: {installed_version} ({module.__file__})")
print(f"cuda-python: {version('cuda-python')}")
PY

pushd cuda_python/docs >/dev/null
# The docs Makefiles use O as an optional Sphinx argument. Ignore an unrelated
# host variable of that name so Sphinx does not treat it as an input filename.
env -u O bash ./build_all_docs.sh latest-only
popd >/dev/null

rm -rf artifacts/docs
mkdir -p artifacts/docs
cp -a cuda_python/docs/build/html/. artifacts/docs/
