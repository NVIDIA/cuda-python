# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the source-build and documentation packages for the Pixi smoke job."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from .bindings_config import BindingsConfig, BindingsPackage, load_config
from .compute_ci_plan import _changed_paths, _git_output, select_source_work


def select_pixi_packages(
    config: BindingsConfig,
    *,
    paths: list[str],
    linked_paths: set[str],
    cuda_env: str | None = None,
) -> tuple[list[BindingsPackage], list[BindingsPackage]]:
    """Use path impacts for PRs, or a requested/current ABI for manual runs."""
    packages = config.package_roots
    docs_roots: set[str] = set()
    if cuda_env is not None:
        selected = (
            [package for package in packages if package.cuda_variant == cuda_env]
            if cuda_env
            else [config.package_for_release_status("current")]
        )
        if not selected:
            registered = ", ".join(package.cuda_variant for package in packages)
            raise ValueError(f"cuda-env must be a registered pixi environment: {registered}")
    else:
        selection = select_source_work(config, paths, linked_paths)
        targets = selection["builds"] | selection["tests"]
        selected_roots = {
            package.package_root
            for package in packages
            if ("pathfinder", None) in targets
            or ("core", package.cuda_variant) in targets
            or any((component, package.package_root) in targets for component in ("bindings", "python"))
        }
        # The wheel path selection deliberately ignores Pixi manifests and docs.
        for path in paths:
            if PurePosixPath(path).name in {"pixi.toml", "pixi.lock"}:
                local = {
                    package.package_root
                    for package in packages
                    if path == f"{package.package_root}/{PurePosixPath(path).name}"
                }
                selected_roots.update(local or {package.package_root for package in packages})
            docs_roots.update(
                package.package_root for package in packages if path.startswith(f"{package.package_root}/docs/")
            )
        selected_roots.update(docs_roots)
        selected = [package for package in packages if package.package_root in selected_roots]
    if not selected:
        raise ValueError("no bindings package selected for source-build smoke tests")
    return selected, [package for package in packages if package.package_root in docs_roots]


def main() -> None:
    paths, linked_paths = [], set()
    cuda_env = None
    if os.environ["EVENT_NAME"] == "workflow_dispatch":
        cuda_env = os.environ["REQUESTED_CUDA_ENV"]
    else:
        merge_base = _git_output("merge-base", "HEAD", os.environ["BASE_SHA"]).decode().strip()
        paths, linked_paths = _changed_paths(merge_base)
    selected, docs = select_pixi_packages(load_config(), paths=paths, linked_paths=linked_paths, cuda_env=cuda_env)
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
        for name, packages in (("BINDINGS_PACKAGES", selected), ("BINDINGS_DOC_PACKAGES", docs)):
            output.write(f"{name}={json.dumps([package.to_dict() for package in packages], separators=(',', ':'))}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write("### Selected source builds\n")
        for package in selected:
            summary.write(f"- {package.package_root}: CUDA {package.cuda_major}, pixi {package.cuda_variant}\n")


if __name__ == "__main__":
    main()
