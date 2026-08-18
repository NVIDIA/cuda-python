#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform commands used by Moon's local and CI task graph.

Moon intentionally uses the system toolchain for this repository. These
commands consume the Python environment prepared by a contributor or CI and
continue to delegate local development tasks to Pixi.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATHS = {
    "root": Path("."),
    "ci": Path("ci"),
    "pathfinder": Path("cuda_pathfinder"),
    "bindings": Path("cuda_bindings"),
    "core": Path("cuda_core"),
    "metapackage": Path("cuda_python"),
    "bindings-benchmarks": Path("benchmarks/cuda_bindings"),
}
DOC_TASKS = {
    "pathfinder": "build-docs",
    "bindings": "build-docs",
    "core": "docs-build",
}
PACKAGE_PROJECTS = ("pathfinder", "bindings", "core", "metapackage")
CYTHON_PROJECTS = ("bindings", "core")
GATE_MARKERS = {
    "force-all",
    "force-all-unowned",
    "build-portable",
    "build-linux-64",
    "build-linux-aarch64",
    "build-windows",
    "test-sdist-linux",
    "test-sdist-windows",
    "test-linux",
    "test-windows",
    "docs",
    "core-api",
    "build-pathfinder",
    "build-bindings",
    "build-core",
    "build-metapackage",
    "test-pathfinder",
    "test-bindings",
    "test-core",
    "test-metapackage",
}


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print(f"+ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603


def _project_path(project: str) -> Path:
    try:
        relative = PROJECT_PATHS[project]
    except KeyError as error:
        raise ValueError(f"unknown project: {project}") from error
    return REPO_ROOT / relative


def _output_path(project: str, directory: str) -> Path:
    repo_root = Path(os.path.abspath(REPO_ROOT))
    project_root = Path(os.path.abspath(_project_path(project)))
    if project_root != repo_root and repo_root not in project_root.parents:
        raise ValueError(f"project must be within {repo_root}: {project_root}")
    output_root = project_root / ".moon-out"
    output = Path(os.path.abspath(output_root / directory))
    if output != output_root and output_root not in output.parents:
        raise ValueError(f"output must be within {output_root}: {output}")
    current = output
    while current != repo_root:
        if current.is_symlink():
            raise ValueError(f"output path must not traverse a symlink: {current}")
        current = current.parent
    return output


def _reset_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _find_one(directory: Path, pattern: str, description: str) -> Path:
    selected = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(selected) != 1:
        raise RuntimeError(f"expected one {description} in {directory}, found {len(selected)}")
    return selected[0]


def _find_one_in(directories: list[Path], pattern: str, description: str) -> Path:
    for directory in directories:
        selected = sorted(path for path in directory.glob(pattern) if path.is_file())
        if len(selected) == 1:
            return selected[0]
        if len(selected) > 1:
            raise RuntimeError(f"expected one {description} in {directory}, found {len(selected)}")
    searched = ", ".join(str(path) for path in directories)
    raise RuntimeError(f"expected one {description}; searched {searched}")


def _artifact_wheel(project: str, lane: str) -> Path:
    if project == "pathfinder":
        directories = [_output_path(project, "wheel-pure"), _project_path(project)]
    elif project == "bindings":
        environment = os.environ.get("CUDA_BINDINGS_ARTIFACTS_DIR")
        directories = [_output_path(project, f"wheel-{lane}")]
        if lane == "previous":
            directories.append(_project_path(project) / "dist-prev")
        elif environment:
            directories.append(Path(environment))
        directories.append(_project_path(project) / "dist")
    elif project == "core":
        environment = os.environ.get("CUDA_CORE_ARTIFACTS_DIR")
        directories = [_output_path(project, f"wheel-{lane}")]
        if environment:
            directories.append(Path(environment))
        directories.append(_project_path(project) / "dist")
    elif project == "metapackage":
        directories = [_output_path(project, "wheel-pure"), REPO_ROOT, _project_path(project)]
    else:
        raise ValueError(f"project does not produce wheel artifacts: {project}")
    return _find_one_in(directories, "*.whl", f"{project} {lane} wheel")


def _copy_files(source: Path, output: Path, patterns: tuple[str, ...]) -> None:
    selected = sorted({path for pattern in patterns for path in source.glob(pattern) if path.is_file()})
    if not selected:
        raise RuntimeError(f"no matching files found in {source}")
    _reset_output(output)
    for source_path in selected:
        shutil.copy2(source_path, output / source_path.name)


def _gate(args: argparse.Namespace) -> None:
    if args.marker not in GATE_MARKERS:
        raise ValueError(f"unknown CI gate marker: {args.marker}")
    output = _output_path("ci", f"ci-gates/{args.marker}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("true\n", encoding="utf-8")


def _pixi_run(project: str, task: str, *, environment: str | None, extra: list[str]) -> None:
    manifest = _project_path(project) / "pixi.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Pixi manifest not found: {manifest}")
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("pixi is required for this task but was not found on PATH")
    command = [pixi, "run", "--manifest-path", str(manifest)]
    selected_environment = environment or os.environ.get("PIXI_ENVIRONMENT_NAME")
    if selected_environment:
        command.extend(["--environment", selected_environment])
    command.append(task)
    command.extend(extra)
    _run(command)


def _pure_wheel(args: argparse.Namespace) -> None:
    if args.project not in {"pathfinder", "metapackage"}:
        raise ValueError("pure-wheel only supports pathfinder and metapackage")
    project_root = _project_path(args.project)
    output = _output_path(args.project, "wheel-pure")
    _reset_output(output)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--verbose",
            "--no-deps",
            "--wheel-dir",
            str(output),
            ".",
        ],
        cwd=project_root,
    )
    _find_one(output, "*.whl", f"{args.project} wheel")


def _cuda_major(lane: str) -> str:
    variable = "BUILD_CUDA_MAJOR" if lane == "current" else "BUILD_PREV_CUDA_MAJOR"
    value = os.environ.get(variable, "")
    if value:
        return value
    if lane == "current":
        version = os.environ.get("BUILD_CUDA_VER") or os.environ.get("CUDA_VER", "")
        if version:
            return version.split(".", maxsplit=1)[0]
    raise RuntimeError(f"{variable} is required for the {lane} CUDA lane")


def _constraint_uri(path: Path, *, in_linux_container: bool) -> str:
    resolved = path.resolve()
    if in_linux_container:
        return f"file:///host{resolved.as_posix()}"
    return resolved.as_uri()


def _constraint_environment(
    project: str,
    lane: str,
    *,
    cibuildwheel: bool,
    from_sdist: bool = False,
) -> dict[str, str]:
    if project not in {"bindings", "core"}:
        return os.environ.copy()

    constraints = _output_path(project, f"constraints-{lane}")
    _reset_output(constraints)
    constraint_file = constraints / "build.txt"
    linux_container = cibuildwheel and os.name != "nt"
    pathfinder_wheel = (
        _find_one(_output_path("pathfinder", "sdist"), "*.whl", "cuda.pathfinder sdist wheel")
        if from_sdist
        else _artifact_wheel("pathfinder", "pure")
    )
    requirements = [("cuda-pathfinder", pathfinder_wheel)]
    if project == "core":
        bindings_wheel = (
            _find_one(_output_path("bindings", "sdist"), "*.whl", "cuda.bindings sdist wheel")
            if from_sdist
            else _artifact_wheel("bindings", lane)
        )
        requirements.append(
            (
                "cuda-bindings",
                bindings_wheel,
            )
        )
    constraint_file.write_text(
        "".join(
            f"{distribution} @ {_constraint_uri(wheel, in_linux_container=linux_container)}\n"
            for distribution, wheel in requirements
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    host_constraint = str(constraint_file.resolve())
    environment["PIP_BUILD_CONSTRAINT"] = host_constraint
    environment["PIP_CONSTRAINT"] = host_constraint
    if project == "core":
        environment["CUDA_CORE_BUILD_MAJOR"] = _cuda_major(lane)

    if cibuildwheel:
        setting = "CIBW_ENVIRONMENT_WINDOWS" if os.name == "nt" else "CIBW_ENVIRONMENT_LINUX"
        container_constraint = f"/host{constraint_file.resolve().as_posix()}" if linux_container else host_constraint
        additions = [
            f'PIP_BUILD_CONSTRAINT="{container_constraint}"',
            f'PIP_CONSTRAINT="{container_constraint}"',
        ]
        if project == "core":
            additions.append(f"CUDA_CORE_BUILD_MAJOR={_cuda_major(lane)}")
        environment[setting] = " ".join(filter(None, [environment.get(setting, ""), *additions]))
    return environment


def _ensure_owned(output: Path) -> None:
    if os.name == "nt":
        return
    owners = {path.stat().st_uid for path in output.rglob("*")}
    if not owners or owners == {os.getuid()}:
        return
    sudo = shutil.which("sudo")
    if sudo is None:
        raise RuntimeError(f"cibuildwheel output is not owned by this user and sudo was not found: {output}")
    _run([sudo, "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(output)])


def _native_wheel(args: argparse.Namespace) -> None:
    if args.project not in {"bindings", "core"}:
        raise ValueError("native-wheel only supports bindings and core")
    if args.project == "bindings" and args.lane != "current":
        raise ValueError("cuda.bindings is only built in the current lane")

    output = _output_path(args.project, f"wheel-{args.lane}")
    _reset_output(output)
    environment = _constraint_environment(args.project, args.lane, cibuildwheel=True)
    _run(
        [
            sys.executable,
            "-m",
            "cibuildwheel",
            "--output-dir",
            str(output),
            str(_project_path(args.project)),
        ],
        env=environment,
    )
    _ensure_owned(output)
    wheel = _find_one(output, "*.whl", f"{args.project} {args.lane} wheel")
    if args.project == "core":
        renamed = wheel.with_name(f"{wheel.stem}.cu{_cuda_major(args.lane)}.whl")
        wheel.rename(renamed)


def _sdist(args: argparse.Namespace) -> None:
    project_root = _project_path(args.project)
    output = _output_path(args.project, "sdist")
    _reset_output(output)
    environment = (
        _constraint_environment(args.project, "current", cibuildwheel=False, from_sdist=True)
        if args.project in {"bindings", "core"}
        else os.environ.copy()
    )
    _run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(output), str(project_root)],
        env=environment,
    )
    archive = _find_one(output, "*.tar.gz", f"{args.project} source distribution")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output),
            str(archive),
        ],
        env=environment,
    )
    _find_one(output, "*.whl", f"{args.project} wheel from source distribution")


def _merge_core_wheels(_args: argparse.Namespace) -> None:
    current = _find_one(_output_path("core", "wheel-current"), "*.whl", "current cuda.core wheel")
    previous = _find_one(_output_path("core", "wheel-previous"), "*.whl", "previous cuda.core wheel")
    output = _output_path("core", "wheel-merged")
    _reset_output(output)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "ci" / "tools" / "merge_cuda_core_wheels.py"),
            str(current),
            str(previous),
            "--output-dir",
            str(output),
        ]
    )
    _find_one(output, "*.whl", "merged cuda.core wheel")


def _pixi_test(args: argparse.Namespace) -> None:
    _pixi_run(args.project, args.task, environment=args.environment, extra=args.extra)


def _pixi_docs(args: argparse.Namespace) -> None:
    _pixi_run(args.project, DOC_TASKS[args.project], environment="docs", extra=[])


def _docs_ci(_args: argparse.Namespace) -> None:
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required to build the combined documentation")
    latest_only = (os.environ.get("CUDA_PYTHON_DOCS_LATEST_ONLY") or "true").lower()
    if latest_only not in {"0", "1", "false", "true"}:
        raise ValueError("CUDA_PYTHON_DOCS_LATEST_ONLY must be true, false, 1, or 0")
    arguments = [bash, "build_all_docs.sh"]
    if latest_only in {"1", "true"}:
        arguments.append("latest-only")
    docs_root = _project_path("metapackage") / "docs"
    _run(arguments, cwd=docs_root)
    source = docs_root / "build" / "html"
    if not source.is_dir():
        raise RuntimeError(f"combined documentation output not found: {source}")
    output = _output_path("root", "docs")
    _reset_output(output)
    shutil.copytree(source, output, dirs_exist_ok=True)


def _cython_test_assets(args: argparse.Namespace) -> None:
    source = _project_path(args.project) / "tests" / "cython"
    wheels = [_artifact_wheel("pathfinder", "pure")]
    if args.project == "bindings":
        wheels.append(_artifact_wheel("bindings", "current"))
    else:
        wheels.extend(
            [
                _artifact_wheel("bindings", "current"),
                _artifact_wheel("core", "current"),
            ]
        )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *(str(wheel) for wheel in wheels),
            "--group",
            f"{_project_path(args.project) / 'pyproject.toml'}:test",
        ]
    )
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required to build Cython test extensions")
    _run([bash, "build_tests.sh"], cwd=source)
    output = _output_path(args.project, "cython-tests")
    _copy_files(source, output, ("test_*.so", "test_*.pyd", "test_*.dylib"))


def _core_test_binaries(_args: argparse.Namespace) -> None:
    source = _project_path("core") / "tests" / "test_binaries"
    _run([sys.executable, str(source / "build_test_binaries.py")])
    output = _output_path("core", "test-binaries")
    _copy_files(source, output, ("*.o", "*.a", "*.lib"))


def _stage_files(source: Path, destination: Path, pattern: str) -> None:
    files = sorted(path for path in source.glob(pattern) if path.is_file())
    if not files:
        raise RuntimeError(f"no files matching {pattern} found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, destination / path.name)


def _installed_test(args: argparse.Namespace) -> None:
    pathfinder_wheel = _artifact_wheel("pathfinder", "pure")
    if pathfinder_wheel.parent != _project_path("pathfinder"):
        _stage_files(pathfinder_wheel.parent, _project_path("pathfinder"), pathfinder_wheel.name)
    environment = os.environ.copy()
    if args.project in {"bindings", "core"}:
        environment.setdefault("CUDA_BINDINGS_ARTIFACTS_DIR", str(_output_path("bindings", "wheel-current")))
        _stage_files(
            _output_path(args.project, "cython-tests"),
            _project_path(args.project) / "tests" / "cython",
            "test_*.*",
        )
    if args.project == "core":
        environment.setdefault("CUDA_CORE_ARTIFACTS_DIR", str(_output_path("core", "wheel-merged")))
        _stage_files(
            _output_path("core", "test-binaries"),
            _project_path("core") / "tests" / "test_binaries",
            "*.*",
        )
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required by ci/tools/run-tests but was not found on PATH")
    _run([bash, str(REPO_ROOT / "ci" / "tools" / "run-tests"), args.project], env=environment)


def _metapackage_install_test(_args: argparse.Namespace) -> None:
    wheels = [
        _artifact_wheel("pathfinder", "pure"),
        _artifact_wheel("bindings", "current"),
        _artifact_wheel("core", "merged"),
    ]
    metapackage = _artifact_wheel("metapackage", "pure")
    requirement = str(metapackage)
    if os.environ.get("LOCAL_CTK", "1") != "1":
        requirement += "[all]"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            *(str(wheel) for wheel in wheels),
            requirement,
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate", help="write one affected CI allocation marker")
    gate.add_argument("marker", choices=tuple(sorted(GATE_MARKERS)))
    gate.set_defaults(handler=_gate)

    pure_wheel = subparsers.add_parser("pure-wheel", help="build one pure-Python wheel")
    pure_wheel.add_argument("project", choices=("pathfinder", "metapackage"))
    pure_wheel.set_defaults(handler=_pure_wheel)

    native_wheel = subparsers.add_parser("native-wheel", help="build one cibuildwheel wheel")
    native_wheel.add_argument("project", choices=("bindings", "core"))
    native_wheel.add_argument("--lane", choices=("current", "previous"), required=True)
    native_wheel.set_defaults(handler=_native_wheel)

    sdist = subparsers.add_parser("sdist", help="build an sdist and verify its wheel build")
    sdist.add_argument("project", choices=PACKAGE_PROJECTS)
    sdist.set_defaults(handler=_sdist)

    merge_wheels = subparsers.add_parser("merge-core-wheels", help="merge current and previous CUDA wheels")
    merge_wheels.set_defaults(handler=_merge_core_wheels)

    pixi_test = subparsers.add_parser("pixi-test", help="run an existing Pixi test or benchmark task")
    pixi_test.add_argument("project", choices=("pathfinder", "bindings", "core", "bindings-benchmarks"))
    pixi_test.add_argument("--task", default="test")
    pixi_test.add_argument("--environment")
    pixi_test.add_argument("extra", nargs="*")
    pixi_test.set_defaults(handler=_pixi_test)

    pixi_docs = subparsers.add_parser("pixi-docs", help="run an existing Pixi documentation task")
    pixi_docs.add_argument("project", choices=tuple(DOC_TASKS))
    pixi_docs.set_defaults(handler=_pixi_docs)

    docs_ci = subparsers.add_parser("docs-ci", help="build and stage the combined CI documentation")
    docs_ci.set_defaults(handler=_docs_ci)

    cython_assets = subparsers.add_parser("cython-test-assets", help="build and stage Cython tests")
    cython_assets.add_argument("project", choices=CYTHON_PROJECTS)
    cython_assets.set_defaults(handler=_cython_test_assets)

    core_binaries = subparsers.add_parser("core-test-binaries", help="build and stage cuda.core test binaries")
    core_binaries.set_defaults(handler=_core_test_binaries)

    installed_test = subparsers.add_parser("installed-test", help="run an installed-wheel package test suite")
    installed_test.add_argument("project", choices=("pathfinder", "bindings", "core"))
    installed_test.set_defaults(handler=_installed_test)

    metapackage_test = subparsers.add_parser(
        "metapackage-install-test", help="verify the local cuda-python wheel set is installable"
    )
    metapackage_test.set_defaults(handler=_metapackage_install_test)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "extra", None) and args.extra[0] == "--":
        args.extra = args.extra[1:]
    args.handler(args)


if __name__ == "__main__":
    main()
