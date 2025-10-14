#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path
from urllib.request import urlopen

# Example URL of an HTML directory listing
CONTENT_URL = "https://developer.download.nvidia.com/compute/cuda/redist"


CYBIND_GENERATED_LIBRARIES = [
    ("cufile", "libcufile", "cufile"),
    ("nvvm", "libnvvm", "nvvm"),
    ("nvjitlink", "libnvjitlink", "nvJitLink"),
]


def fetch_headers(version: str, library_name: str, dest_dir: Path):
    def tar_filter(members):
        for tarinfo in members:
            name = Path(tarinfo.name)
            parts = name.parts
            try:
                idx = parts.index("include")
            except ValueError:
                continue
            tarinfo.name = str(Path(*parts[idx + 1 :]))
            yield tarinfo

    output_dir = dest_dir / Path(version)
    if output_dir.exists():
        print(f"Skipping header download for {library_name} {version}, already exists")
        return

    output_dir.mkdir()

    json_url = f"{CONTENT_URL}/redistrib_{version}.json"
    with urlopen(json_url) as resp:  # noqa: S310
        content = json.loads(resp.read().decode("utf-8"))
    if library := content.get(library_name):
        archive_url = f"{CONTENT_URL}/{library['linux-x86_64']['relative_path']}"
        print(f"Fetching package {archive_url}")
        with tempfile.NamedTemporaryFile() as tmp:
            tmppath = Path(tmp.name)

            with tmppath.open("wb") as f, urlopen(archive_url) as resp:  # noqa: S310
                f.write(resp.read())

            with tarfile.open(tmppath, "r:xz") as tar:
                tar.extractall(  # noqa: S202
                    members=tar_filter(tar.getmembers()),
                    path=output_dir,
                    filter="fully_trusted",
                )
    else:
        print(f"No {library_name} in version {version}")


def update_config(version: str, config_path: Path) -> None:
    # This is pretty brittle, but will be better when/if we move all the config to YAML

    out = []
    in_version_section = False
    with config_path.open() as f:
        for line in f:
            if line.strip() == "'versions': [":
                in_version_section = True
            if in_version_section and line.strip() == "],":
                out.append(f"                ('{version}', ),\n")
                in_version_section = False
            out.append(line)

    with config_path.open("w") as f:
        f.write("".join(out))


def run_cybind(cybind_repo: Path, cuda_python_repo: Path, libraries: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)

        venv.create(tempdir_path, with_pip=True)
        subprocess.check_call(  # noqa: S603
            [
                str(tempdir_path / "bin" / "python"),
                "-m",
                "pip",
                "install",
                str(cybind_repo),
            ]
        )
        try:
            subprocess.check_call(  # noqa: S603
                [
                    str(tempdir_path / "bin" / "python"),
                    "-m",
                    "cybind",
                    "--generate",
                    *libraries,
                    "--output-dir",
                    str(cuda_python_repo / "cuda_bindings"),
                ]
            )
        except subprocess.CalledProcessError:
            print("Error running cybind.")
            print("This probably indicates an issue introduced with the new headers.")
            print("If necessary, you can edit the headers and re-run this script.")
            return 1


def update_version_file(version: str, version_path: Path, is_prev: bool) -> str:
    if is_prev:
        key = "prev_build"
    else:
        key = "build"

    with version_path.open() as f:
        content = json.load(f)
    existing_version = content["cuda"][key]["version"]
    content["cuda"][key]["version"] = version

    with version_path.open("w") as f:
        content = json.dump(content, f, indent=2)
        # json.dump doesn't add a trailing newline
        f.write("\n")

    return existing_version


def update_matrix(existing_version: str, new_version: str, matrix_path: Path) -> None:
    # It would be less brittle to update using JSON here, but that messes up the formatting

    with matrix_path.open() as f:
        content = f.read()

    content = re.sub(rf'"CUDA_VER": "{existing_version}"', f'"CUDA_VER": "{new_version}"', content)

    with matrix_path.open("w") as f:
        f.write(content)


def main(version: str, cuda_python_repo: Path, cybind_repo: Path, is_prev: bool):
    cybind_headers_path = cybind_repo / "assets" / "headers"
    cybind_config_path = cybind_repo / "assets" / "configs"

    for libname, distname, subdir in CYBIND_GENERATED_LIBRARIES:
        fetch_headers(version, distname, cybind_headers_path / subdir)
        update_config(version, cybind_config_path / f"config_{libname}.py")

    existing_version = update_version_file(version, cuda_python_repo / "ci" / "versions.json", is_prev)
    update_matrix(existing_version, version, cuda_python_repo / "ci" / "test-matrix.json")

    # Do this last, because, if anything, it's the thing that's likely to fail
    if run_cybind(cybind_repo, cuda_python_repo, [x[0] for x in CYBIND_GENERATED_LIBRARIES]):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update cuda-python for a new version of the CTK")
    parser.add_argument(
        "--cybind-repo",
        type=Path,
        help="Path to a checkout of cybind (default: ../cybind relative to cuda-python)",
    )
    parser.add_argument(
        "--is-prev",
        action="store_true",
        help="When given, update the previous, not latest version",
    )
    parser.add_argument(
        "version",
        type=str,
        help="Version to move to",
    )
    args = parser.parse_args()

    cuda_python_repo = Path(__file__).parents[1]

    if args.cybind_repo is None:
        args.cybind_repo = cuda_python_repo.parent / "cybind"

    print("Before running this script, you need to:")
    print("  - Create a new branch in this repo based on upstream/main")
    print(f"  - Create a new branch in a cybind checkout at {args.cybind_repo} based on upstream/main")
    print()
    print(f"This will add CTK {args.version} as the {'previous' if args.is_prev else 'latest'} version.")
    print("Proceed? [y/N]")
    resp = input().strip().lower()
    if resp != "y":
        print("Aborting")

    main(args.version, cuda_python_repo, args.cybind_repo, args.is_prev)

    print("Remaining manual steps:")
    print("- Add a changelog entry:")
    print(
        f"* Updated the ``cuda.bindings.runtime`` module to statically link "
        f"against the CUDA Runtime library from CUDA Toolkit {args.version}."
    )
    print("- Inspect the changes to this repo and cybind, commit and submit PRs.")
