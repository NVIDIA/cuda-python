# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fetch_ctk_redistrib import (
    filter_components,
    get_component_relative_path,
    host_platform_to_subdir,
    main,
    validate_metadata_url,
)


def _sample_metadata() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "cuda_nvcc": {
            "linux-x86_64": {"relative_path": "linux-x86_64/cuda_nvcc.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/cuda_nvcc.tar.xz"},
            "windows-x86_64": {"relative_path": "windows-x86_64/cuda_nvcc.zip"},
        },
        "libcufile": {
            "linux-x86_64": {"relative_path": "linux-x86_64/libcufile.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/libcufile.tar.xz"},
        },
        "libnvjitlink": {
            "linux-x86_64": {"relative_path": "linux-x86_64/libnvjitlink.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/libnvjitlink.tar.xz"},
            "windows-x86_64": {"relative_path": "windows-x86_64/libnvjitlink.zip"},
        },
        "cuda_crt": {
            "linux-x86_64": {"relative_path": "linux-x86_64/cuda_crt.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/cuda_crt.tar.xz"},
            "windows-x86_64": {"relative_path": "windows-x86_64/cuda_crt.zip"},
        },
        "libnvvm": {
            "linux-x86_64": {"relative_path": "linux-x86_64/libnvvm.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/libnvvm.tar.xz"},
            "windows-x86_64": {"relative_path": "windows-x86_64/libnvvm.zip"},
        },
        "libnvfatbin": {
            "linux-x86_64": {"relative_path": "linux-x86_64/libnvfatbin.tar.xz"},
            "linux-sbsa": {"relative_path": "linux-sbsa/libnvfatbin.tar.xz"},
            "windows-x86_64": {"relative_path": "windows-x86_64/libnvfatbin.zip"},
        },
        "libcudla": {
            "linux-sbsa": {"relative_path": "linux-sbsa/libcudla.tar.xz"},
            "linux-aarch64": {"relative_path": "linux-aarch64/libcudla.tar.xz"},
        },
    }


def _write_metadata(tmp_path) -> str:
    path = tmp_path / "redistrib.json"
    path.write_text(json.dumps(_sample_metadata()), encoding="utf-8")
    return str(path)


class TestHostPlatformToSubdir:
    def test_linux_aarch64_maps_to_linux_sbsa(self):
        assert host_platform_to_subdir("linux-aarch64") == "linux-sbsa"

    def test_unknown_platform_raises(self):
        with pytest.raises(ValueError, match="unsupported host-platform"):
            host_platform_to_subdir("darwin-64")


class TestValidateMetadataUrl:
    def test_accepts_https_url(self):
        url = "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.1.json"
        assert validate_metadata_url(url) == url

    def test_rejects_file_url(self):
        with pytest.raises(ValueError, match="https URL"):
            validate_metadata_url("file:///tmp/redistrib.json")


class TestFilterComponents:
    def test_applies_static_version_and_platform_filters(self):
        filtered, skipped = filter_components(
            _sample_metadata(),
            host_platform="win-64",
            cuda_version="11.8.0",
            components="cuda_nvcc,libcufile,libnvjitlink,cuda_crt,libnvvm",
        )

        assert filtered == ["cuda_nvcc"]
        assert skipped == []

    def test_drops_components_missing_from_selected_subdir(self):
        filtered, skipped = filter_components(
            _sample_metadata(),
            host_platform="linux-64",
            cuda_version="13.2.1",
            components="cuda_nvcc,libcudla,libnvfatbin",
        )

        assert filtered == ["cuda_nvcc", "libnvfatbin"]
        assert skipped == ["libcudla"]


class TestGetComponentRelativePath:
    def test_returns_relative_path_for_selected_platform(self):
        assert (
            get_component_relative_path(
                _sample_metadata(),
                host_platform="win-64",
                component="libnvfatbin",
            )
            == "windows-x86_64/libnvfatbin.zip"
        )

    def test_raises_when_component_is_missing_for_subdir(self):
        with pytest.raises(KeyError, match="not available"):
            get_component_relative_path(
                _sample_metadata(),
                host_platform="linux-64",
                component="libcudla",
            )


class TestMain:
    def test_filter_components_cli_uses_local_metadata(self, tmp_path, capsys):
        metadata_path = _write_metadata(tmp_path)

        rc = main(
            [
                "filter-components",
                "--host-platform",
                "linux-64",
                "--cuda-version",
                "13.2.1",
                "--components",
                "cuda_nvcc,libcudla,libnvfatbin",
                "--metadata-path",
                metadata_path,
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == "cuda_nvcc,libnvfatbin\n"
        assert "Skipping unsupported CTK component 'libcudla'" in captured.err

    def test_component_relative_path_cli_uses_local_metadata(self, tmp_path, capsys):
        metadata_path = _write_metadata(tmp_path)

        rc = main(
            [
                "component-relative-path",
                "--host-platform",
                "linux-aarch64",
                "--component",
                "cuda_nvcc",
                "--metadata-path",
                metadata_path,
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == "linux-sbsa/cuda_nvcc.tar.xz\n"
        assert captured.err == ""
