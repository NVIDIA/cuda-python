# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.pathfinder._utils import toolkit_info


@pytest.fixture(autouse=True)
def _clear_cuda_header_version_cache():
    toolkit_info.read_cuda_header_version.cache_clear()
    yield
    toolkit_info.read_cuda_header_version.cache_clear()


def test_encoded_cuda_version_from_encoded_decodes_major_minor():
    assert toolkit_info.EncodedCudaVersion.from_encoded(13020) == toolkit_info.EncodedCudaVersion(
        encoded=13020,
        major=13,
        minor=2,
    )


def test_encoded_cuda_version_from_encoded_accepts_decimal_string():
    assert toolkit_info.EncodedCudaVersion.from_encoded("13020") == toolkit_info.EncodedCudaVersion(
        encoded=13020,
        major=13,
        minor=2,
    )


def test_encoded_cuda_version_from_encoded_raises_helpful_error_for_invalid_string():
    with pytest.raises(
        ValueError,
        match=r"EncodedCudaVersion\.from_encoded\(\) expected an integer or decimal string, got '13\.2'",
    ):
        toolkit_info.EncodedCudaVersion.from_encoded("13.2")


@pytest.mark.parametrize("encoded", [-1, "-1"])
def test_encoded_cuda_version_from_encoded_rejects_negative_values(encoded):
    with pytest.raises(
        ValueError,
        match=r"EncodedCudaVersion\.from_encoded\(\) expected a non-negative encoded CUDA version, got -1",
    ):
        toolkit_info.EncodedCudaVersion.from_encoded(encoded)


def test_parse_cuda_header_version_returns_parsed_dataclass():
    header_text = """
    #ifndef CUDA_H
    #define CUDA_H
    #define CUDA_VERSION 13020
    #endif
    """

    assert toolkit_info.parse_cuda_header_version(header_text) == toolkit_info.CudaToolkitVersion(
        encoded=13020,
        major=13,
        minor=2,
    )


def test_cuda_toolkit_version_from_encoded_returns_subclass_instance():
    version = toolkit_info.CudaToolkitVersion.from_encoded(12090)

    assert version == toolkit_info.CudaToolkitVersion(
        encoded=12090,
        major=12,
        minor=9,
    )
    assert type(version) is toolkit_info.CudaToolkitVersion


def test_parse_cuda_header_version_returns_none_when_macro_is_missing():
    header_text = """
    #ifndef CUDA_H
    #define CUDA_H
    #define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
    #endif
    """

    assert toolkit_info.parse_cuda_header_version(header_text) is None


def test_read_cuda_header_version_reads_file_and_returns_parsed_dataclass(tmp_path):
    cuda_h_path = tmp_path / "cuda.h"
    cuda_h_path.write_text(
        """
        #ifndef CUDA_H
        #define CUDA_H
        #define CUDA_VERSION 12090 /* CUDA 12.9 */
        #endif
        """,
        encoding="utf-8",
    )

    assert toolkit_info.read_cuda_header_version(str(cuda_h_path)) == toolkit_info.CudaToolkitVersion(
        encoded=12090,
        major=12,
        minor=9,
    )


def test_read_cuda_header_version_tolerates_non_utf8_bytes(tmp_path):
    cuda_h_path = tmp_path / "cuda.h"
    cuda_h_path.write_bytes(
        b"#ifndef CUDA_H\n"
        b"#define CUDA_H\n"
        b"\xff\xfe invalid bytes in comment or banner\n"
        b"#define CUDA_VERSION 12080\n"
        b"#endif\n"
    )

    assert toolkit_info.read_cuda_header_version(str(cuda_h_path)) == toolkit_info.CudaToolkitVersion(
        encoded=12080,
        major=12,
        minor=8,
    )


def test_read_cuda_header_version_wraps_parse_failures(tmp_path):
    cuda_h_path = tmp_path / "cuda.h"
    cuda_h_path.write_text(
        """
        #ifndef CUDA_H
        #define CUDA_H
        #endif
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        toolkit_info.ReadCudaHeaderVersionError,
        match="Failed to read the CUDA Toolkit version from cuda.h",
    ) as exc_info:
        toolkit_info.read_cuda_header_version(str(cuda_h_path))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "does not define CUDA_VERSION" in str(exc_info.value.__cause__)
