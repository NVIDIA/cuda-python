# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
from cuda_python_test_helpers.subprocess_runner import run_python_snippet
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

from cuda.pathfinder import find_nvidia_header_directory

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Native NVML shim uses the Linux loader")


_SHIM_SOURCE = r"""
#include <nvml.h>
#include <stdint.h>

static unsigned int query_count;

unsigned int test_get_query_count(void) {
    return query_count;
}

const char *nvmlErrorString(nvmlReturn_t status) {
    (void)status;
    return "native NVML test error";
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int count,
                                    nvmlFieldValue_t *values) {
    ++query_count;
    if ((uintptr_t)device == 2) {
        return NVML_ERROR_NO_PERMISSION;
    }
    for (int i = 0; i < count; ++i) {
        switch (values[i].fieldId) {
        case 1:  /* Successful native no-op. */
            break;
        case 2:
            values[i].timestamp = 12345;
            values[i].latencyUsec = 7;
            values[i].valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
            values[i].value.uiVal = 17;
            values[i].nvmlReturn = NVML_SUCCESS;
            break;
        case 3:
            values[i].nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
            break;
        case 4:
            values[i].nvmlReturn = NVML_ERROR_UNKNOWN;
            break;
        case 5:
            values[i].valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
            values[i].value.uiVal = values[i].scopeId;
            values[i].nvmlReturn = NVML_SUCCESS;
            break;
        }
    }
    return NVML_SUCCESS;
}
"""

_SHIM_SETUP = """
import ctypes
import os

shim = ctypes.CDLL(os.environ["CUDA_PYTHON_TEST_NVML_SHIM"], mode=ctypes.RTLD_GLOBAL)
from cuda.bindings import nvml
"""


@pytest.fixture(scope="module")
def ctx():
    """Override the parent CUDA-context fixture for these native-shim tests."""


@pytest.fixture(scope="module")
def field_values_shim(tmp_path_factory):
    header_directory = find_nvidia_header_directory("cudart")
    if header_directory is None or not (Path(header_directory) / "nvml.h").is_file():
        pytest.skip("NVML header is required to compile the native test shim")

    compiler = new_compiler()
    customize_compiler(compiler)
    if shutil.which(compiler.compiler[0]) is None:
        pytest.skip("C compiler is required to compile the native test shim")
    directory = tmp_path_factory.mktemp("nvml-field-values")
    source = directory / "field_values.c"
    source.write_text(_SHIM_SOURCE, encoding="utf-8")
    objects = compiler.compile([str(source)], output_dir=str(directory), include_dirs=[header_directory])
    library = directory / "libnvidia-ml.so.1"
    compiler.link_shared_object(objects, str(library))
    return library


def _run_with_shim(library, code):
    # A fresh process keeps the native symbol cache and shim out of other tests.
    run_python_snippet(
        _SHIM_SETUP + textwrap.dedent(code),
        timeout=30,
        extra_env={
            "CUDA_PYTHON_TEST_NVML_SHIM": str(library),
            "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (str(library.parent), os.environ.get("LD_LIBRARY_PATH")))),
        },
    )


@pytest.mark.parametrize("initial_status", [0, 3, 5, 12345])
@pytest.mark.agent_authored(model="gpt-6")
def test_unpopulated_field_values_report_unknown(field_values_shim, initial_status):
    _run_with_shim(
        field_values_shim,
        f"""
        fields = nvml.FieldValue(2)
        fields.field_id = [1, 1]
        fields.scope_id = [0, 11]
        fields.nvml_return = {initial_status}
        for _ in range(2):
            result = nvml.device_get_field_values(1, fields)
            assert list(result.nvml_return) == [nvml.Return.ERROR_UNKNOWN] * 2
            assert list(result.field_id) == [1, 1]
            assert list(result.scope_id) == [0, 11]
            fields.nvml_return = nvml.Return.SUCCESS
        """,
    )


@pytest.mark.agent_authored(model="gpt-6")
def test_field_values_preserve_native_results_and_scopes(field_values_shim):
    _run_with_shim(
        field_values_shim,
        """
        import numpy as np

        fields = nvml.FieldValue(4)
        fields.field_id = [2, 3, 1, 4]
        fields.scope_id = [7, 8, 9, 10]
        fields.nvml_return = nvml.Return.SUCCESS
        result = nvml.device_get_field_values(1, fields)
        assert list(result.nvml_return) == [
            nvml.Return.SUCCESS, nvml.Return.ERROR_NOT_SUPPORTED,
            nvml.Return.ERROR_UNKNOWN, nvml.Return.ERROR_UNKNOWN,
        ]
        assert result[0].timestamp == 12345
        assert result[0].latency_usec == 7
        assert result[0].value_type == nvml.ValueType.UNSIGNED_INT
        assert result[0].value.ui_val[0] == 17
        assert list(result.field_id) == [2, 3, 1, 4]
        assert list(result.scope_id) == [7, 8, 9, 10]

        original_empty = np.empty
        def poisoned_empty(*args, **kwargs):
            array = original_empty(*args, **kwargs)
            array.view(np.uint8).fill(0xA5)
            return array

        np.empty = poisoned_empty
        try:
            result = nvml.device_get_field_values(1, [5, 5])
        finally:
            np.empty = original_empty
        assert list(result.scope_id) == [0, 0]
        assert list(result.nvml_return) == [nvml.Return.SUCCESS] * 2
        assert list(result.value.ui_val) == [0, 0]

        result = nvml.device_get_field_values(1, [(5, 19), (5, 43)])
        assert list(result.scope_id) == [19, 43]
        assert list(result.nvml_return) == [nvml.Return.SUCCESS] * 2
        assert list(result.value.ui_val) == [19, 43]

        for values in ([1], [(1, 11)]):
            result = nvml.device_get_field_values(1, values)
            assert result[0].nvml_return == nvml.Return.ERROR_UNKNOWN
        """,
    )


@pytest.mark.agent_authored(model="gpt-6")
def test_field_values_preserve_aggregate_error(field_values_shim):
    _run_with_shim(
        field_values_shim,
        """
        import pytest

        with pytest.raises(nvml.NoPermissionError) as caught:
            nvml.device_get_field_values(2, [1])
        assert caught.value.status == nvml.Return.ERROR_NO_PERMISSION
        """,
    )


@pytest.mark.agent_authored(model="gpt-6")
def test_empty_field_values_do_not_call_native_query(field_values_shim):
    _run_with_shim(
        field_values_shim,
        """
        assert len(nvml.device_get_field_values(2, [])) == 0
        assert len(nvml.device_get_field_values(2, nvml.FieldValue(0))) == 0
        assert shim.test_get_query_count() == 0
        """,
    )
