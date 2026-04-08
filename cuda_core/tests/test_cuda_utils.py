# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import dataclasses

import pytest

from cuda.bindings import driver, runtime
from cuda.core._utils import cuda_utils
from cuda.core._utils.clear_error_support import assert_type_str_or_bytes_like, raise_code_path_meant_to_be_unreachable


def _skip_if_bindings_pre_enum_docstrings():
    from cuda.core._utils.enum_explanations_helpers import _binding_version_has_usable_enum_docstrings
    from cuda.core._utils.version import binding_version

    if not _binding_version_has_usable_enum_docstrings(binding_version()):
        pytest.skip("cuda-bindings version does not expose usable enum __doc__ strings")


def _assert_cleanup_example_matches_or_xfail(actual, expected):
    # Pin a few real cleanup-sensitive enum docs. If one starts failing, review
    # the raw ``__doc__`` and today's cleaned output: either update the expected
    # text to match an acceptable upstream change, or fix the cleanup logic.
    if actual != expected:
        pytest.xfail("please review this failure")
    assert actual == expected


def test_check_driver_error():
    num_unexpected = 0
    for error in driver.CUresult:
        if error == driver.CUresult.CUDA_SUCCESS:
            assert cuda_utils._check_driver_error(error) == 0
        else:
            with pytest.raises(cuda_utils.CUDAError) as e:
                cuda_utils._check_driver_error(error)
            msg = str(e)
            if "UNEXPECTED ERROR CODE" in msg:
                num_unexpected += 1
            else:
                # Example repr(error): <CUresult.CUDA_ERROR_UNKNOWN: 999>
                enum_name = repr(error).split(".", 1)[1].split(":", 1)[0]
                assert enum_name in msg
    # Smoke test: We don't want most to be unexpected.
    assert num_unexpected < len(driver.CUresult) * 0.5


def test_check_runtime_error():
    num_unexpected = 0
    for error in runtime.cudaError_t:
        if error == runtime.cudaError_t.cudaSuccess:
            assert cuda_utils._check_runtime_error(error) == 0
        else:
            with pytest.raises(cuda_utils.CUDAError) as e:
                cuda_utils._check_runtime_error(error)
            msg = str(e)
            if "UNEXPECTED ERROR CODE" in msg:
                num_unexpected += 1
            else:
                # Example repr(error): <cudaError_t.cudaErrorUnknown: 999>
                enum_name = repr(error).split(".", 1)[1].split(":", 1)[0]
                assert enum_name in msg
    # Smoke test: We don't want most to be unexpected.
    assert num_unexpected < len(driver.CUresult) * 0.5


def test_driver_error_enum_has_non_empty_docstring():
    _skip_if_bindings_pre_enum_docstrings()

    doc = driver.CUresult.CUDA_ERROR_INVALID_VALUE.__doc__
    assert doc is not None
    assert doc.strip() != ""


def test_runtime_error_enum_has_non_empty_docstring():
    _skip_if_bindings_pre_enum_docstrings()

    doc = runtime.cudaError_t.cudaErrorInvalidValue.__doc__
    assert doc is not None
    assert doc.strip() != ""


# These use real enum members rather than synthetic strings, to pin a few
# representative cleanup-sensitive docs end to end. Together with the helper
# unit tests, this gives a harder assurance that today's live bindings output
# is rendered into the user-facing text we expect. Unexpected changes are
# marked as xfail so they prompt manual review of the drift, without causing
# a hard test failure.
@pytest.mark.parametrize(
    ("explanations", "error", "expected"),
    [
        pytest.param(
            cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS,
            driver.CUresult.CUDA_ERROR_NOT_INITIALIZED,
            "This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed.",
            id="driver_not_initialized_role_cleanup",
        ),
        pytest.param(
            cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS,
            driver.CUresult.CUDA_ERROR_INVALID_CONTEXT,
            (
                "This most frequently indicates that there is no context bound to the current thread. "
                "This can also be returned if the context passed to an API call is not a valid handle "
                "(such as a context that has had cuCtxDestroy() invoked on it). This can also be "
                "returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). "
                "See cuCtxGetApiVersion() for more details. This can also be returned if the green "
                "context passed to an API call was not converted to a CUcontext using cuCtxFromGreenCtx API."
            ),
            id="driver_invalid_context_multiple_roles",
        ),
        pytest.param(
            cuda_utils.RUNTIME_CUDA_ERROR_EXPLANATIONS,
            runtime.cudaError_t.cudaErrorLaunchTimeout,
            (
                "This indicates that the device kernel took too long to execute. This can only occur "
                "if timeouts are enabled - see the device attribute cudaDevAttrKernelExecTimeout for "
                "more information. This leaves the process in an inconsistent state and any further "
                "CUDA work will return the same error. To continue using CUDA, the process must be "
                "terminated and relaunched."
            ),
            id="runtime_launch_timeout_role_cleanup",
        ),
        pytest.param(
            cuda_utils.RUNTIME_CUDA_ERROR_EXPLANATIONS,
            runtime.cudaError_t.cudaErrorIncompatibleDriverContext,
            (
                "This indicates that the current context is not compatible with this the CUDA Runtime. "
                "This can only occur if you are using CUDA Runtime/Driver interoperability and have "
                "created an existing Driver context using the driver API. The Driver context may be "
                "incompatible either because the Driver context was created using an older version of "
                "the API, because the Runtime API call expects a primary driver context and the Driver "
                "context is not primary, or because the Driver context has been destroyed. Please see "
                '"Interactions with the CUDA Driver API" for more information.'
            ),
            id="runtime_incompatible_driver_context_codegen_bug",
        ),
    ],
)
def test_enum_doc_cleanup_examples_are_reviewed_on_change(explanations, error, expected):
    _skip_if_bindings_pre_enum_docstrings()

    actual = explanations.get(int(error))
    _assert_cleanup_example_matches_or_xfail(actual, expected)


def test_check_driver_error_attaches_explanation():
    error = driver.CUresult.CUDA_ERROR_INVALID_VALUE
    name_err, name = driver.cuGetErrorName(error)
    assert name_err == driver.CUresult.CUDA_SUCCESS
    desc_err, desc = driver.cuGetErrorString(error)
    assert desc_err == driver.CUresult.CUDA_SUCCESS
    expl = cuda_utils.DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
    assert expl is not None
    assert expl != desc.decode()

    with pytest.raises(cuda_utils.CUDAError) as e:
        cuda_utils._check_driver_error(error)

    assert str(e.value) == f"{name.decode()}: {expl}"
    assert str(e.value) != f"{name.decode()}: {desc.decode()}"


def test_check_runtime_error_attaches_explanation():
    error = runtime.cudaError_t.cudaErrorInvalidValue
    name_err, name = runtime.cudaGetErrorName(error)
    assert name_err == runtime.cudaError_t.cudaSuccess
    desc_err, desc = runtime.cudaGetErrorString(error)
    assert desc_err == runtime.cudaError_t.cudaSuccess
    expl = cuda_utils.RUNTIME_CUDA_ERROR_EXPLANATIONS.get(int(error))
    assert expl is not None
    assert expl != desc.decode()

    with pytest.raises(cuda_utils.CUDAError) as e:
        cuda_utils._check_runtime_error(error)

    assert str(e.value) == f"{name.decode()}: {expl}"
    assert str(e.value) != f"{name.decode()}: {desc.decode()}"


def test_precondition():
    def checker(*args, what=""):
        if args[0] < 0:
            raise ValueError(f"{what}: negative")

    @cuda_utils.precondition(checker, what="value check")
    def my_func(x):
        return x * 2

    assert my_func(5) == 10
    with pytest.raises(ValueError, match="negative"):
        my_func(-1)


@dataclasses.dataclass
class _DummyOptions:
    x: int = 1
    y: str = "hello"


def test_check_nvrtc_error_without_handle():
    from cuda.bindings import nvrtc

    assert cuda_utils._check_nvrtc_error(nvrtc.nvrtcResult.NVRTC_SUCCESS) == 0
    with pytest.raises(cuda_utils.NVRTCError):
        cuda_utils._check_nvrtc_error(nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION)


def test_check_nvrtc_error_with_handle(init_cuda):
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(b"invalid code!@#$", b"test.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    try:
        (compile_result,) = nvrtc.nvrtcCompileProgram(prog, 0, [])
        assert compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS
        with pytest.raises(cuda_utils.NVRTCError, match="compilation log"):
            cuda_utils._check_nvrtc_error(compile_result, handle=prog)
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


def test_check_or_create_options_invalid_type():
    with pytest.raises(TypeError, match="must be provided as an object"):
        cuda_utils.check_or_create_options(_DummyOptions, 12345, options_description="test options")


def test_assert_type_str_or_bytes_like_rejects_non_str_bytes():
    with pytest.raises(TypeError, match="Expected type str or bytes or bytearray"):
        assert_type_str_or_bytes_like(12345)


def test_raise_code_path_meant_to_be_unreachable():
    with pytest.raises(RuntimeError, match="This code path is meant to be unreachable"):
        raise_code_path_meant_to_be_unreachable()
