# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings

import numpy as np
import pytest

from cuda.core import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Linker,
    LinkerOptions,
    Program,
    ProgramOptions,
    _linker,
    launch,
)
from cuda.core._module import ObjectCode
from cuda.core._program import _can_load_generated_ptx
from cuda.core._utils.cuda_utils import CUDAError

ARCH = "sm_" + "".join(f"{i}" for i in Device().compute_capability)

kernel_a = """
extern __device__ int B();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(B(), 1);}
"""
device_function_b = "__device__ int B() { return 0; }"
device_function_c = "__device__ int C(int a, int b) { return a + b; }"

is_culink_backend = _linker._decide_nvjitlink_or_driver()
if not is_culink_backend:
    from cuda.bindings import nvjitlink

    nvJitLinkError = nvjitlink.nvJitLinkError
    nvjitlink_version = nvjitlink.version()
    has_linked_ltoir_bindings = all(hasattr(nvjitlink, name) for name in ("get_linked_ltoir_size", "get_linked_ltoir"))
else:
    nvjitlink_version = (0, 0)
    has_linked_ltoir_bindings = False

    class nvJitLinkError(Exception):
        pass


@pytest.fixture
def compile_ptx_functions(init_cuda):
    # Without -rdc (relocatable device code) option, the generated ptx will not included any unreferenced
    # device functions, causing the link to fail
    object_code_a_ptx = Program(kernel_a, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    object_code_b_ptx = Program(device_function_b, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    object_code_c_ptx = Program(device_function_c, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")

    return object_code_a_ptx, object_code_b_ptx, object_code_c_ptx


@pytest.fixture
def compile_ltoir_functions(init_cuda):
    object_code_a_ltoir = Program(kernel_a, "c++", ProgramOptions(link_time_optimization=True)).compile("ltoir")
    object_code_b_ltoir = Program(device_function_b, "c++", ProgramOptions(link_time_optimization=True)).compile(
        "ltoir"
    )
    object_code_c_ltoir = Program(device_function_c, "c++", ProgramOptions(link_time_optimization=True)).compile(
        "ltoir"
    )

    return object_code_a_ltoir, object_code_b_ltoir, object_code_c_ltoir


options = [
    LinkerOptions(),
    LinkerOptions(arch=ARCH, verbose=True),
    LinkerOptions(arch=ARCH, max_register_count=32),
    LinkerOptions(arch=ARCH, optimization_level=3),
    LinkerOptions(arch=ARCH, debug=True),
    LinkerOptions(arch=ARCH, lineinfo=True),
]
if not is_culink_backend:
    options += [
        LinkerOptions(arch=ARCH, time=True),
        LinkerOptions(arch=ARCH, optimize_unused_variables=True),
        LinkerOptions(arch=ARCH, ptxas_options="-v"),
        LinkerOptions(arch=ARCH, ptxas_options=["-v", "--verbose"]),
        LinkerOptions(arch=ARCH, ptxas_options=("-v", "--verbose")),
        LinkerOptions(arch=ARCH, split_compile=0),
        LinkerOptions(arch=ARCH, split_compile_extended=1),
        # The following options are supported by nvjitlink and deprecated by culink
        LinkerOptions(arch=ARCH, ftz=True),
        LinkerOptions(arch=ARCH, prec_div=True),
        LinkerOptions(arch=ARCH, prec_sqrt=True),
        LinkerOptions(arch=ARCH, fma=True),
        LinkerOptions(arch=ARCH, kernels_used="A"),
        LinkerOptions(arch=ARCH, kernels_used=["C", "B"]),
        LinkerOptions(arch=ARCH, kernels_used=("C", "B")),
        LinkerOptions(arch=ARCH, variables_used="var1"),
        LinkerOptions(arch=ARCH, variables_used=["var1", "var2"]),
        LinkerOptions(arch=ARCH, variables_used=("var1", "var2")),
    ]
    if nvjitlink_version >= (12, 5):
        options.append(LinkerOptions(arch=ARCH, no_cache=True))


@pytest.mark.parametrize("options", options)
def test_linker_init(compile_ptx_functions, options):
    linker = Linker(*compile_ptx_functions, options=options)
    object_code = linker.link("cubin")
    assert isinstance(object_code, ObjectCode)
    assert Linker.which_backend() == ("driver" if is_culink_backend else "nvJitLink")


def test_linker_init_invalid_arch(compile_ptx_functions):
    err = AttributeError if is_culink_backend else nvjitlink.nvJitLinkError
    with pytest.raises(err):
        options = LinkerOptions(arch="99", ptx=True)
        Linker(*compile_ptx_functions, options=options)


@pytest.mark.skipif(is_culink_backend, reason="culink does not support ptx option")
def test_linker_link_ptx_nvjitlink(compile_ltoir_functions):
    options = LinkerOptions(arch=ARCH, link_time_optimization=True, ptx=True)
    linker = Linker(*compile_ltoir_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


@pytest.mark.skipif(not is_culink_backend, reason="nvjitlink requires lto for ptx linking")
def test_linker_link_ptx_culink(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("ptx")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_cubin(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linked_code = linker.link("cubin")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_ptx_multiple(compile_ptx_functions):
    ptxes = tuple(ObjectCode.from_ptx(obj.code) for obj in compile_ptx_functions)
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*ptxes, options=options)
    linked_code = linker.link("cubin")
    assert isinstance(linked_code, ObjectCode)
    assert linked_code.name == options.name


def test_linker_link_invalid_target_type(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    with pytest.raises(ValueError):
        linker.link("invalid_target")


def test_linker_get_error_log(compile_ptx_functions):
    options = LinkerOptions(name="ABC", arch=ARCH)

    replacement_kernel = """
extern __device__ int Z();
extern __device__ int C(int a, int b);
__global__ void A() { int result = C(Z(), 1);}
"""
    dummy_program = Program(
        replacement_kernel, "c++", ProgramOptions(name="CBA", relocatable_device_code=True)
    ).compile("ptx")
    linker = Linker(dummy_program, *(compile_ptx_functions[1:]), options=options)
    try:
        linker.link("cubin")

    except (nvJitLinkError, CUDAError):
        log = linker.get_error_log()
        assert isinstance(log, str)
        # TODO when 4902246 is addressed, we can update this to cover nvjitlink as well
        # The error is coming from the input object that's being linked (CBA), not the output object (ABC).
        if is_culink_backend:
            assert log.rstrip("\x00") == "error   : Undefined reference to '_Z1Zv' in 'CBA'"


def test_linker_get_info_log(compile_ptx_functions):
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linker.link("cubin")
    log = linker.get_info_log()
    assert isinstance(log, str)


@pytest.mark.skipif(is_culink_backend, reason="as_bytes() only supported for nvjitlink backend")
def test_linker_options_as_bytes_nvjitlink():
    """Test LinkerOptions.as_bytes() for nvJitLink backend"""
    options = LinkerOptions(arch="sm_80", debug=True, ftz=True, max_register_count=32)
    nvjitlink_options = options.as_bytes("nvjitlink")

    # Should return list of bytes
    assert isinstance(nvjitlink_options, list)
    assert all(isinstance(opt, bytes) for opt in nvjitlink_options)

    # Decode to check content
    options_str = [opt.decode() for opt in nvjitlink_options]
    assert "-arch=sm_80" in options_str
    assert "-g" in options_str
    assert "-ftz=true" in options_str
    assert "-maxrregcount=32" in options_str


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(is_culink_backend, reason="as_bytes() only supported for nvjitlink backend")
@pytest.mark.parametrize("value,expected_count", [(None, 0), (False, 0), (True, 1)])
def test_linker_options_incremental_as_bytes(value, expected_count):
    options = LinkerOptions(arch="sm_80", incremental=value)
    assert options.as_bytes().count(b"-r") == expected_count


@pytest.mark.parametrize("backend", ("invalid", "driver"))
def test_linker_options_as_bytes_invalid_backend(backend):
    """Test LinkerOptions.as_bytes() with invalid backend"""
    options = LinkerOptions(arch="sm_80")
    with pytest.raises(ValueError, match="only supports 'nvjitlink' backend"):
        options.as_bytes(backend)


def test_linker_logs_cached_after_link(compile_ptx_functions):
    """After a successful link(), get_error_log/get_info_log should return cached strings."""
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    linker.link("cubin")
    err_log = linker.get_error_log()
    info_log = linker.get_info_log()
    assert isinstance(err_log, str)
    assert isinstance(info_log, str)
    # Calling again should return the same observable values.
    assert linker.get_error_log() == err_log
    assert linker.get_info_log() == info_log


@pytest.mark.agent_authored(model="gpt-5.6")
def test_closed_linker_rejects_link_but_preserves_cached_logs(compile_ptx_functions):
    linker = Linker(*compile_ptx_functions, options=LinkerOptions(arch=ARCH))
    linker.link("cubin")
    error_log = linker.get_error_log()
    info_log = linker.get_info_log()
    linker.close()

    assert linker.is_closed
    assert bool(linker) is True  # Preserve backward-compatible truthiness after close.
    assert linker.get_error_log() == error_log
    assert linker.get_info_log() == info_log
    with pytest.raises(RuntimeError, match="Linker has been closed"):
        linker.link("cubin")


def test_linker_handle(compile_ptx_functions):
    """Linker.handle returns a non-null handle object."""
    options = LinkerOptions(arch=ARCH)
    linker = Linker(*compile_ptx_functions, options=options)
    handle = linker.handle
    assert handle is not None
    assert int(handle) != 0


@pytest.mark.agent_authored(model="gpt-5")
@pytest.mark.skipif(not is_culink_backend, reason="driver backend regression test")
def test_driver_linker_lifetime_no_heap_corruption(compile_ptx_functions):
    if not _can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")

    linker = Linker(*compile_ptx_functions, options=LinkerOptions(arch=ARCH))
    linker.link("cubin")
    linker.close()
    del linker

    obj_a = Program(kernel_a, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    obj_b = Program(device_function_b, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    obj_c = Program(device_function_c, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    linker = Linker(obj_a, obj_b, obj_c, options=LinkerOptions(arch=ARCH))
    linker.link("cubin")
    linker.close()


@pytest.mark.agent_authored(model="gpt-5")
@pytest.mark.skipif(not is_culink_backend, reason="driver backend regression test")
def test_driver_linker_preserves_error_log_after_close(init_cuda):
    if not _can_load_generated_ptx():
        pytest.skip("PTX version too new for current driver")

    bad_kernel = """
extern __device__ int Z();
__global__ void A() { int r = Z(); }
"""
    bad_obj = Program(bad_kernel, "c++", ProgramOptions(relocatable_device_code=True)).compile("ptx")
    linker = Linker(bad_obj, options=LinkerOptions(arch=ARCH))
    with pytest.raises(CUDAError):
        linker.link("cubin")

    error_log = linker.get_error_log()
    assert error_log
    linker.close()
    assert linker.get_error_log() == error_log
    assert isinstance(linker.get_info_log(), str)


@pytest.mark.skipif(is_culink_backend, reason="nvjitlink options only tested with nvjitlink backend")
def test_linker_options_nvjitlink_options_as_str():
    """_prepare_nvjitlink_options(as_bytes=False) returns plain strings."""
    opts = LinkerOptions(arch=ARCH, debug=True, lineinfo=True)
    options = opts._prepare_nvjitlink_options(as_bytes=False)
    assert isinstance(options, list)
    assert all(isinstance(o, str) for o in options)
    assert f"-arch={ARCH}" in options
    assert "-g" in options
    assert "-lineinfo" in options


class TestWhichBackendClassmethod:
    def test_which_backend_returns_nvjitlink(self, monkeypatch):
        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", True)
        assert Linker.which_backend() == "nvJitLink"

    def test_which_backend_returns_driver(self, monkeypatch):
        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", False)
        assert Linker.which_backend() == "driver"

    def test_which_backend_invokes_probe_when_not_memoised(self, monkeypatch):
        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", None)
        called = []

        def fake_decide():
            called.append(True)
            return False  # False = not falling back to driver = nvJitLink

        monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", fake_decide)
        result = Linker.which_backend()
        assert result == "nvJitLink"
        assert called, "_decide_nvjitlink_or_driver was not called"

    @pytest.mark.agent_authored(model="gpt-5.6")
    def test_which_backend_caches_nvjitlink_module_and_version(self, monkeypatch):
        class NvJitLink:
            version_calls = 0

            @classmethod
            def version(cls):
                cls.version_calls += 1
                return (13, 4)

        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", None)
        monkeypatch.setattr(_linker, "_nvjitlink", None)
        monkeypatch.setattr(_linker, "_nvjitlink_version", None)
        monkeypatch.setattr(_linker, "_optional_cuda_import", lambda _name: NvJitLink)
        monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", lambda _nvjitlink: True)

        assert Linker.which_backend() == "nvJitLink"
        assert _linker._nvjitlink is NvJitLink
        assert _linker._nvjitlink_version == (13, 4)
        assert NvJitLink.version_calls == 1

        assert Linker.which_backend() == "nvJitLink"
        assert NvJitLink.version_calls == 1

    @pytest.mark.agent_authored(model="grok-4.5")
    def test_which_backend_falls_back_when_nvjitlink_too_old(self, monkeypatch):
        """Regression test for #2408: old nvJitLink must not crash which_backend()."""
        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", None)
        monkeypatch.setattr(_linker, "_driver", None)
        monkeypatch.setattr(_linker, "_nvjitlink", None)
        monkeypatch.setattr(_linker, "_nvjitlink_version", None)

        def fake__optional_cuda_import(modname, probe_function=None):
            assert modname == "cuda.bindings.nvjitlink"
            assert probe_function is None
            return object()

        monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)
        monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", lambda _nvjitlink: False)

        with pytest.warns(RuntimeWarning, match="too old \\(<12.3\\)"):
            assert Linker.which_backend() == "driver"

        assert _linker._use_nvjitlink_backend is False

    @pytest.mark.agent_authored(model="grok-4.5")
    def test_which_backend_falls_back_when_dylib_missing(self, monkeypatch):
        """Missing nvJitLink dylib must fall back without raising."""
        from cuda.pathfinder import DynamicLibNotFoundError

        monkeypatch.setattr(_linker, "_use_nvjitlink_backend", None)
        monkeypatch.setattr(_linker, "_driver", None)
        monkeypatch.setattr(_linker, "_nvjitlink", None)
        monkeypatch.setattr(_linker, "_nvjitlink_version", None)

        def raise_missing(_nvjitlink):
            raise DynamicLibNotFoundError("missing")

        def fake__optional_cuda_import(modname, probe_function=None):
            assert modname == "cuda.bindings.nvjitlink"
            assert probe_function is None
            return object()

        monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", raise_missing)
        monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)

        with pytest.warns(RuntimeWarning, match="cuda.bindings.nvjitlink is not available"):
            assert Linker.which_backend() == "driver"

        assert _linker._use_nvjitlink_backend is False

    def test_which_backend_is_classmethod(self):
        attr = inspect.getattr_static(Linker, "which_backend")
        assert isinstance(attr, classmethod)

    def test_which_backend_is_not_property(self):
        """which_backend is a classmethod, not a property.

        This is an intentional breaking change from the prior ``backend`` property API.
        All call sites must use parens: ``Linker.which_backend()``.
        """
        attr = inspect.getattr_static(Linker, "which_backend")
        assert not isinstance(attr, property)


@pytest.fixture
def driver_binding(monkeypatch):
    """Pin _linker._driver to the real driver module so driver-backend tests run under any backend."""
    from cuda.bindings import driver

    monkeypatch.setattr(_linker, "_driver", driver)
    return driver


def test_prepare_driver_options_all_supported(driver_binding):
    """Exercise every supported branch of _prepare_driver_options."""
    driver = driver_binding
    opts = LinkerOptions(
        arch="sm_80",
        max_register_count=32,
        verbose=True,
        link_time_optimization=True,
        optimization_level=2,
        debug=True,
        lineinfo=True,
        no_cache=True,
    )
    formatted, keys = opts._prepare_driver_options()
    assert len(formatted) == len(keys)
    assert len(keys) == 4 + 8  # 4 fixed log-buffer entries + 8 options set above

    # Skip log-buffer entries; verify key-to-value mapping (catches swap/dup/wrong-value).
    payload_keys = keys[4:]
    assert len(set(payload_keys)) == len(payload_keys), f"duplicate option keys: {payload_keys}"
    option_to_value = dict(zip(payload_keys, formatted[4:]))
    assert option_to_value[driver.CUjit_option.CU_JIT_TARGET] == driver.CUjit_target.CU_TARGET_COMPUTE_80
    assert option_to_value[driver.CUjit_option.CU_JIT_MAX_REGISTERS] == 32
    assert option_to_value[driver.CUjit_option.CU_JIT_LOG_VERBOSE] == 1
    assert option_to_value[driver.CUjit_option.CU_JIT_LTO] == 1
    assert option_to_value[driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL] == 2
    assert option_to_value[driver.CUjit_option.CU_JIT_GENERATE_DEBUG_INFO] == 1
    assert option_to_value[driver.CUjit_option.CU_JIT_GENERATE_LINE_INFO] == 1
    assert option_to_value[driver.CUjit_option.CU_JIT_CACHE_MODE] == driver.CUjit_cacheMode.CU_JIT_CACHE_OPTION_NONE


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"ftz": True}, "ftz option is deprecated"),
        ({"prec_div": True}, "prec_div option is deprecated"),
        ({"prec_sqrt": True}, "prec_sqrt option is deprecated"),
        ({"fma": True}, "fma options is deprecated"),
        ({"kernels_used": "my_kernel"}, "kernels_used is deprecated"),
        ({"variables_used": "my_var"}, "variables_used is deprecated"),
        ({"optimize_unused_variables": True}, "optimize_unused_variables is deprecated"),
    ],
)
def test_prepare_driver_options_deprecated_warnings(driver_binding, kwargs, match):
    """Each driver-deprecated option emits a DeprecationWarning."""
    opts = LinkerOptions(**kwargs)
    with pytest.warns(DeprecationWarning, match=match):
        opts._prepare_driver_options()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"time": True}, "time option is not supported by the driver API"),
        ({"ptx": True}, "ptx option is not supported by the driver API"),
        ({"ptxas_options": ["-v"]}, "ptxas_options option is not supported by the driver API"),
        ({"split_compile": 0}, "split_compile option is not supported by the driver API"),
        ({"split_compile_extended": 1}, "split_compile_extended option is not supported by the driver API"),
    ],
)
def test_prepare_driver_options_unsupported_raises(driver_binding, kwargs, match):
    """Each nvjitlink-only option raises ValueError on the driver backend."""
    opts = LinkerOptions(**kwargs)
    with pytest.raises(ValueError, match=match):
        opts._prepare_driver_options()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_prepare_driver_options_rejects_incremental(driver_binding):
    options = LinkerOptions(incremental=True)
    with pytest.raises(ValueError, match="incremental option is not supported by the driver API"):
        options._prepare_driver_options()


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize("value", [True, False])
def test_numba_debug_warns_and_is_ignored(value):
    """No linking backend reads ``numba_debug``, so it is ignored -- but not
    silently, which was the bug in #2640.

    The gate is ``is not None``, not truthiness: it is the field itself that is
    deprecated, so ``numba_debug=False`` earns the notice too even though it
    asks for nothing.
    """
    with pytest.warns(DeprecationWarning, match="numba_debug is not supported by any linking backend"):
        opts = LinkerOptions(arch="sm_80", debug=True, numba_debug=value)
    # Warned, not rejected, and the rest of the option set is untouched.
    assert opts._prepare_nvjitlink_options(as_bytes=True) == [b"-arch=sm_80", b"-g"]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_numba_debug_unset_does_not_warn():
    """The deprecation notice fires only when the field is explicitly set."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        options = LinkerOptions(arch="sm_80", debug=True)._prepare_nvjitlink_options(as_bytes=True)
    assert options == [b"-arch=sm_80", b"-g"]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_numba_debug_ignored_by_driver_backend_too(driver_binding):
    """The cuLink driver API has no CUjit_option for numba_debug either, so it
    is ignored there as well rather than reaching the driver."""
    with pytest.warns(DeprecationWarning, match="numba_debug"):
        opts = LinkerOptions(arch="sm_80", numba_debug=True)
    formatted_options, option_keys = opts._prepare_driver_options()
    assert not any("NUMBA" in str(key) for key in option_keys)


def test_linker_empty_object_codes_raises():
    """Linker with no ObjectCode raises ValueError."""
    with pytest.raises(ValueError, match="At least one ObjectCode object must be provided"):
        Linker()


def test_as_bytes_nvjitlink_unavailable(monkeypatch):
    """as_bytes('nvjitlink') raises RuntimeError when the backend is unavailable."""
    monkeypatch.setattr(_linker, "_use_nvjitlink_backend", False)
    opts = LinkerOptions(arch="sm_80")
    with pytest.raises(RuntimeError, match="nvJitLink backend is not available"):
        opts.as_bytes("nvjitlink")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_require_nvjitlink_version_reports_required_and_detected_versions(monkeypatch):
    monkeypatch.setattr(_linker, "_nvjitlink_version", (13, 1))

    with pytest.raises(RuntimeError, match=r"requires nvJitLink 13\.2 or newer; found 13\.1"):
        _linker._require_nvjitlink_version((13, 2), "incremental linking")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_require_nvjitlink_version_accepts_boundary_version(monkeypatch):
    monkeypatch.setattr(_linker, "_nvjitlink_version", (13, 2))

    _linker._require_nvjitlink_version((13, 2), "incremental linking")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_linked_ltoir_output_requires_new_enough_runtime(monkeypatch):
    monkeypatch.setattr(_linker, "_nvjitlink_version", (13, 2))

    with pytest.raises(RuntimeError, match=r"LTOIR output requires nvJitLink 13\.3 or newer; found 13\.2"):
        _linker._linked_ltoir_output_module()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_linked_ltoir_output_requires_new_enough_bindings(monkeypatch):
    class NvJitLinkWithoutLinkedLtoir:
        pass

    monkeypatch.setattr(_linker, "_nvjitlink", NvJitLinkWithoutLinkedLtoir)
    monkeypatch.setattr(_linker, "_nvjitlink_version", (13, 3))

    with pytest.raises(RuntimeError, match="cuda-bindings with get_linked_ltoir_size and get_linked_ltoir"):
        _linker._linked_ltoir_output_module()


incremental_caller = r"""
extern "C" __device__ int incremental_helper();
extern "C" __global__ void incremental_kernel(int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = incremental_helper();
    }
}
"""

incremental_helper = r"""
extern "C" __device__ int incremental_helper() { return 42; }
"""


def _compile_incremental_inputs(target_type):
    if target_type == "ptx":
        options = ProgramOptions(relocatable_device_code=True)
    else:
        options = ProgramOptions(link_time_optimization=True)
    caller = Program(incremental_caller, "c++", options).compile(target_type)
    helper = Program(incremental_helper, "c++", options).compile(target_type)
    return caller, helper


def _launch_incrementally_linked_kernel(device, linked_code):
    kernel = linked_code.get_kernel("incremental_kernel")
    stream = device.create_stream()
    try:
        with LegacyPinnedMemoryResource().allocate(4) as host_buffer:
            result = np.from_dlpack(host_buffer).view(np.int32)
            try:
                with device.memory_resource.allocate(4, stream=stream) as device_buffer:
                    result[:] = 0

                    launch(stream, LaunchConfig(grid=1, block=1), kernel, device_buffer)
                    device_buffer.copy_to(host_buffer, stream=stream)
                    stream.sync()
                    actual = int(result[0])
            finally:
                # Drop the DLPack view before releasing its pinned allocation.
                result = None

        assert actual == 42
    finally:
        try:
            stream.sync()
        finally:
            stream.close()


@pytest.mark.human_reviewed
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 2),
    reason="incremental linking requires nvJitLink 13.2 or newer",
)
def test_incremental_cubin_round_trip(init_cuda):
    caller, helper = _compile_incremental_inputs("ptx")

    partial = Linker(caller, options=LinkerOptions(arch=ARCH, incremental=True)).link("cubin")
    assert partial.code_type == "cubin"

    resolved_partial = Linker(
        partial,
        helper,
        options=LinkerOptions(arch=ARCH, incremental=True),
    ).link("cubin")
    assert resolved_partial.code_type == "cubin"
    _launch_incrementally_linked_kernel(init_cuda, resolved_partial)

    final = Linker(resolved_partial, options=LinkerOptions(arch=ARCH)).link("cubin")
    _launch_incrementally_linked_kernel(init_cuda, final)


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 3) or not has_linked_ltoir_bindings,
    reason="linked LTOIR output requires nvJitLink 13.3 or newer and matching cuda-bindings",
)
def test_incremental_ltoir_round_trip(init_cuda):
    caller, helper = _compile_incremental_inputs("ltoir")
    incremental_options = LinkerOptions(
        arch=ARCH,
        incremental=True,
        link_time_optimization=True,
    )

    partial = Linker(caller, options=incremental_options).link("ltoir")
    assert partial.code_type == "ltoir"

    resolved_partial = Linker(partial, helper, options=incremental_options).link("ltoir")
    assert resolved_partial.code_type == "ltoir"

    final = Linker(
        resolved_partial,
        options=LinkerOptions(arch=ARCH, link_time_optimization=True),
    ).link("cubin")
    _launch_incrementally_linked_kernel(init_cuda, final)


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 3) or not has_linked_ltoir_bindings,
    reason="linked LTOIR output requires nvJitLink 13.3 or newer and matching cuda-bindings",
)
def test_complete_ltoir_round_trip_matches_direct_cubin(init_cuda):
    caller, helper = _compile_incremental_inputs("ltoir")
    options = LinkerOptions(arch=ARCH, link_time_optimization=True)

    linked_ltoir = Linker(caller, helper, options=options).link("ltoir")
    direct_cubin = Linker(caller, helper, options=options).link("cubin")
    round_trip_cubin = Linker(linked_ltoir, options=options).link("cubin")

    assert linked_ltoir.code_type == "ltoir"
    assert round_trip_cubin.code == direct_cubin.code
    _launch_incrementally_linked_kernel(init_cuda, round_trip_cubin)


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 2),
    reason="incremental linking requires nvJitLink 13.2 or newer",
)
def test_incremental_lto_cubin_round_trip(init_cuda):
    caller, helper = _compile_incremental_inputs("ltoir")
    partial = Linker(
        caller,
        options=LinkerOptions(
            arch=ARCH,
            incremental=True,
            link_time_optimization=True,
        ),
    ).link("cubin")

    assert partial.code_type == "cubin"
    assert partial.code.startswith(b"\x7fELF")
    assert int.from_bytes(partial.code[16:18], "little") == 1  # ET_REL

    final = Linker(
        partial,
        helper,
        options=LinkerOptions(arch=ARCH, link_time_optimization=True),
    ).link("cubin")
    _launch_incrementally_linked_kernel(init_cuda, final)


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 3) or not has_linked_ltoir_bindings,
    reason="linked LTOIR output requires nvJitLink 13.3 or newer and matching cuda-bindings",
)
@pytest.mark.parametrize("non_ltoir_type", ("ptx", "cubin"))
def test_ltoir_output_rejects_inputs_without_ltoir(init_cuda, non_ltoir_type):
    caller, _ = _compile_incremental_inputs("ltoir")
    other_kernel = 'extern "C" __global__ void other_kernel() {}'
    non_ltoir_input = Program(
        other_kernel,
        "c++",
        ProgramOptions(relocatable_device_code=True),
    ).compile(non_ltoir_type)
    linker = Linker(
        caller,
        non_ltoir_input,
        options=LinkerOptions(
            arch=ARCH,
            incremental=True,
            link_time_optimization=True,
        ),
    )

    with pytest.raises(ValueError, match='LTOIR output is not supported with "ptx" or "cubin" inputs'):
        linker.link("ltoir")


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(
    is_culink_backend or nvjitlink_version < (13, 2),
    reason="incremental linking requires nvJitLink 13.2 or newer",
)
def test_incremental_link_rejects_ptx_output(compile_ptx_functions):
    linker = Linker(
        *compile_ptx_functions,
        options=LinkerOptions(arch=ARCH, incremental=True),
    )
    with pytest.raises(ValueError, match="PTX output is not supported for incremental linking"):
        linker.link("ptx")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_incremental_link_rejects_ptx_option(compile_ptx_functions):
    with pytest.raises(ValueError, match="incremental and ptx output options cannot be used together"):
        Linker(
            *compile_ptx_functions,
            options=LinkerOptions(
                arch=ARCH,
                incremental=True,
                link_time_optimization=True,
                ptx=True,
            ),
        )


@pytest.mark.agent_authored(model="gpt-5.6")
@pytest.mark.skipif(is_culink_backend, reason="LTOIR output requires nvJitLink")
def test_ltoir_output_requires_lto(compile_ptx_functions):
    linker = Linker(*compile_ptx_functions, options=LinkerOptions(arch=ARCH))
    with pytest.raises(ValueError, match="LTOIR output requires link_time_optimization=True"):
        linker.link("ltoir")
