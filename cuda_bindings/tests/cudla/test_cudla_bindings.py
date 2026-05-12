# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

cudla = pytest.importorskip("cuda.bindings.cudla")


def _cudla_library_available():
    """Check if the cuDLA shared library is loaded and usable."""
    try:
        from cuda.bindings._internal import cudla as _inner

        return _inner._inspect_function_pointer("__cudlaGetVersion") != 0
    except Exception:
        return False


requires_cudla_library = pytest.mark.skipif(
    not _cudla_library_available(),
    reason="cuDLA library not available (requires NVIDIA Orin with DLA)",
)


def _make_fence(fence_value):
    fence = cudla.Fence()
    fence.fence = fence_value
    fence.type = int(cudla.FenceType.NVSCISYNC_FENCE)
    return fence


def _load_fake_cudla_library(tmp_path):
    if sys.platform == "win32":
        pytest.skip("fake cuDLA backend test is Linux-only")

    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler available for fake cuDLA backend test")

    source_path = tmp_path / "fake_cudla.c"
    library_path = tmp_path / "libfakecudla.so"
    source_path.write_text(
        textwrap.dedent(
            """\
            #include <stddef.h>
            #include <stdint.h>

            typedef int32_t cudlaStatus;
            typedef int32_t cudlaModuleAttributeType;
            typedef int32_t cudlaDevAttributeType;
            typedef void* cudlaDevHandle;
            typedef void* cudlaModule;

            typedef union {
                uint8_t unifiedAddressingSupported;
                uint32_t deviceVersion;
            } cudlaDevAttribute;

            typedef struct {
                char name[81];
                uint64_t size;
                uint64_t n;
                uint64_t c;
                uint64_t h;
                uint64_t w;
                uint8_t dataFormat;
                uint8_t dataType;
                uint8_t dataCategory;
                uint8_t pixelFormat;
                uint8_t pixelMapping;
                uint32_t stride[8];
            } cudlaModuleTensorDescriptor;

            typedef union {
                uint32_t numInputTensors;
                uint32_t numOutputTensors;
                cudlaModuleTensorDescriptor* inputTensorDesc;
                cudlaModuleTensorDescriptor* outputTensorDesc;
            } cudlaModuleAttribute;

            typedef struct {
                cudlaModule moduleHandle;
                uint64_t** outputTensor;
                uint32_t numOutputTensors;
                uint32_t numInputTensors;
                uint64_t** inputTensor;
                void* waitEvents;
                void* signalEvents;
            } cudlaTask;

            cudlaStatus cudlaGetVersion(uint64_t* version) {
                if (version != NULL) {
                    *version = 13002075ULL;
                }
                return 0;
            }

            cudlaStatus cudlaDeviceGetCount(uint64_t* numDevices) {
                if (numDevices != NULL) {
                    *numDevices = 1;
                }
                return 0;
            }

            cudlaStatus cudlaCreateDevice(const uint64_t device, cudlaDevHandle* devHandle, const uint32_t flags) {
                (void)device;
                (void)flags;
                if (devHandle != NULL) {
                    *devHandle = (void*)0x1234;
                }
                return 0;
            }

            cudlaStatus cudlaMemRegister(
                const cudlaDevHandle devHandle,
                const uint64_t* ptr,
                const size_t size,
                uint64_t** devPtr,
                const uint32_t flags
            ) {
                (void)devHandle;
                (void)ptr;
                (void)size;
                (void)flags;
                if (devPtr != NULL) {
                    *devPtr = (uint64_t*)0x5678;
                }
                return 0;
            }

            cudlaStatus cudlaModuleLoadFromMemory(
                const cudlaDevHandle devHandle,
                const uint8_t* moduleData,
                const size_t moduleSize,
                cudlaModule* moduleHandle,
                const uint32_t flags
            ) {
                (void)devHandle;
                (void)moduleData;
                (void)moduleSize;
                (void)flags;
                if (moduleHandle != NULL) {
                    *moduleHandle = (void*)0x2222;
                }
                return 0;
            }

            cudlaStatus cudlaModuleGetAttributes(
                const cudlaModule moduleHandle,
                const cudlaModuleAttributeType attrType,
                cudlaModuleAttribute* attribute
            ) {
                (void)moduleHandle;
                (void)attrType;
                if (attribute != NULL) {
                    attribute->numInputTensors = 0;
                }
                return 0;
            }

            cudlaStatus cudlaModuleUnload(const cudlaModule moduleHandle, const uint32_t flags) {
                (void)moduleHandle;
                (void)flags;
                return 0;
            }

            cudlaStatus cudlaSubmitTask(
                const cudlaDevHandle devHandle,
                const cudlaTask* tasks,
                const uint32_t numTasks,
                void* stream,
                const uint32_t flags
            ) {
                (void)devHandle;
                (void)tasks;
                (void)numTasks;
                (void)stream;
                (void)flags;
                return 0;
            }

            cudlaStatus cudlaDeviceGetAttribute(
                const cudlaDevHandle devHandle,
                const cudlaDevAttributeType attrib,
                cudlaDevAttribute* attribute
            ) {
                (void)devHandle;
                (void)attrib;
                if (attribute != NULL) {
                    attribute->deviceVersion = 1;
                }
                return 0;
            }

            cudlaStatus cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* devPtr) {
                (void)devHandle;
                (void)devPtr;
                return 0;
            }

            cudlaStatus cudlaGetLastError(const cudlaDevHandle devHandle) {
                (void)devHandle;
                return 0;
            }

            cudlaStatus cudlaDestroyDevice(const cudlaDevHandle devHandle) {
                (void)devHandle;
                return 0;
            }

            cudlaStatus cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) {
                (void)devHandle;
                (void)timeout;
                return 0;
            }
            """
        ),
        encoding="utf-8",
    )
    try:
        subprocess.run(  # noqa: S603 - trusted compiler path from shutil.which and temp test inputs
            [compiler, "-shared", "-fPIC", "-o", str(library_path), str(source_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"failed to build fake cuDLA backend test library: {exc}")

    mode = getattr(ctypes, "RTLD_GLOBAL", getattr(os, "RTLD_GLOBAL", 0))
    return ctypes.CDLL(str(library_path), mode=mode)


# ---------------------------------------------------------------------------
# Enum tests (always run -- no library needed)
# ---------------------------------------------------------------------------


class TestEnums:
    def test_status_values(self):
        assert cudla.Status.Success == 0
        assert cudla.Status.ErrorInvalidParam == 1
        assert cudla.Status.ErrorOutOfResources == 2
        assert cudla.Status.ErrorCreationFailed == 3
        assert cudla.Status.ErrorInvalidAddress == 4
        assert cudla.Status.ErrorOs == 5
        assert cudla.Status.ErrorCuda == 6
        assert cudla.Status.ErrorUmd == 7
        assert cudla.Status.ErrorInvalidDevice == 8
        assert cudla.Status.ErrorInvalidAttribute == 9
        assert cudla.Status.ErrorIncompatibleDlaSWVersion == 10
        assert cudla.Status.ErrorMemoryRegistered == 11
        assert cudla.Status.ErrorInvalidModule == 12
        assert cudla.Status.ErrorUnsupportedOperation == 13
        assert cudla.Status.ErrorNvSci == 14
        assert cudla.Status.ErrorDlaErrInvalidInput == 0x40000001
        assert cudla.Status.ErrorDlaErrInvalidPreAction == 0x40000002
        assert cudla.Status.ErrorDlaErrNoMem == 0x40000003
        assert cudla.Status.ErrorDlaErrProcessorBusy == 0x40000004
        assert cudla.Status.ErrorDlaErrTaskStatusMismatch == 0x40000005
        assert cudla.Status.ErrorDlaErrEngineTimeout == 0x40000006
        assert cudla.Status.ErrorDlaErrDataMismatch == 0x40000007
        assert cudla.Status.ErrorUnknown == 0x7FFFFFFF

    def test_status_member_count(self):
        assert len(cudla.Status) == 23

    def test_mode_values(self):
        assert cudla.Mode.CUDA_DLA == 0
        assert cudla.Mode.STANDALONE == 1

    def test_module_attribute_type_values(self):
        assert cudla.ModuleAttributeType.NUM_INPUT_TENSORS == 0
        assert cudla.ModuleAttributeType.NUM_OUTPUT_TENSORS == 1
        assert cudla.ModuleAttributeType.INPUT_TENSOR_DESCRIPTORS == 2
        assert cudla.ModuleAttributeType.OUTPUT_TENSOR_DESCRIPTORS == 3
        assert cudla.ModuleAttributeType.NUM_OUTPUT_TASK_STATISTICS == 4
        assert cudla.ModuleAttributeType.OUTPUT_TASK_STATISTICS_DESCRIPTORS == 5

    def test_fence_type_values(self):
        assert cudla.FenceType.NVSCISYNC_FENCE == 1
        assert cudla.FenceType.NVSCISYNC_FENCE_SOF == 2

    def test_module_load_flags(self):
        assert cudla.ModuleLoadFlags.MODULE_DEFAULT == 0
        assert cudla.ModuleLoadFlags.MODULE_ENABLE_FAULT_DIAGNOSTICS == 1

    def test_submission_flags(self):
        assert cudla.SubmissionFlags.SUBMIT_NOOP == 1
        assert cudla.SubmissionFlags.SUBMIT_SKIP_LOCK_ACQUIRE == 2
        assert cudla.SubmissionFlags.SUBMIT_DIAGNOSTICS_TASK == 4

    def test_access_permission_flags(self):
        assert cudla.AccessPermissionFlags.READ_WRITE_PERM == 0
        assert cudla.AccessPermissionFlags.READ_ONLY_PERM == 1
        assert cudla.AccessPermissionFlags.TASK_STATISTICS == 2

    def test_dev_attribute_type(self):
        assert cudla.DevAttributeType.UNIFIED_ADDRESSING == 0
        assert cudla.DevAttributeType.DEVICE_VERSION == 1


# ---------------------------------------------------------------------------
# POD type tests (always run -- no library needed)
# ---------------------------------------------------------------------------


class TestPodTypes:
    def test_external_memory_handle_desc(self):
        desc = cudla.ExternalMemoryHandleDesc()
        desc.size_ = 4096
        assert desc.size_ == 4096
        desc.ext_buf_object = 0xABCD
        assert desc.ext_buf_object == 0xABCD

    def test_external_semaphore_handle_desc(self):
        desc = cudla.ExternalSemaphoreHandleDesc()
        desc.ext_sync_object = 0x1234
        assert desc.ext_sync_object == 0x1234

    def test_module_tensor_descriptor_fields(self):
        desc = cudla.ModuleTensorDescriptor()
        assert desc.size_ == 0
        assert desc.n == 0
        assert desc.c == 0
        assert desc.h == 0
        assert desc.w == 0
        assert desc.data_format == 0
        assert desc.data_type == 0
        assert desc.data_category == 0
        assert desc.pixel_format == 0
        assert desc.pixel_mapping == 0

    def test_module_tensor_descriptor_name(self):
        desc = cudla.ModuleTensorDescriptor()
        name = desc.name
        assert isinstance(name, (str, bytes))

    def test_module_tensor_descriptor_stride(self):
        desc = cudla.ModuleTensorDescriptor()
        stride = desc.stride
        assert len(stride) == 8

    def test_fence(self):
        fence = cudla.Fence()
        fence.fence = 0xBEEF
        assert fence.fence == 0xBEEF
        fence.type = int(cudla.FenceType.NVSCISYNC_FENCE)
        assert fence.type == 1

    def test_dev_attribute(self):
        attr = cudla.DevAttribute()
        assert attr.unified_addressing_supported == 0
        assert attr.device_version == 0
        attr.unified_addressing_supported = 1
        assert attr.unified_addressing_supported == 1
        attr.device_version = 0x20
        assert attr.device_version == 0x20

    def test_module_attribute(self):
        attr = cudla.ModuleAttribute()
        assert attr.num_input_tensors == 0
        assert attr.num_output_tensors == 0
        attr.num_input_tensors = 3
        assert attr.num_input_tensors == 3
        attr.num_output_tensors = 1
        assert attr.num_output_tensors == 1

    def test_wait_events(self):
        we = cudla.WaitEvents()
        assert we.pre_fences == []

    def test_signal_events(self):
        se = cudla.SignalEvents()
        assert se.eof_fences == []

    def test_task_construction(self):
        task = cudla.Task()
        task.module_handle = 0xDEAD
        assert task.module_handle == 0xDEAD

    def test_task_input_tensor_auto_size(self):
        task = cudla.Task()
        task.input_tensor = [0x1000, 0x2000, 0x3000]
        assert len(task.input_tensor) == 3

    def test_task_output_tensor_auto_size(self):
        task = cudla.Task()
        task.output_tensor = [0x4000]
        assert len(task.output_tensor) == 1

    def test_task_combined(self):
        task = cudla.Task()
        task.module_handle = 0xABCD
        task.input_tensor = [0x1000, 0x2000]
        task.output_tensor = [0x3000]
        task.wait_events = 0
        task.signal_events = 0
        assert task.module_handle == 0xABCD
        assert len(task.input_tensor) == 2
        assert len(task.output_tensor) == 1

    def test_pod_ptr_is_nonzero(self):
        """Verify that int(pod) returns a nonzero pointer (memory is allocated)."""
        task = cudla.Task()
        assert int(task) != 0
        desc = cudla.ModuleTensorDescriptor()
        assert int(desc) != 0


# ---------------------------------------------------------------------------
# Error type tests (always run -- no library needed)
# ---------------------------------------------------------------------------


class TestErrorType:
    def test_cudla_error_is_exception(self):
        assert issubclass(cudla.CudlaError, Exception)

    def test_cudla_error_stores_status(self):
        err = cudla.CudlaError(int(cudla.Status.ErrorInvalidParam))
        assert err.status == int(cudla.Status.ErrorInvalidParam)

    def test_cudla_error_str(self):
        err = cudla.CudlaError(int(cudla.Status.ErrorInvalidParam))
        assert "ErrorInvalidParam" in str(err)


# ---------------------------------------------------------------------------
# API surface tests (always run -- no library needed)
# ---------------------------------------------------------------------------


class TestApiSurface:
    """Verify that all expected functions exist as callable attributes."""

    @pytest.mark.parametrize(
        "func_name",
        [
            "get_version",
            "device_get_count",
            "create_device",
            "destroy_device",
            "mem_register",
            "mem_unregister",
            "module_load_from_memory",
            "module_get_attributes",
            "module_unload",
            "submit_task",
            "device_get_attribute",
            "get_last_error",
            "set_task_timeout_in_ms",
        ],
    )
    def test_function_exists(self, func_name):
        assert callable(getattr(cudla, func_name))


class TestDocumentedApiSurface:
    def test_documented_functions_exist(self):
        documented = [
            "import_external_memory",
            "import_external_semaphore",
            "get_nv_sci_sync_attributes",
        ]
        missing = [func_name for func_name in documented if not hasattr(cudla, func_name)]
        assert not missing, f"documented cuDLA functions are missing: {missing}"

    def test_documented_status_success_member_exists(self):
        assert hasattr(cudla.Status, "SUCCESS")


class TestFenceArraySemantics:
    def test_wait_events_pre_fences_round_trip(self):
        wait_events = cudla.WaitEvents()
        wait_events.pre_fences = [_make_fence(0x1010), _make_fence(0x2020)]

        pre_fences = wait_events.pre_fences
        assert len(pre_fences) == 2
        assert [fence.fence for fence in pre_fences] == [0x1010, 0x2020]

    def test_signal_events_eof_fences_round_trip(self):
        signal_events = cudla.SignalEvents()
        signal_events.eof_fences = [_make_fence(0x3030), _make_fence(0x4040)]

        eof_fences = signal_events.eof_fences
        assert len(eof_fences) == 2
        assert [fence.fence for fence in eof_fences] == [0x3030, 0x4040]


class TestTaskReferenceRetention:
    @pytest.mark.parametrize(
        ("attr_name", "wrapper_factory"),
        [
            ("wait_events", cudla.WaitEvents),
            ("signal_events", cudla.SignalEvents),
        ],
    )
    def test_task_retains_assigned_event_wrappers(self, attr_name, wrapper_factory):
        task = cudla.Task()
        wrapper = wrapper_factory()

        baseline_refcount = sys.getrefcount(wrapper)
        setattr(task, attr_name, wrapper)

        assert sys.getrefcount(wrapper) > baseline_refcount


class TestStandaloneMode:
    def test_create_device_accepts_standalone_mode_when_backend_supports_it(self, tmp_path):
        if _cudla_library_available():
            pytest.skip("requires a host without a preloaded cuDLA runtime")

        fake_cudla = _load_fake_cudla_library(tmp_path)
        assert fake_cudla is not None

        handle = cudla.create_device(0, int(cudla.Mode.STANDALONE))
        assert handle == 0x1234


# ---------------------------------------------------------------------------
# Function tests (hardware-gated -- skipped when libcudla.so is unavailable)
# ---------------------------------------------------------------------------


@requires_cudla_library
class TestFunctions:
    def test_get_version(self):
        version = cudla.get_version()
        assert version > 0

    def test_device_get_count(self):
        count = cudla.device_get_count()
        assert count >= 0

    def test_create_destroy_device(self):
        from cuda.bindings import driver, runtime

        driver.cuInit(0)
        runtime.cudaSetDevice(0)

        handle = cudla.create_device(0, int(cudla.Mode.CUDA_DLA))
        try:
            assert handle != 0
        finally:
            cudla.destroy_device(handle)

    def test_mem_register_unregister(self):
        from cuda.bindings import driver, runtime

        driver.cuInit(0)
        runtime.cudaSetDevice(0)

        dev_handle = cudla.create_device(0, int(cudla.Mode.CUDA_DLA))
        try:
            buf_size = 1024
            err, gpu_ptr = runtime.cudaMalloc(buf_size)
            assert err.value == 0, f"cudaMalloc failed: {err}"
            try:
                registered_ptr = cudla.mem_register(dev_handle, int(gpu_ptr), buf_size, 0)
                assert registered_ptr != 0
                cudla.mem_unregister(dev_handle, registered_ptr)
            finally:
                runtime.cudaFree(gpu_ptr)
        finally:
            cudla.destroy_device(dev_handle)
