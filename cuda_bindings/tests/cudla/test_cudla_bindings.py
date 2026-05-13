# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import pytest

cudla = pytest.importorskip("cuda.bindings.cudla")


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
