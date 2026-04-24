# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass

from libc.stdint cimport intptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.version cimport cy_binding_version, cy_driver_version


__all__ = [
    "DeviceResources",
    "SMResource",
    "SMResourceOptions",
    "WorkqueueResource",
    "WorkqueueResourceOptions",
]


cdef inline void _check_green_ctx_support() except *:
    cdef tuple drv = cy_driver_version()
    cdef tuple bind = cy_binding_version()
    if drv < (12, 4, 0):
        raise NotImplementedError(
            "Green context support requires CUDA driver 12.4 or newer. "
            f"Using driver version {'.'.join(map(str, drv))}"
        )
    if bind < (12, 4, 0):
        raise NotImplementedError(
            "Green context support requires cuda.bindings 12.4 or newer. "
            f"Using cuda.bindings version {'.'.join(map(str, bind))}"
        )


cdef inline void _check_workqueue_support() except *:
    cdef tuple drv = cy_driver_version()
    cdef tuple bind = cy_binding_version()
    if drv < (13, 1, 0):
        raise NotImplementedError(
            "WorkqueueResource requires CUDA driver 13.1 or newer. "
            f"Using driver version {'.'.join(map(str, drv))}"
        )
    if bind < (13, 1, 0):
        raise NotImplementedError(
            "WorkqueueResource requires cuda.bindings 13.1 or newer. "
            f"Using cuda.bindings version {'.'.join(map(str, bind))}"
        )


@dataclass
class SMResourceOptions:
    """Options for :meth:`SMResource.split`.

    ``count`` determines the number of requested groups. Scalar ``count`` or
    ``None`` creates one group; a sequence creates ``len(count)`` groups. Other
    sequence fields must match the length of ``count``.
    """

    count: int | SequenceABC | None = None
    coscheduled_sm_count: int | SequenceABC | None = None
    preferred_coscheduled_sm_count: int | SequenceABC | None = None


@dataclass
class WorkqueueResourceOptions:
    """Options for :meth:`WorkqueueResource.configure`."""

    sharing_scope: str | None = None


cdef inline bint _is_sequence(object value):
    return (
        isinstance(value, SequenceABC)
        and not isinstance(value, (str, bytes, bytearray))
    )


cdef int _resolve_group_count(object options) except -1:
    cdef object count = options.count
    cdef int n_groups
    cdef object value
    cdef str field_name

    if count is None or isinstance(count, int):
        n_groups = 1
    elif _is_sequence(count):
        n_groups = len(count)
        if n_groups == 0:
            raise ValueError("count sequence must not be empty")
    else:
        raise TypeError(f"count must be int, Sequence, or None, got {type(count)}")

    if n_groups == 1:
        for field_name in (
            "coscheduled_sm_count",
            "preferred_coscheduled_sm_count",
        ):
            value = getattr(options, field_name)
            if _is_sequence(value):
                raise ValueError(
                    f"{field_name} is a Sequence but count is scalar; "
                    "count must be a Sequence to specify multiple groups"
                )
    else:
        for field_name in (
            "coscheduled_sm_count",
            "preferred_coscheduled_sm_count",
        ):
            value = getattr(options, field_name)
            if _is_sequence(value) and len(value) != n_groups:
                raise ValueError(
                    f"{field_name} has length {len(value)}, expected {n_groups} "
                    "(must match count)"
                )
    return n_groups


cdef object _broadcast_field(object value, int n_groups):
    if _is_sequence(value):
        return list(value)
    return [value] * n_groups


cdef inline unsigned int _as_uint(object value, str field_name) except? 0:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int or None, got {type(value)}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return <unsigned int>value


cdef inline unsigned int _count_to_sm_count(object value) except? 0:
    if value is None:
        return 0
    return _as_uint(value, "count")


cdef inline bint _can_use_structured_sm_split():
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        return cy_driver_version() >= (13, 1, 0) and cy_binding_version() >= (13, 1, 0)
    ELSE:
        return False


cdef inline void _check_split_by_count_support() except *:
    cdef tuple drv = cy_driver_version()
    cdef tuple bind = cy_binding_version()
    if drv < (12, 4, 0):
        raise NotImplementedError(
            "SMResource.split() requires CUDA driver 12.4 or newer. "
            f"Using driver version {'.'.join(map(str, drv))}"
        )
    if bind < (12, 4, 0):
        raise NotImplementedError(
            "SMResource.split() requires cuda.bindings 12.4 or newer. "
            f"Using cuda.bindings version {'.'.join(map(str, bind))}"
        )


cdef object _resolve_split_by_count_request(object options):
    cdef int n_groups = _resolve_group_count(options)
    cdef list counts = _broadcast_field(options.count, n_groups)
    cdef object first = counts[0]
    cdef object value
    cdef unsigned int min_count

    if options.coscheduled_sm_count is not None:
        raise NotImplementedError(
            "SMResourceOptions.coscheduled_sm_count requires the CUDA 13.1 "
            "structured SM split API"
        )
    if options.preferred_coscheduled_sm_count is not None:
        raise NotImplementedError(
            "SMResourceOptions.preferred_coscheduled_sm_count requires the "
            "CUDA 13.1 structured SM split API"
        )

    for value in counts[1:]:
        if value != first:
            raise NotImplementedError(
                "CUDA 12 SM splitting only supports homogeneous count values; "
                "use CUDA 13.1 or newer for per-group counts"
            )

    min_count = _count_to_sm_count(first)
    return n_groups, min_count


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef void _fill_group_params(
        cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS* params,
        int n_groups,
        object options,
    ) except *:
        cdef list counts = _broadcast_field(options.count, n_groups)
        cdef list coscheduled = _broadcast_field(options.coscheduled_sm_count, n_groups)
        cdef list preferred = _broadcast_field(options.preferred_coscheduled_sm_count, n_groups)
        cdef int i

        for i in range(n_groups):
            memset(&params[i], 0, sizeof(cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS))
            params[i].smCount = _count_to_sm_count(counts[i])
            if coscheduled[i] is not None:
                params[i].coscheduledSmCount = _as_uint(coscheduled[i], "coscheduled_sm_count")
            if preferred[i] is not None:
                params[i].preferredCoscheduledSmCount = _as_uint(
                    preferred[i], "preferred_coscheduled_sm_count"
                )
            params[i].flags = 0


    cdef object _split_with_general_api(SMResource sm, object options, bint dry_run):
        cdef int n_groups = _resolve_group_count(options)
        cdef cydriver.CUdevResource* result = NULL
        cdef cydriver.CUdevResource remaining
        cdef cydriver.CUdevResource synth
        cdef cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS* params = NULL
        cdef list groups = []
        cdef int i

        params = <cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS*>malloc(
            n_groups * sizeof(cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS)
        )
        if params == NULL:
            raise MemoryError()

        try:
            _fill_group_params(params, n_groups, options)

            if not dry_run:
                result = <cydriver.CUdevResource*>malloc(
                    n_groups * sizeof(cydriver.CUdevResource)
                )
                if result == NULL:
                    raise MemoryError()

            memset(&remaining, 0, sizeof(cydriver.CUdevResource))
            with nogil:
                HANDLE_RETURN(cydriver.cuDevSmResourceSplit(
                    result,
                    <unsigned int>n_groups,
                    &sm._resource,
                    &remaining,
                    0,
                    params,
                ))

            if result != NULL:
                for i in range(n_groups):
                    groups.append(SMResource._from_dev_resource(result[i]))
                return groups, SMResource._from_dev_resource(remaining)

            for i in range(n_groups):
                memset(&synth, 0, sizeof(cydriver.CUdevResource))
                synth.type = cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
                synth.sm.smCount = params[i].smCount
                groups.append(SMResource._from_dry_run_resource(synth))
            return groups, SMResource._from_dry_run_resource(remaining)
        finally:
            if params != NULL:
                free(params)
            if result != NULL:
                free(result)
ELSE:
    cdef object _split_with_general_api(SMResource sm, object options, bint dry_run):
        raise NotImplementedError(
            "SMResource.split() requires cuda.core to be built with CUDA 13.x bindings"
        )


cdef object _split_with_count_api(SMResource sm, object options, bint dry_run):
    cdef object request = _resolve_split_by_count_request(options)
    cdef unsigned int nb_groups = <unsigned int>request[0]
    cdef unsigned int min_count = <unsigned int>request[1]
    cdef unsigned int actual_groups = nb_groups
    cdef cydriver.CUdevResource* result = NULL
    cdef cydriver.CUdevResource remaining
    cdef list groups = []
    cdef int i

    result = <cydriver.CUdevResource*>malloc(nb_groups * sizeof(cydriver.CUdevResource))
    if result == NULL:
        raise MemoryError()

    try:
        memset(&remaining, 0, sizeof(cydriver.CUdevResource))
        with nogil:
            HANDLE_RETURN(cydriver.cuDevSmResourceSplitByCount(
                result,
                &actual_groups,
                &sm._resource,
                &remaining,
                0,
                min_count,
            ))

        for i in range(actual_groups):
            if dry_run:
                groups.append(SMResource._from_dry_run_resource(result[i]))
            else:
                groups.append(SMResource._from_dev_resource(result[i]))
        if dry_run:
            return groups, SMResource._from_dry_run_resource(remaining)
        return groups, SMResource._from_dev_resource(remaining)
    finally:
        free(result)


cdef class SMResource:
    """SM resource queried from a device. Not user-constructible."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SMResource cannot be instantiated directly. "
            "Use dev.resources.sm or SMResource.split()."
        )

    @staticmethod
    cdef SMResource _from_dev_resource(cydriver.CUdevResource res):
        cdef SMResource self = SMResource.__new__(SMResource)
        self._resource = res
        self._is_usable = True
        return self

    @staticmethod
    cdef SMResource _from_dry_run_resource(cydriver.CUdevResource res):
        cdef SMResource self = SMResource.__new__(SMResource)
        self._resource = res
        self._is_usable = False
        return self

    @property
    def handle(self) -> int:
        """Return the address of the underlying ``CUdevResource`` struct."""
        return <intptr_t>&self._resource

    @property
    def sm_count(self) -> int:
        """Total SMs available in this resource."""
        return self._resource.sm.smCount

    @property
    def min_partition_size(self) -> int:
        """Minimum SM count required to create a partition."""
        return self._resource.sm.minSmPartitionSize

    @property
    def coscheduled_alignment(self) -> int:
        """Number of SMs guaranteed to be co-scheduled."""
        return self._resource.sm.smCoscheduledAlignment

    @property
    def flags(self) -> int:
        """Raw flags from the underlying SM resource."""
        return self._resource.sm.flags

    def split(self, options not None, *, bint dry_run=False):
        """Split this SM resource into groups plus a remainder."""
        if not isinstance(options, SMResourceOptions):
            raise TypeError(f"options must be SMResourceOptions, got {type(options)}")
        _resolve_group_count(options)
        _check_green_ctx_support()
        if _can_use_structured_sm_split():
            return _split_with_general_api(self, options, dry_run)
        _check_split_by_count_support()
        return _split_with_count_api(self, options, dry_run)


cdef class WorkqueueResource:
    """Workqueue resource. Not user-constructible."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "WorkqueueResource cannot be instantiated directly. "
            "Use dev.resources.workqueue."
        )

    @staticmethod
    cdef WorkqueueResource _from_dev_resources(
        cydriver.CUdevResource wq_config,
        cydriver.CUdevResource wq,
    ):
        cdef WorkqueueResource self = WorkqueueResource.__new__(WorkqueueResource)
        self._wq_config_resource = wq_config
        self._wq_resource = wq
        return self

    @property
    def handle(self) -> int:
        """Return the address of the underlying config ``CUdevResource`` struct."""
        return <intptr_t>&self._wq_config_resource

    def configure(self, options not None):
        """Configure the workqueue resource in place."""
        _check_green_ctx_support()
        _check_workqueue_support()
        if not isinstance(options, WorkqueueResourceOptions):
            raise TypeError(f"options must be WorkqueueResourceOptions, got {type(options)}")
        if options.sharing_scope is None:
            return None

        IF CUDA_CORE_BUILD_MAJOR >= 13:
            if options.sharing_scope == "device_ctx":
                self._wq_config_resource.wqConfig.sharingScope = (
                    cydriver.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_DEVICE_CTX
                )
            elif options.sharing_scope == "green_ctx_balanced":
                self._wq_config_resource.wqConfig.sharingScope = (
                    cydriver.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
                )
            else:
                raise ValueError(
                    f"Unknown sharing_scope: {options.sharing_scope!r}. "
                    "Expected 'device_ctx' or 'green_ctx_balanced'."
                )
        ELSE:
            raise NotImplementedError(
                "WorkqueueResource requires cuda.core to be built with CUDA 13.x bindings"
            )


cdef class DeviceResources:
    """Namespace for hardware resource query. Not user-constructible."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DeviceResources cannot be instantiated directly. "
            "Use dev.resources."
        )

    @staticmethod
    cdef DeviceResources _init(int device_id):
        cdef DeviceResources self = DeviceResources.__new__(DeviceResources)
        self._device_id = device_id
        return self

    @property
    def sm(self) -> SMResource:
        """Query SM resources from this device."""
        _check_green_ctx_support()
        cdef cydriver.CUdevResource res
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                <cydriver.CUdevice>self._device_id,
                &res,
                cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            ))
        return SMResource._from_dev_resource(res)

    @property
    def workqueue(self) -> WorkqueueResource:
        """Query workqueue resources from this device."""
        _check_green_ctx_support()
        _check_workqueue_support()
        cdef cydriver.CUdevResource wq_config
        cdef cydriver.CUdevResource wq

        IF CUDA_CORE_BUILD_MAJOR >= 13:
            with nogil:
                HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                    <cydriver.CUdevice>self._device_id,
                    &wq_config,
                    cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
                ))
                HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                    <cydriver.CUdevice>self._device_id,
                    &wq,
                    cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE,
                ))
            return WorkqueueResource._from_dev_resources(wq_config, wq)
        ELSE:
            raise NotImplementedError(
                "WorkqueueResource requires cuda.core to be built with CUDA 13.x bindings"
            )
