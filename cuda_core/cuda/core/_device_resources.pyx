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
from cuda.core._resource_handles cimport ContextHandle, GreenCtxHandle, as_cu, get_context_green_ctx
from cuda.core._utils.cuda_utils cimport check_or_create_options, HANDLE_RETURN
from cuda.core._utils.cuda_utils import is_sequence
from cuda.core._utils.version cimport cy_binding_version, cy_driver_version


__all__ = [
    "DeviceResources",
    "SMResource",
    "SMResourceOptions",
    "WorkqueueResource",
    "WorkqueueResourceOptions",
]


# Module-level cached version checks (trinary: 0=unchecked, 1=supported, -1=unsupported)
cdef int _green_ctx_checked = 0
cdef int _workqueue_checked = 0
cdef str _green_ctx_err_msg = ""
cdef str _workqueue_err_msg = ""


cdef inline int _check_green_ctx_support() except?-1:
    global _green_ctx_checked, _green_ctx_err_msg
    if _green_ctx_checked == 1:
        return 0
    if _green_ctx_checked == -1:
        raise RuntimeError(_green_ctx_err_msg)
    cdef tuple drv = cy_driver_version()
    cdef tuple bind = cy_binding_version()
    if drv < (12, 4, 0):
        _green_ctx_err_msg = (
            "Green context support requires CUDA driver 12.4 or newer "
            f"(current driver: {'.'.join(map(str, drv))})"
        )
        _green_ctx_checked = -1
        raise RuntimeError(_green_ctx_err_msg)
    if bind < (12, 4, 0):
        _green_ctx_err_msg = (
            "Green context support requires cuda.bindings 12.4 or newer "
            f"(current bindings: {'.'.join(map(str, bind))})"
        )
        _green_ctx_checked = -1
        raise RuntimeError(_green_ctx_err_msg)
    _green_ctx_checked = 1
    return 0


cdef inline int _check_workqueue_support() except?-1:
    global _workqueue_checked, _workqueue_err_msg
    if _workqueue_checked == 1:
        return 0
    if _workqueue_checked == -1:
        raise RuntimeError(_workqueue_err_msg)
    cdef tuple drv = cy_driver_version()
    cdef tuple bind = cy_binding_version()
    if drv < (13, 1, 0):
        _workqueue_err_msg = (
            "WorkqueueResource requires CUDA driver 13.1 or newer "
            f"(current driver: {'.'.join(map(str, drv))})"
        )
        _workqueue_checked = -1
        raise RuntimeError(_workqueue_err_msg)
    if bind < (13, 1, 0):
        _workqueue_err_msg = (
            "WorkqueueResource requires cuda.bindings 13.1 or newer "
            f"(current bindings: {'.'.join(map(str, bind))})"
        )
        _workqueue_checked = -1
        raise RuntimeError(_workqueue_err_msg)
    _workqueue_checked = 1
    return 0


@dataclass
cdef class SMResourceOptions:
    """Customizable :obj:`SMResource.split` options.

    Each field accepts a scalar (for a single group) or a ``Sequence``
    (for multiple groups). ``count`` drives the number of groups; other
    ``Sequence`` fields must match its length.

    Attributes
    ----------
    count : int or Sequence[int], optional
        Requested SM count per group. ``None`` means discovery mode
        (auto-detect). (Default to ``None``)
    coscheduled_sm_count : int or Sequence[int], optional
        Minimum number of SMs guaranteed to be co-scheduled in each
        group. (Default to ``None``)
    preferred_coscheduled_sm_count : int or Sequence[int], optional
        Preferred co-scheduled SM count; the driver tries to satisfy
        this but may fall back to ``coscheduled_sm_count``.
        (Default to ``None``)
    """

    count: int | SequenceABC | None = None
    coscheduled_sm_count: int | SequenceABC | None = None
    preferred_coscheduled_sm_count: int | SequenceABC | None = None


@dataclass
cdef class WorkqueueResourceOptions:
    """Customizable :obj:`WorkqueueResource.configure` options.

    Attributes
    ----------
    sharing_scope : str, optional
        Workqueue sharing scope. Accepted values: ``"device_ctx"``
        or ``"green_ctx_balanced"``. (Default to ``None``)
    """

    sharing_scope: str | None = None


cdef inline int _validate_split_field_length(
    object value, str field_name, int n_groups, bint count_is_scalar
) except?-1:
    if count_is_scalar:
        if is_sequence(value):
            raise ValueError(
                f"{field_name} is a Sequence but count is scalar; "
                "count must be a Sequence to specify multiple groups"
            )
    elif is_sequence(value) and len(value) != n_groups:
        raise ValueError(
            f"{field_name} has length {len(value)}, expected {n_groups} "
            "(must match count)"
        )
    return 0


cdef inline int _resolve_group_count(SMResourceOptions options) except?-1:
    cdef object count = options.count
    cdef int n_groups
    cdef bint count_is_scalar

    if count is None or isinstance(count, int):
        n_groups = 1
        count_is_scalar = True
    elif is_sequence(count):
        n_groups = len(count)
        if n_groups == 0:
            raise ValueError("count sequence must not be empty")
        count_is_scalar = False
    else:
        raise TypeError(f"count must be int, Sequence, or None, got {type(count)}")

    _validate_split_field_length(
        options.coscheduled_sm_count,
        "coscheduled_sm_count",
        n_groups,
        count_is_scalar,
    )
    _validate_split_field_length(
        options.preferred_coscheduled_sm_count,
        "preferred_coscheduled_sm_count",
        n_groups,
        count_is_scalar,
    )
    return n_groups


cdef inline object _broadcast_field(object value, int n_groups):
    if is_sequence(value):
        return list(value)
    return [value] * n_groups


cdef inline unsigned int _to_sm_count(object value) except? 0:
    """Convert a count value to unsigned int. None maps to 0 (discovery)."""
    if value is None:
        return 0
    if value < 0:
        raise ValueError(f"count must be non-negative, got {value}")
    return <unsigned int>(value)


cdef int _structured_split_checked = 0

cdef inline bint _can_use_structured_sm_split():
    """Check if cuDevSmResourceSplit (13.1+) is available. Cached."""
    global _structured_split_checked
    if _structured_split_checked != 0:
        return _structured_split_checked == 1
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        if cy_driver_version() >= (13, 1, 0) and cy_binding_version() >= (13, 1, 0):
            _structured_split_checked = 1
            return True
    _structured_split_checked = -1
    return False


cdef object _resolve_split_by_count_request(SMResourceOptions options):
    cdef int n_groups = _resolve_group_count(options)
    cdef list counts = _broadcast_field(options.count, n_groups)
    cdef object first = counts[0]
    cdef object value
    cdef unsigned int min_count

    if options.coscheduled_sm_count is not None:
        raise RuntimeError(
            "SMResourceOptions.coscheduled_sm_count requires the CUDA 13.1 "
            "structured SM split API"
        )
    if options.preferred_coscheduled_sm_count is not None:
        raise RuntimeError(
            "SMResourceOptions.preferred_coscheduled_sm_count requires the "
            "CUDA 13.1 structured SM split API"
        )

    for value in counts[1:]:
        if value != first:
            raise RuntimeError(
                "CUDA 12 SM splitting only supports homogeneous count values; "
                "use CUDA 13.1 or newer for per-group counts"
            )

    min_count = _to_sm_count(first)
    return n_groups, min_count


IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline int _fill_group_params(
        cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS* params,
        int n_groups,
        SMResourceOptions options,
    ) except?-1:
        cdef list counts = _broadcast_field(options.count, n_groups)
        cdef list coscheduled = _broadcast_field(options.coscheduled_sm_count, n_groups)
        cdef list preferred = _broadcast_field(options.preferred_coscheduled_sm_count, n_groups)
        cdef int i

        for i in range(n_groups):
            memset(&params[i], 0, sizeof(cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS))
            params[i].smCount = _to_sm_count(counts[i])
            if coscheduled[i] is not None:
                params[i].coscheduledSmCount = <unsigned int>(coscheduled[i])
            if preferred[i] is not None:
                params[i].preferredCoscheduledSmCount = <unsigned int>(preferred[i])
            params[i].flags = 0
        return 0


    cdef object _split_with_general_api(SMResource sm, SMResourceOptions options, bint dry_run):
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
                    <unsigned int>(n_groups),
                    &sm._resource,
                    &remaining,
                    0,
                    params,
                ))

            if result != NULL:
                for i in range(n_groups):
                    groups.append(SMResource._from_split_resource(result[i], sm, True))
                return groups, SMResource._from_split_resource(remaining, sm, True)

            for i in range(n_groups):
                memset(&synth, 0, sizeof(cydriver.CUdevResource))
                synth.type = cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
                synth.sm.smCount = params[i].smCount
                groups.append(SMResource._from_split_resource(synth, sm, False))
            return groups, SMResource._from_split_resource(remaining, sm, False)
        finally:
            if params != NULL:
                free(params)
            if result != NULL:
                free(result)
ELSE:
    cdef object _split_with_general_api(SMResource sm, SMResourceOptions options, bint dry_run):
        raise RuntimeError(
            "SMResource.split() requires cuda.core to be built with CUDA 13.x bindings"
        )


cdef object _split_with_count_api(SMResource sm, SMResourceOptions options, bint dry_run):
    cdef object request = _resolve_split_by_count_request(options)
    cdef unsigned int nb_groups = <unsigned int>(request[0])
    cdef unsigned int min_count = <unsigned int>(request[1])
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
                groups.append(SMResource._from_split_resource(result[i], sm, False))
            else:
                groups.append(SMResource._from_split_resource(result[i], sm, True))
        if dry_run:
            return groups, SMResource._from_split_resource(remaining, sm, False)
        return groups, SMResource._from_split_resource(remaining, sm, True)
    finally:
        free(result)


cdef inline unsigned int _sm_resource_granularity(int device_id) except? 0:
    cdef int major

    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceGetAttribute(
            &major,
            cydriver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            <cydriver.CUdevice>(device_id),
        ))
    if major >= 9:
        return 8
    return 2


cdef inline unsigned int _fallback_if_zero(unsigned int value, unsigned int fallback) noexcept:
    if value != 0:
        return value
    return fallback


cdef class SMResource:
    """Represent an SM (streaming multiprocessor) resource partition.

    Instances are returned by :obj:`DeviceResources.sm` or
    :meth:`SMResource.split` and cannot be instantiated directly.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SMResource cannot be instantiated directly. "
            "Use dev.resources.sm or SMResource.split()."
        )

    @staticmethod
    cdef SMResource _from_dev_resource(cydriver.CUdevResource res, int device_id):
        cdef SMResource self = SMResource.__new__(SMResource)
        self._resource = res
        self._sm_count = res.sm.smCount
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            self._min_partition_size = res.sm.minSmPartitionSize
            self._coscheduled_alignment = res.sm.smCoscheduledAlignment
            self._flags = res.sm.flags
        ELSE:
            self._min_partition_size = _sm_resource_granularity(device_id)
            self._coscheduled_alignment = self._min_partition_size
            self._flags = 0
        self._is_usable = True
        return self

    @staticmethod
    cdef SMResource _from_split_resource(cydriver.CUdevResource res, SMResource parent, bint is_usable):
        cdef SMResource self = SMResource.__new__(SMResource)
        self._resource = res
        self._sm_count = res.sm.smCount
        IF CUDA_CORE_BUILD_MAJOR >= 13:
            self._min_partition_size = _fallback_if_zero(
                res.sm.minSmPartitionSize,
                parent._min_partition_size,
            )
            self._coscheduled_alignment = _fallback_if_zero(
                res.sm.smCoscheduledAlignment,
                parent._coscheduled_alignment,
            )
            self._flags = res.sm.flags
        ELSE:
            self._min_partition_size = parent._min_partition_size
            self._coscheduled_alignment = parent._coscheduled_alignment
            self._flags = parent._flags
        self._is_usable = is_usable
        return self

    @property
    def handle(self) -> int:
        """Return the address of the underlying ``CUdevResource`` struct."""
        return <intptr_t>(&self._resource)

    @property
    def sm_count(self) -> int:
        """Total SMs available in this resource."""
        return self._sm_count

    @property
    def min_partition_size(self) -> int:
        """Minimum SM count required to create a partition."""
        return self._min_partition_size

    @property
    def coscheduled_alignment(self) -> int:
        """Number of SMs guaranteed to be co-scheduled."""
        return self._coscheduled_alignment

    @property
    def flags(self) -> int:
        """Raw flags from the underlying SM resource."""
        return self._flags

    def split(self, options not None, *, bint dry_run=False):
        """Split this SM resource into groups and a remainder.

        Parameters
        ----------
        options : :obj:`SMResourceOptions`
            Split configuration (count, co-scheduling constraints).
        dry_run : bool, optional
            If ``True``, return filled-in metadata without creating
            usable resource objects. (Default to ``False``)

        Returns
        -------
        tuple[list[:obj:`SMResource`], :obj:`SMResource`]
            ``(groups, remainder)`` where each group holds a disjoint
            SM partition and *remainder* holds any unassigned SMs.
        """
        cdef SMResourceOptions opts = check_or_create_options(
            SMResourceOptions, options, "SM resource options"
        )
        _resolve_group_count(opts)
        _check_green_ctx_support()
        if _can_use_structured_sm_split():
            return _split_with_general_api(self, opts, dry_run)
        # SplitByCount requires the same 12.4+ as green ctx support (already checked above)
        return _split_with_count_api(self, opts, dry_run)


cdef class WorkqueueResource:
    """Represent a workqueue resource for a device or green context.

    Merges ``CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG`` and
    ``CU_DEV_RESOURCE_TYPE_WORKQUEUE`` under one user-facing type.
    Instances are returned by :obj:`DeviceResources.workqueue` and
    cannot be instantiated directly.
    """

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
        return <intptr_t>(&self._wq_config_resource)

    def configure(self, options not None):
        """Configure the workqueue resource in place.

        Parameters
        ----------
        options : :obj:`WorkqueueResourceOptions`
            Configuration options (sharing scope, etc.).
        """
        cdef WorkqueueResourceOptions opts = check_or_create_options(
            WorkqueueResourceOptions, options, "Workqueue resource options"
        )
        _check_green_ctx_support()
        _check_workqueue_support()
        if opts.sharing_scope is None:
            return None

        IF CUDA_CORE_BUILD_MAJOR >= 13:
            if opts.sharing_scope == "device_ctx":
                self._wq_config_resource.wqConfig.sharingScope = (
                    cydriver.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_DEVICE_CTX
                )
            elif opts.sharing_scope == "green_ctx_balanced":
                self._wq_config_resource.wqConfig.sharingScope = (
                    cydriver.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
                )
            else:
                raise ValueError(
                    f"Unknown sharing_scope: {opts.sharing_scope!r}. "
                    "Expected 'device_ctx' or 'green_ctx_balanced'."
                )
        ELSE:
            raise RuntimeError(
                "WorkqueueResource requires cuda.core to be built with CUDA 13.x bindings"
            )


cdef class DeviceResources:
    """Namespace for hardware resource queries.

    When obtained via :obj:`Device.resources`, queries return full device
    resources. When obtained via :obj:`Context.resources` or
    :obj:`Stream.resources`, queries return the resources provisioned for
    that context.

    This class cannot be instantiated directly.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DeviceResources cannot be instantiated directly. "
            "Use dev.resources or ctx.resources."
        )

    @staticmethod
    cdef DeviceResources _init(int device_id):
        cdef DeviceResources self = DeviceResources.__new__(DeviceResources)
        self._device_id = device_id
        # _h_context is default empty — queries use cuDeviceGetDevResource
        return self

    @staticmethod
    cdef DeviceResources _init_from_ctx(ContextHandle h_context, int device_id):
        cdef DeviceResources self = DeviceResources.__new__(DeviceResources)
        self._device_id = device_id
        self._h_context = h_context
        return self

    cdef inline int _query_sm(self, cydriver.CUdevResource* res) except?-1 nogil:
        """Query SM resource from either device or context."""
        cdef GreenCtxHandle h_green
        if self._h_context:
            h_green = get_context_green_ctx(self._h_context)
            if h_green:
                HANDLE_RETURN(cydriver.cuGreenCtxGetDevResource(
                    as_cu(h_green), res,
                    cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                ))
            else:
                HANDLE_RETURN(cydriver.cuCtxGetDevResource(
                    as_cu(self._h_context), res,
                    cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                ))
        else:
            HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                <cydriver.CUdevice>(self._device_id), res,
                cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            ))
        return 0

    @property
    def sm(self) -> SMResource:
        """Return the :obj:`SMResource` for this device or context."""
        _check_green_ctx_support()
        cdef cydriver.CUdevResource res
        with nogil:
            self._query_sm(&res)
        return SMResource._from_dev_resource(res, self._device_id)

    @property
    def workqueue(self) -> WorkqueueResource:
        """Return the :obj:`WorkqueueResource` for this device or context."""
        _check_green_ctx_support()
        _check_workqueue_support()
        cdef cydriver.CUdevResource _wq_config
        cdef cydriver.CUdevResource _wq

        IF CUDA_CORE_BUILD_MAJOR >= 13:
            cdef GreenCtxHandle h_green
            if self._h_context:
                h_green = get_context_green_ctx(self._h_context)
                if h_green:
                    # Green context query
                    with nogil:
                        HANDLE_RETURN(cydriver.cuGreenCtxGetDevResource(
                            as_cu(h_green),
                            &_wq_config,
                            cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
                        ))
                        HANDLE_RETURN(cydriver.cuGreenCtxGetDevResource(
                            as_cu(h_green),
                            &_wq,
                            cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE,
                        ))
                else:
                    # Primary context query
                    with nogil:
                        HANDLE_RETURN(cydriver.cuCtxGetDevResource(
                            as_cu(self._h_context),
                            &_wq_config,
                            cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
                        ))
                        HANDLE_RETURN(cydriver.cuCtxGetDevResource(
                            as_cu(self._h_context),
                            &_wq,
                            cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE,
                        ))
            else:
                # Device-level query
                with nogil:
                    HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                        <cydriver.CUdevice>(self._device_id),
                        &_wq_config,
                        cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,
                    ))
                    HANDLE_RETURN(cydriver.cuDeviceGetDevResource(
                        <cydriver.CUdevice>(self._device_id),
                        &_wq,
                        cydriver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE,
                    ))
            return WorkqueueResource._from_dev_resources(_wq_config, _wq)
        ELSE:
            raise RuntimeError(
                "WorkqueueResource requires cuda.core to be built with CUDA 13.x bindings"
            )
