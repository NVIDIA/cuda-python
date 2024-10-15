# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Optional

from cuda import cuda
from cuda.core.experimental._utils import check_or_create_options
from cuda.core.experimental._utils import CUDAError
from cuda.core.experimental._utils import handle_return


@dataclass
class EventOptions:
    enable_timing: Optional[bool] = False
    busy_waited_sync: Optional[bool] = False
    support_ipc: Optional[bool] = False


class Event:

    __slots__ = ("_handle", "_timing_disabled", "_busy_waited")

    def __init__(self):
        # minimal requirements for the destructor
        self._handle = None
        raise NotImplementedError(
            "directly creating an Event object can be ambiguous. Please call "
            "call Stream.record().")

    @staticmethod
    def _init(options: Optional[EventOptions]=None):
        self = Event.__new__(Event)
        # minimal requirements for the destructor
        self._handle = None

        options = check_or_create_options(EventOptions, options, "Event options")
        flags = 0x0
        self._timing_disabled = False
        self._busy_waited = False
        if not options.enable_timing:
            flags |= cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING
            self._timing_disabled = True
        if options.busy_waited_sync:
            flags |= cuda.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            self._busy_waited = True
        if options.support_ipc:
            raise NotImplementedError("TODO")
        self._handle = handle_return(cuda.cuEventCreate(flags))
        return self

    def __del__(self):
        self.close()

    def close(self):
        # Destroy the event.
        if self._handle:
            handle_return(cuda.cuEventDestroy(self._handle))
            self._handle = None

    @property
    def is_timing_disabled(self) -> bool:
        # Check if this instance can be used for the timing purpose.
        return self._timing_disabled

    @property
    def is_sync_busy_waited(self) -> bool:
        # Check if the event synchronization would keep the CPU busy-waiting.
        return self._busy_waited

    @property
    def is_ipc_supported(self) -> bool:
        # Check if this instance can be used for IPC.
        raise NotImplementedError("TODO")

    def sync(self):
        # Sync over the event.
        handle_return(cuda.cuEventSynchronize(self._handle))

    @property
    def is_done(self) -> bool:
        # Return True if all captured works have been completed,
        # otherwise False.
        result, = cuda.cuEventQuery(self._handle)
        if result == cuda.CUresult.CUDA_SUCCESS:
            return True
        elif result == cuda.CUresult.CUDA_ERROR_NOT_READY:
            return False
        else:
            raise CUDAError(f"unexpected error: {result}")

    @property
    def handle(self) -> int:
        return int(self._handle)
