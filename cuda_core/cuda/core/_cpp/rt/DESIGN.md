# Resource Handles Design

This document describes the resource handle abstraction in cuda.core, which provides
robust lifetime management for CUDA resources.

## Overview

The cuda-core Python library provides a high-level interface to CUDA resources such as
Context, Device, Stream, and Event. These objects correspond to resources managed by
the CUDA Driver API, each having explicit creation and destruction routines. Several
of these CUDA resources also participate in non-trivial ownership hierarchies (e.g.,
a stream belongs to a context), and releasing them may require additional arguments
or other resources (e.g., a device pointer freed through a specific stream).

### Goals

The goal of the handle abstraction is to provide a robust, explicit, and Python-agnostic
layer for ownership and lifetime management of CUDA resources. The intent is to use
handles as the backbone of the cuda-core resource hierarchy, enabling cuda-core Python
objects to manipulate handles rather than work directly with raw CUDA resources.

While Python-facing objects expose convenient APIs and additional behaviors, the handle
layer isolates all concerns related to resource lifetime. By cleanly separating these
responsibilities, we achieve:

- **Clearer architecture** with minimal cross-layer coupling
- **Safe transfer of resource ownership** between Python and other domains, including C++
- **Ability to preserve resource validity** independent of Python
- **Well-specified semantics** for immutability, ownership, and reachability
- **Simplified reasoning about resource lifetimes**, especially with nested or dependent resources

### Handle Semantics

Resource handles provide **referentially transparent** wrappers around CUDA resources:

- **No rebinding**: A handle always refers to the same resource.
- **No invalidation**: If a handle exists, its resource is valid.
- **Structural dependencies**: If resource A depends on resource B, A's handle
  embeds B's handle, automatically extending B's lifetime.

This eliminates global lifetime analysis. Correctness is enforced structurally—if you
have a handle, you have a valid resource.

Handles to child graphs and their nodes are a narrow exception. These handles
are borrowed references to CUDA resources owned by a mutable parent graph.
Removing or replacing the owning graph node destroys those resources, forcing
cuda-core to invalidate any outstanding handles. See
[Graph Node Attachments](GRAPH_ATTACHMENTS.md).

## Handle Types

All handles are `std::shared_ptr` aliases that expose only the raw CUDA resource:

```cpp
using ContextHandle = std::shared_ptr<const CUcontext>;
using StreamHandle = std::shared_ptr<const CUstream>;
using EventHandle = std::shared_ptr<const CUevent>;
using MemoryPoolHandle = std::shared_ptr<const CUmemoryPool>;
using DevicePtrHandle = std::shared_ptr<const CUdeviceptr>;
```

Internally, handles use **shared pointer aliasing**: the actual managed object is a
"box" containing the resource, its dependencies, and any state needed for destruction.
The public handle points only to the raw resource field, keeping the API minimal.

### Why shared_ptr?

- **Automatic reference counting**: Resources are released when the last reference
  disappears.
- **Cross-language stability**: Works across Python/C++ boundaries without relying
  on Python's garbage collector.
- **Interpreter independence**: Resources remain valid even during Python shutdown.
- **Type-erased deleters**: Destruction logic is captured at creation time, supporting
  diverse lifetime strategies.

## Accessing Handle Values

Handles can be accessed in three ways via overloaded helper functions:

| Function | Returns | Use Case | Notes
|----------|---------|----------|-------|
| `as_cu(h)` | Raw CUDA type (e.g., `CUstream`) | Passing to CUDA APIs | An attribute of `cuda.bindings.cydriver` |
| `as_intptr(h)` | `intptr_t` | Python interop, foreign code | |
| `as_py(h)` | Python wrapper object | Returning to Python callers | An attribute of `cuda.bindings.driver`

These overloads exist because `std::shared_ptr` cannot have additional attributes.
Wrapping handles in Python objects would be superfluous overhead for internal use,
so we provide these helpers instead.

Example usage from Cython:

```cython
# Get raw handle for CUDA API calls
cdef CUstream raw_stream = as_cu(h_stream)  # cuda.bindings.cydriver.CUstream

# Get as integer for other use cases
return hash(as_intptr(h_stream))

# Get Python wrapper for returning to user
return as_py(h_stream)  # cuda.bindings.driver.CUstream
```

## Code Structure

### Directory Layout

```
cuda/core/
├── _rt.pyx                  # Cython module (compiles everything under _cpp/rt/)
├── _rt.pxd                  # Cython declarations for consumer modules
└── _cpp/rt/
    ├── rt.hpp               # Module umbrella, named only by _rt.pyx
    ├── handles.hpp          # Consumer umbrella, named only by _rt.pxd
    ├── types.hpp            # Handle aliases, tagged values, inline accessors
    ├── api.hpp              # Prototypes of the handle factories and accessors
    ├── driver_api.hpp/.cpp  # Driver function-pointer table, version-gated shims
    ├── error.hpp/.cpp       # Thread-local error state, non-propagating reporting
    ├── py.hpp               # The one header that includes <Python.h>
    ├── context_scope.hpp    # Scoped-context helpers (namespace detail)
    ├── internal.hpp         # Registry, cleanup wrappers, deferred-cleanup item (namespace detail)
    ├── py_report.cpp, py_deferred_cleanup.cpp        # Python-coupled bodies
    └── context.cpp, stream.cpp, event.cpp, memory.cpp, program.cpp,
        graph.cpp, graph_exec.cpp, texture.cpp        # One resource family per file
```

### Build Implications

Every `.cpp` under `_cpp/rt/` is compiled into the one `_rt` extension module.
Other Cython modules in cuda.core do **not** link against this code
directly—they `cimport` functions from `_rt.pxd`, and calls go through
`_rt.so` at runtime.

## Cross-Module Function Sharing

**Problem**: Cython extension modules compile independently. If multiple modules
(`_memory.pyx`, `_ipc.pyx`, etc.) each linked the C++ under `_cpp/rt/`, they would
each have their own copies of:

- Static driver function pointers
- Thread-local error state
- Other static data, including global caches

**Solution**: Only `_rt.so` links the C++ code. The `.pyx` file
uses `cdef extern from` to declare C++ functions with Cython-accessible names:

```cython
# In _rt.pyx
cdef extern from "_cpp/rt/rt.hpp" namespace "cuda_core::rt":
    StreamHandle create_stream_handle "cuda_core::rt::create_stream_handle" (
        ContextHandle h_ctx, unsigned int flags, int priority) nogil
    # ... other functions
```

The `.pxd` file declares these same functions so other modules can `cimport` them:

```cython
# In _rt.pxd
cdef StreamHandle create_stream_handle(
    ContextHandle h_ctx, unsigned int flags, int priority) noexcept nogil
```

The `cdef extern from` declaration in the `.pyx` satisfies the `.pxd` declaration
directly—no wrapper functions are needed. When consumer modules `cimport` these
functions, Cython generates calls through `_rt.so` at runtime.
This ensures all static and thread-local state lives in a single shared library,
avoiding the duplicate state problem.

## CUDA driver function pointers from cuda-bindings

**Problem**: cuda.core cannot link against `libcuda.so` at build time, and it
must not load the driver itself. cuda-bindings owns driver loading and symbol
resolution. It resolves symbols with `cuGetProcAddress`, which also selects the
ABI variant and the per-thread-default-stream variant.

Until #2783, the C++ called cuda-bindings' *Cython wrappers*, extracted from
`cydriver.__pyx_capi__`. When the driver lacked a function, a wrapper raised a
Python exception that C++ never saw and returned a sentinel `CUresult`. The
exception surfaced later as `SystemError`. A wrapper could also be absent when
the installed cuda-bindings was older than the build, so some pointers were
optional and probed for null.

**Solution**: `driver_api.hpp` lists every driver function that the C++ calls,
with the CUDA version cuda-bindings requests it at:

```cpp
#define CUDA_CORE_DRIVER_FUNCTIONS(X) \
    X(cuStreamCreateWithPriority, 5050) \
    X(cuGreenCtxStreamCreate, 12050) \
    ...
```

The list declares the `p_cuXxx` pointers and builds a table of
`{key, name, slot, introduced}` entries. The key is `"__"` plus the symbol that
`cuda.h` maps the name to (`cuStreamDestroy` -> `"__cuStreamDestroy_v2"`). That
is how cuda-bindings names the slot in
`cuda.bindings._internal.driver._inspect_function_pointers()`. Because the key
follows the header's macros, the build requires the header's major.minor to
equal cuda-bindings' (see "Build-time version guards").

`ensure_fn_table()` (`py_driver_fns.cpp`) fills the table lazily, the first
time a `DRIVER_CALL(name, args...)` finds its pointer null. `import cuda.core`
therefore never touches the driver. The fill acquires the GIL, calls
`_inspect_function_pointers()`, and copies every entry's address into its
`p_` pointer under a mutex that is never held across a Python call. It then
checks that every function introduced at or before the CUDA major series'
first release is present. A null one means that the driver is older than the
series, and the fill fails with that message.

After the fill, a pointer is either the driver's entry point or null. A null
pointer means that the installed driver does not provide that newer function.
Cython gates those functions on the driver version (`cy_driver_version()`). The
C++ never gates them with a null check.

A `DRIVER_CALL` that still finds null after the fill is a gate bug or a failed
fill. It reports through `report_message()` and returns
`CUDA_ERROR_NOT_INITIALIZED` from a trampoline of the right signature. It never
dereferences null and never throws, so it is safe in `noexcept` deleters. The
`pw_` wrappers go through the same path.

Some owning handle constructors do not call the driver, but their deleter does
(`create_graph_handle`, ...). Those constructors call `ensure_fn_table()` so
that the fill never happens in a deleter. NVRTC, NVVM and nvJitLink have one
table each. `create_*_handle` fills it after the library has loaded.

Two rules follow:

1. Do not make a `DRIVER_CALL` while the caller holds a C++ lock, because the
   fill acquires the GIL. Resolve the table before the lock
   (`ensure_fn_table(FnTable::driver)`) and use the raw pointer inside it,
   marked `// raw:` (see `deviceptr_import_ipc`).
2. If the Cython layer gates a driver function, gate it at the version
   cuda-bindings requests it at (the number in the table), not at the version
   the driver first shipped it.

`tests/test_rt_layout.py` checks the table against cuda-bindings' loader. It
also checks that no raw `p_` call exists outside the machinery and the marked
lines.

## Build-time version guards

cuda.core supports one build configuration per CUDA major series. The `cuda.h`
it compiles against has the same major.minor as the cuda-bindings it is built
with, and that cuda-bindings is at or above the series' floor
(`cuda/core/_bindings_floor.py`). `build_hooks.py` enforces both before
compilation and defines `CUDA_CORE_BUILD_MAJOR` and
`CUDA_CORE_MIN_CUDA_VERSION` for the C++ compiler. `versions.hpp`, the first
include of the tree, re-checks `cuda.h` against them with `#error`.

The C++ branches on `CUDA_CORE_BUILD_MAJOR` only, and only where the two major
series differ. Minor-version fences (`#if CUDA_VERSION >= 130x0`) are not
allowed. They compiled features out of source builds against an older header.
The run-time checks looked at cuda-bindings and the driver, so they never
noticed (https://github.com/NVIDIA/cuda-python/issues/2783). The driver-version
gates in Cython, never the C++ layer, decide whether the *driver* provides a
function. `tests/test_rt_layout.py` enforces that `versions.hpp` is the only
file under `_cpp/` that names `CUDA_VERSION`.

## Key Implementation Details

### Structural Dependencies

When a resource depends on another, its handle embeds the dependency:

```cpp
struct StreamBox {
    CUstream resource;
    ContextHandle h_context;  // Keeps context alive
};
```

The shared pointer's custom deleter captures any additional state needed for
destruction. This ensures resources are always destroyed in the correct order.

Graph node parameters require a specialized ownership model built on
`OpaqueHandle`; see [Graph Node Attachments](GRAPH_ATTACHMENTS.md).

### GIL Management

Handle destructors may run from any thread. The implementation includes RAII guards
(`GILReleaseGuard`, `GILAcquireGuard`) that:

- Release the GIL before calling CUDA APIs (for parallelism)
- Handle Python finalization gracefully (avoid GIL operations during shutdown)
- Ensure Python object manipulation happens with GIL held

The handle API functions work with or without the GIL held. Cython calls most
of them from `with nogil` blocks and some with the GIL held. They never require
the GIL and never acquire it while they hold a C++ lock. If necessary, they
release the GIL before they call CUDA driver API functions. The only places
that acquire the GIL are the reporting paths (`pw_*`, `report_*`) and the
one-time function-table fill (`ensure_fn_table()`). Neither may run while a
C++ lock is held.

**The GIL is the outermost lock.** Code that holds a C++ lock (a registry's
mutex, `ipc_import_mutex`, any `std::mutex`) must not acquire or reacquire the
GIL while the lock is held: no `report_*` or `pw_*` calls, no
`GILAcquireGuard`, and no `GILReleaseGuard` whose destructor runs inside the
locked region. Code that needs a C++ lock and may run with the GIL held
releases the GIL first (`GILReleaseGuard` before `lock_guard`). Otherwise a
thread blocked on the lock while holding the GIL deadlocks with the lock holder
waiting for the GIL (#2840). Collect statuses under the lock and report after it
is released, as `deviceptr_import_ipc` does: `cleanup_in_context` takes an
`after_cleanup` hook that runs once the cleanup is done and before anything that
may run user code, and the deleter passes one that unlocks its
`std::unique_lock`. The registries store `weak_ptr`s,
so erasing an entry under a registry lock never runs a deleter.

### Static Initialization and Deadlock Hazards

When writing C++ code that interacts with Python, a subtle deadlock can occur
when combining C++ static variable initialization with Python's GIL. This is
known as the "double locking" or "latent deadlock" problem.

**The hazard**: C++11 guarantees thread-safe static initialization using an
implicit guard mutex. If a static initializer calls Python C API functions
(like `PyImport_ImportModule`), and those functions release and reacquire
the GIL internally, a deadlock can occur:

1. Thread T1 holds GIL, enters static initialization (locks guard mutex)
2. T1's initializer releases GIL (may occur via any Python API call)
3. Thread T2 acquires GIL, tries to enter same static initialization
4. T2 blocks on guard mutex (held by T1)
5. T1 tries to reacquire GIL (held by T2)
6. **Deadlock**: T1 waits for GIL, T2 waits for guard mutex

This is documented in detail by the pybind11 project:
https://github.com/pybind/pybind11/blob/master/docs/advanced/deadlock.md

**General rule**: When holding the GIL, avoid acquiring any C++ lock (including
implicit ones like static initialization guards) if the critical section may
call Python C API functions. Many Python API calls can internally release and
reacquire the GIL, creating the second lock ordering that risks deadlock.

### Error Handling

Handle API functions do not raise Python exceptions. Instead, they return an empty
handle (null `shared_ptr`) on failure and store the error code in thread-local state.
Callers should check for failure and retrieve the error using `get_last_error()`:

```cython
cdef StreamHandle h = create_stream_handle(h_ctx, flags, priority)
if not h:
    # Handle creation failed - get the CUDA error code
    cdef CUresult err = get_last_error()
    # ... handle error (e.g., raise Python exception)
```

This design allows handle functions to be called from `nogil` blocks without requiring
GIL acquisition for exception handling on the success path. The error state is
thread-local, so concurrent calls from different threads do not interfere.

Related functions:
- `get_last_error()`: Returns and clears the most recent error
- `peek_last_error()`: Returns the error without clearing it
- `clear_last_error()`: Clears the error state

The C++ layer never raises Python exceptions: it runs `nogil` and `noexcept`,
and is called from deleters, CUDA callbacks and GIL-released code where raising
is impossible. Status is turned into `CUDAError` in one place, `HANDLE_RETURN`
in the Cython layer. Which status convention a function uses is decided by its
return value. Factories return the handle, so their status goes to thread-local
`err` and is read with `get_last_error()`. Functions that do not produce a
handle (`context_synchronize`, `context_get_device`, `graph_node_set_params`,
the `graph_*_attachment` family, `deviceptr_alloc_raw`) return the `CUresult`
directly and deliver results through out-parameters, mirroring the driver API;
their callers `HANDLE_RETURN` the value. The two conventions never mix.

### Context-scoped operations

Operations that must run in a specific context use `invoke_in_context` /
`invoke_in_context_or_undo` (propagating paths) and `cleanup_in_context`
(deleters). They switch the current context, run the operation, and restore the
caller's context. `cleanup_in_context` emits its reports only after that
restoration, so the user code a `CUDAWarning` runs (filters, `showwarning`)
observes the caller's context. When restoration fails after the operation
succeeded, the creation is undone and the restoration status is returned. When both fail, the
operation status is returned. Either way the helper records a thread-local
detail keyed to the returned status (`take_last_error_detail(status)`) that
`_check_driver_error` attaches to the raised `CUDAError` as a PEP 678 note
(appended to the message on Python 3.10), so the user learns that the caller's
context was not restored, which context is current and, for a double failure,
why restoration failed. Keying the detail to its status narrows, but does not
remove, misattribution: a caller that drops the status (an empty handle raised
as a generic error) leaves the detail behind, and a later error on the same
thread with the same status code picks it up. `enter_context` clears stale
detail at the next context-scoped operation. Issue #2760 removes this
thread-local state in favor of explicit status returns. Tests inject restoration failures with
`set_context_restore_fault_for_testing()`.

### Reporting from non-propagating paths

Deleters and CUDA callbacks cannot raise. They report through
`report_cuda_error()` / `report_message()` (the `pw_*` wrappers decorate
destroy calls with it and name the resource handle in the message, so Python's
warning registry does not collapse independent failures of one call), which emit a `cuda.core.CUDAWarning` through
the Python warnings machinery when the interpreter is usable, deliver an
escalated warning as an unraisable exception, and fall back to stderr when the
GIL cannot be taken (for example during finalization). `CUDA_ERROR_DEINITIALIZED`
is never reported because it means the driver is shutting down. No status is
discarded silently anywhere in this layer, and nothing in this layer may
terminate the process; see `docs/source/error_handling.rst` and the "Failure handling"
section of `AGENTS.md` for the policy.

A rollback that fails inside a Cython `except` block is not a non-propagating
path: `attach_rollback_failure()` attaches it as a note to the exception being
handled (`PyErr_GetHandledException`, Python 3.11+) and falls back to a report
only when there is no such exception or notes are unavailable.

### Which channel to use

Pick the channel by where the failure happens. Every failure goes through
exactly one of these; none is ever dropped.

| Where you are | Use | Result |
|---|---|---|
| Cython, on a path that can raise | `HANDLE_RETURN(status)` | Raises `CUDAError`. A restoration detail recorded by the C++ helper becomes a note on the exception. |
| Cython, after a handle constructor returned an empty handle | `HANDLE_RETURN(get_last_error())`, immediately | Same. Transitional: #2760 makes constructors return the status instead. |
| C++, a helper that runs an operation in another context | Return the `CUresult`; `exit_context` records the restoration detail | Cython raises it. Transitional: #2760 returns the restoration status as a second out-parameter. |
| Cython, inside an `except` block whose rollback failed | `attach_rollback_failure(op, status, detail)` | Adds a note to the exception being handled. Reports instead if nothing is being handled or notes do not exist (Python 3.10). |
| C++, a deleter or deferred cleanup | A `pw_*` wrapper, or `report_cuda_error()` / `report_message()` | Emits `CUDAWarning`. Never raises. |
| Cython or Python, a `__dealloc__` or destructor-path callback | `warnings.warn(msg, CUDAWarning, stacklevel=2)` | Same. |
| A CUDA callback thread | Nothing that needs the GIL. Hand the work to the deferred-cleanup queue with `Py_AddPendingCall` | CUDA forbids driver calls there, and acquiring the GIL there can deadlock with a GIL holder blocked in a driver call. GIL-free C API that only schedules work is fine. |

### `p_` versus `pw_`

A `DRIVER_CALL` (a `p_` function pointer) calls the driver and nothing else,
once the table is filled. Its `pw_` twin calls the driver and, if the call
fails, acquires the GIL and runs Python: the warning filters, `showwarning`, or
`sys.unraisablehook`. Any of those can be user code,
and user code can call back into cuda.core. This is the one place where the
handle layer runs code it does not control, and it is the entry point through
which a thread holding a C++ lock can deadlock (see "GIL Management").

Python exceptions raised by that code never become C++ exceptions: the C API
reports them as return codes, and `report_message` hands them to
`sys.unraisablehook`. Nothing on the report path may allocate or throw, since a
deleter is `noexcept`.

So: use `pw_` only in deleters and cleanup paths that hold no C++ lock and have
finished updating the layer's own state. Where a lock must stay held, call
`p_`, keep the status, and report after the lock is released, as
`deviceptr_import_ipc` does. CUDA callback threads need no extra rule for
`pw_`: the driver call is forbidden there, so the wrapper is too. The general
rule for those threads is no GIL and no Python objects; GIL-free scheduling
calls such as `Py_AddPendingCall` are how work leaves them.

## Usage from Cython

```cython
from cuda.core._rt cimport (
    StreamHandle,
    create_stream_handle,
    as_cu,
    as_intptr,
    as_py,
    get_last_error,
)

# Create a stream
cdef StreamHandle h_stream = create_stream_handle(h_ctx, flags, priority)
if not h_stream:
    HANDLE_RETURN(get_last_error())

# Use in CUDA API
cuStreamSynchronize(as_cu(h_stream))

# Return to Python
return as_py(h_stream)
```

## Summary

The resource handle design:

1. **Separates resource management** into its own layer, independent of Python objects.
2. **Encodes lifetimes structurally** via embedded handle dependencies.
3. **Uses Cython's `cimport` mechanism** to share C++ code across modules without
   duplicate static/thread-local state.
4. **Resolves CUDA driver symbols** through the driver entry points that cuda-bindings
   resolves (`_inspect_function_pointers()`), filled lazily by `ensure_fn_table()` on first use.
5. **Provides overloaded accessors** (`as_cu`, `as_intptr`, `as_py`) since handles cannot
   have attributes without unnecessary Python object wrappers.

This architecture ensures CUDA resources are managed correctly regardless of Python
garbage collection timing, interpreter shutdown, or cross-language usage patterns.
