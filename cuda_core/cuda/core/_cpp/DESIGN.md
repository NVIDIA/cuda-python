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
├── _resource_handles.pyx    # Cython module (compiles resource_handles.cpp)
├── _resource_handles.pxd    # Cython declarations for consumer modules
└── _cpp/
    ├── resource_handles.hpp       # C++ API declarations
    └── resource_handles.cpp       # C++ implementation
```

### Build Implications

The `_cpp/` subdirectory contains C++ source files that are compiled into the
`_resource_handles` extension module. Other Cython modules in cuda.core do **not**
link against this code directly—they `cimport` functions from
`_resource_handles.pxd`, and calls go through `_resource_handles.so` at runtime.

## Cross-Module Function Sharing

**Problem**: Cython extension modules compile independently. If multiple modules
(`_memory.pyx`, `_ipc.pyx`, etc.) each linked `resource_handles.cpp`, they would
each have their own copies of:

- Static driver function pointers
- Thread-local error state
- Other static data, including global caches

**Solution**: Only `_resource_handles.so` links the C++ code. The `.pyx` file
uses `cdef extern from` to declare C++ functions with Cython-accessible names:

```cython
# In _resource_handles.pyx
cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    StreamHandle create_stream_handle "cuda_core::create_stream_handle" (
        ContextHandle h_ctx, unsigned int flags, int priority) nogil
    # ... other functions
```

The `.pxd` file declares these same functions so other modules can `cimport` them:

```cython
# In _resource_handles.pxd
cdef StreamHandle create_stream_handle(
    ContextHandle h_ctx, unsigned int flags, int priority) noexcept nogil
```

The `cdef extern from` declaration in the `.pyx` satisfies the `.pxd` declaration
directly—no wrapper functions are needed. When consumer modules `cimport` these
functions, Cython generates calls through `_resource_handles.so` at runtime.
This ensures all static and thread-local state lives in a single shared library,
avoiding the duplicate state problem.

## CUDA Driver API Capsule (`_CUDA_DRIVER_API_V1`)

**Problem**: cuda.core cannot directly call CUDA driver functions because:

1. We don't want to link against `libcuda.so` at build time.
2. The driver symbols must be resolved dynamically through cuda-bindings.

**Solution**: `_resource_handles.pyx` creates a capsule containing CUDA driver
function pointers obtained from cuda-bindings:

```cpp
struct CudaDriverApiV1 {
    uint32_t abi_version;
    uint32_t struct_size;

    uintptr_t cuDevicePrimaryCtxRetain;
    uintptr_t cuDevicePrimaryCtxRelease;
    uintptr_t cuStreamCreateWithPriority;
    uintptr_t cuStreamDestroy;
    // ... etc
};
```

The C++ code retrieves this capsule once (via `load_driver_api()`) and caches the
function pointers for subsequent use.

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

### GIL Management

Handle destructors may run from any thread. The implementation includes RAII guards
(`GILReleaseGuard`, `GILAcquireGuard`) that:

- Release the GIL before calling CUDA APIs (for parallelism)
- Handle Python finalization gracefully (avoid GIL operations during shutdown)
- Ensure Python object manipulation happens with GIL held

The handle API functions are safe to call with or without the GIL held. They
will release the GIL (if necessary) before calling CUDA driver API functions.

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

## Usage from Cython

```cython
from cuda.core._resource_handles cimport (
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
4. **Uses a capsule** to resolve CUDA driver symbols dynamically through cuda-bindings.
5. **Provides overloaded accessors** (`as_cu`, `as_intptr`, `as_py`) since handles cannot
   have attributes without unnecessary Python object wrappers.

This architecture ensures CUDA resources are managed correctly regardless of Python
garbage collection timing, interpreter shutdown, or cross-language usage patterns.
