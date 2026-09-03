This file describes `cuda_core`, the high-level Pythonic CUDA subpackage in the
`cuda-python` monorepo.

## Scope and principles

- **Role**: provide higher-level CUDA abstractions (`Device`, `Stream`,
  `Program`, `Linker`, memory resources, graphs) on top of `cuda.bindings`.
- **API intent**: keep interfaces Pythonic while preserving explicit CUDA
  behavior and error visibility.
- **Compatibility**: changes should remain compatible with supported
  `cuda.bindings` major versions (12.x and 13.x).

## Package architecture

- **Main package**: `cuda/core/` contains most Cython modules (`*.pyx`, `*.pxd`)
  implementing runtime behaviors and public objects.
- **Subsystems**:
  - memory/resource stack: `cuda/core/_memory/`
  - system-level APIs: `cuda/core/system/`
  - compile/link path: `_program.pyx`, `_linker.pyx`, `_module.pyx`
  - execution path: `_launcher.pyx`, `_launch_config.pyx`, `_stream.pyx`
- **C++ helpers**: module-specific C++ implementations live under
  `cuda/core/_cpp/`.
- **Build backend**: `build_hooks.py` handles Cython extension setup and build
  dependency wiring.

## Build and version coupling

- `build_hooks.py` determines CUDA major version from `CUDA_CORE_BUILD_MAJOR`
  or CUDA headers (`CUDA_HOME`/`CUDA_PATH`) and uses it for build decisions.
- Source builds require CUDA headers available through `CUDA_HOME` or
  `CUDA_PATH`.
- `cuda_core` expects `cuda.bindings` to be present and version-compatible.

## Testing expectations

- **Primary tests**: `pytest tests/`
- **Cython tests**:
  - build: `tests/cython/build_tests.sh` (or platform equivalent)
  - run: `pytest tests/cython/`
- **Examples**: validate affected examples in `examples/` when changing user
  workflows or public APIs.

## Runtime/build environment notes

- Runtime env vars commonly relevant:
  - `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`
  - `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`
- Build env vars commonly relevant:
  - `CUDA_HOME` / `CUDA_PATH`
  - `CUDA_CORE_BUILD_MAJOR`
  - `CUDA_PYTHON_PARALLEL_LEVEL`
  - `CUDA_PYTHON_COVERAGE`

## Editing guidance

- Keep user-facing behaviors coherent with docs and examples, especially around
  stream semantics, memory ownership, and compile/link flows.
- Reuse existing shared utilities in `cuda/core/_utils/` before adding new
  helpers.
- When changing Cython signatures or cimports, verify related `.pxd` and
  call-site consistency.
- Prefer explicit error propagation over silent fallback paths.
- If you change public behavior, update tests and docs under `docs/source/`.

## Concurrency and free-threading

`cuda.core` ships free-threaded (no-GIL) wheels and builds with
`freethreading_compatible=True`. The user-facing policy lives in
`docs/source/concurrency.rst`; the invariants below are for contributors. Reviewers
and agents should flag violations.

- **Reads are safe; mutation is the boundary**: concurrent reads of an object are
  supported, but concurrent mutation of the same public object (e.g., building one
  graph from two threads, or `close()` racing another call) is the caller's
  responsibility -- do not add locks to make it safe. Prefer immutable designs to
  keep the mutable, thread-unsafe surface small. Protecting library-internal state
  *is* in scope.
- **Distinct objects can still collide via shared driver state**: operating on
  separate objects is not automatically safe when they share driver or context
  state (e.g., changing peer device access while another thread touches affected
  memory). Synchronizing these cases is the caller's responsibility; do not try to
  lock around them internally.
- **Protect internal cached/module-level state**: guard lazily-populated
  `cdef object` caches and module-level state so concurrent access cannot corrupt
  interpreter state (CPython reference counts -- a strictly free-threading hazard).
  Established patterns are `@cython.critical_section` on accessors (#2215), an
  atomic initialization flag (#2216), and `dict.setdefault` for identity caches
  (#2217). Guard state only on objects that are legitimately shared between threads;
  objects that are not meant to be shared (e.g., the thread-local `Device`) do not
  need such guards (see #2321). Reference-count integrity is guaranteed; cache
  value-identity/idempotency is not.
- **Entry points assume the GIL is held**: the helpers in `_cpp/resource_handles.*`
  are called from Cython with the GIL held and do not re-acquire it. Driver and
  destructor callbacks run at arbitrary times, so they take the GIL (`with gil`)
  and probe for interpreter shutdown before touching Python objects.
- **Lock ordering -- release the GIL before entering the driver**: any CUDA work
  reachable from a host callback or a retained object's `__del__` must release the
  GIL before calling the driver, to avoid GIL/driver-lock deadlocks (see the
  `_py_host_trampoline` path and numba-cuda#321). Objects retained into a graph
  (kernel arguments, memcpy/memset operands, `dst_owner`/`src_owner`, and
  host-callback closures) inherit this contract.

## Failure handling

The user-facing contract lives in `docs/source/error_handling.rst`; the rules
below are for contributors. Reviewers and agents should flag violations.

- **Raise by default**: any failure on a path where an exception can propagate
  raises. Driver statuses go through `HANDLE_RETURN` (Cython) or are returned as
  `CUresult` from the C++ handle layer and then `HANDLE_RETURN`ed; never
  replace a `CUresult` with a generic `RuntimeError`, and drain
  `get_last_error()` immediately after a handle constructor returns empty so a
  stale status cannot be misattributed later.
- **Guarantees**: a call that creates a resource must create nothing when it
  raises (undo the creation if a later step fails). Every call except
  `Device.set_current` must leave the calling thread's current context as it
  found it. Do not hand-roll `cuCtxPush/Pop/SetCurrent` sequences in Cython; use
  the handle layer's scoped-context helpers (`invoke_in_context`,
  `invoke_in_context_or_undo`, `cleanup_in_context`, `context_get_device`,
  `graph_node_set_params`) so the failure handling exists in one place.
- **Publish before you raise**: when a driver mutation has succeeded and a later
  step can still fail, commit whatever keeps that mutation memory-safe (for
  example the graph attachment that retains a node's new owners) before raising
  the later error. Rolling back the retention of a live mutation creates a
  dangling reference. When ownership cannot be established, retain the
  resources anyway (leak) rather than release them; a leak is always preferred
  to a use-after-free.
- **Non-propagating paths never raise and never discard a status**: shared_ptr
  deleters, `__dealloc__`, CUDA callbacks and cleanup after a failure report
  through one channel, `report_cuda_error()` / `report_message()` in C++ (the
  `pw_*` wrappers) or `warnings.warn(..., CUDAWarning)` in Cython and Python,
  which emits `cuda.core.CUDAWarning`. No `print(file=sys.stderr)` and no
  `fprintf` outside that helper. `CUDA_ERROR_DEINITIALIZED` is filtered by the
  helper because it means the driver is shutting down.
- **Rollback failure**: the original exception propagates; the failed rollback
  is reported out of band (or chained with `raise ... from` when a second
  exception must be raised). Bare `except:` is acceptable only for
  rollback-then-`raise` blocks.
- **Finalization**: once `py_is_finalizing()` is true, do no Python work from
  destructors or callbacks and accept the leak (see
  `_cpp/resource_handles.hpp` and `_cpp/GRAPH_ATTACHMENTS.md`).
- **Aborting**: `std::abort` (or any process termination) is reserved for an
  internal invariant violation where continuing could corrupt memory or produce
  silently wrong results *and* no leak-based fallback exists. A failed CUDA
  call, including a failed context restoration, never qualifies: raise or
  report instead. There is currently no such path; if one is ever needed it
  must go through a single helper that writes a diagnostic (call, CUDA error,
  invariant, "please report") to stderr before aborting, must never trigger
  during interpreter finalization or for driver-shutdown errors, and must be
  called out in the docs and release notes. An *implicit* abort (an exception
  escaping a `noexcept` function or a deleter, including `std::bad_alloc` from
  an allocation inside `noexcept` code) is a bug (#1489, #2417), not a policy
  choice: `noexcept` helpers must not allocate, or must catch what they call.
- **Testing**: inject restoration failures with
  `cuda.core._resource_handles._set_context_restore_fault_for_testing`; assert
  reports with `pytest.warns(CUDAWarning)` or `warnings.catch_warnings`, never
  by matching stderr text.

## API design guidelines

These are some API design guidelines we try to follow when adding new APIs to
`cuda.core`.  These rules only apply to public APIs.  Private implementation
details can violate these rules at any time.

Public APIs are defined as symbols defined in `__all__` within modules or
subpackages that are not prefixed with `_`.

In code reviews, any violations of this section should be considered
suggestions, not hard rules.  Consistency with existing API design in this code
base is also important.

### Unintentional exposure of symbols

The following things should not be exposed as part of the public API:

- Private symbols (prefixed with `_`)
- Symbols from a third-party module or the standard library
- Helper classes that can not be instantiated from Python

### Naming

As a blanket rule, we follow the naming guidelines for capitalization in PEP 8.

Naming should be consistent.  We should use the same English words for the same
concepts throughout the public API.  When abbreviations are used, they should be
commonly understood, and they should also be used consistently across the public
API.

For all attributes of a class:

- Properties and member variables should be nouns
- Methods should be verbs
- Methods that take no arguments, are idempotent and cheap (O(1) or trivial),
  and do not mutate observable state should be properties

Make sure conceptual pairs match, e.g. add/remove, get/set, create/delete,
alloc/free.

Free functions should be verbs.

### Enumerations

Enumerations from the underlying `cuda_bindings` should not be re-exposed.
Instead, a new `StrEnum` subclass should be used to define the values.  Anywhere
a `StrEnum` is accepted as an argument, a `str` should also be acceptable.  An
invalid value should raise an exception.  When a function returns a `str` drawn
from a small number of values, return a `StrEnum` subclass instead.

For `__post_init__` validation in frozen dataclasses, use the
`not isinstance(value, EnumType) → try EnumType(value) except (ValueError,
TypeError)` pattern (modelled on `_normalize_enum` in
`cuda/core/texture/_texture.pyx`). This accepts the enum itself or a valid
string, and raises `ValueError` eagerly for any other type rather than
silently storing it.

### Exception handling

Raising exceptions is preferred over a C-style return code that must be checked
by the user.

### Type annotations

Python or Cython type annotations should be included for all public APIs.  Avoid
the use of `Any` unless absolutely necessary.  The argument and return types as
defined in the docstrings should match the type annotations.

Python imports should generally be outside of an if typing.TYPE_CHECK: block, even if the imported object is only used in type annotations. Use if typing.TYPE_CHECK: only to avoid creating import cycles. (This guidance maximizes compatibility with the cross-reference mechanisms in Sphinx.)

### Semantics

APIs should exist for both manual resource management (such as `close()`) and
automatic resource management, using context managers or destructors where
appropriate.  Context managers should be implemented with `__enter__` and
`__exit__`, not `contextlib.contextmanager`.  For destructors use `__dealloc__`
where possible, otherwise `__del__`.

### Documentation

The entirety of the public API should be documented in `api.rst` or one of the
subpages linked from it.  Classes that are not directly instantiable but which
may be returned through the public API should be documented in `api_private.rst`
so that they are documented but don't appear in the main index.

### API stability

Reviews should point out where existing public APIs are broken.
