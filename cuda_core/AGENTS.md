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

## Resource lifetime management

CUDA resources (contexts, streams, events, memory pools, device pointers,
libraries, kernels, graphs, arrays, textures/surfaces, etc.) are owned through a
single C++ `std::shared_ptr`-based handle layer -- never by raw driver handles
stored in Python objects. This is the canonical pattern; see
`cuda/core/_cpp/DESIGN.md` and `cuda/core/_cpp/REGISTRY_DESIGN.md`, with
`_stream.pyx`/`_stream.pxd`, `_event.pyx`, and `_memory/_buffer.pyx` as the
reference consumers. When adding or changing a resource-owning type, follow it:

- **Store a handle, not a raw resource.** A `cdef class` holds a `*Handle`
  (a `shared_ptr[const CUxxx]` alias from `_resource_handles.pxd`), not a raw
  `cydriver.CUxxx` handle plus a `bint _owning` flag.
- **`close()` is `self._handle.reset()`.** Do not call `cu*Destroy` directly in
  `close()`, and do not define a `__dealloc__`/`__del__` that calls `cu*Destroy`.
  Destruction runs in the handle's deleter (GIL released) when the last
  reference drops; do not swallow destruction errors.
- **Encode dependencies structurally, not with Python refs.** If resource A
  depends on B (stream->context, texture->backing, mipmap level->parent), embed
  B's handle in A's C++ box so the deleter keeps B alive. Python attributes such
  as `_parent_ref`/`_source_ref` are acceptable only for introspection, never
  for lifetime correctness.
- **Create via a `create_*_handle` factory.** Creation returns an empty handle
  on failure and stashes the code in thread-local state; callers check
  `if not h: HANDLE_RETURN(get_last_error())`.
- **Adding a new resource type** means adding to
  `_cpp/resource_handles.{hpp,cpp}` a handle alias, a box (resource + embedded
  dependency handles), a GIL-guarded deleter, `create_*` functions, and
  `as_cu`/`as_intptr`/`as_py` overloads; then declaring them in
  `_resource_handles.{pxd,pyx}` and populating any new driver pointers from
  `cydriver.__pyx_capi__`. Only `_resource_handles.so` links the C++; consumer
  modules `cimport` from `_resource_handles.pxd`.
- **Distinguish identical integer handle types.** `CUdeviceptr`, `CUtexObject`,
  and `CUsurfObject` are all `unsigned long long`, so `shared_ptr<const T>`
  would collapse to one C++ type and break the `as_*` overload set. Wrap such
  handles in distinct `TaggedHandle<T, Tag>` types (as the NVVM / nvJitLink /
  TexObject / SurfObject handles do).

Do not introduce a second, parallel ownership model (raw handle + `__dealloc__`
+ Python keepalive) for CUDA resources -- it reintroduces the GC-ordering,
interpreter-shutdown, double-free, and cross-language hazards the handle layer
exists to prevent.

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
- **Orchestrated run**: from repo root, `scripts/run_tests.sh core`.

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
`__exit__`, not `contextlib.contextmanager`.

For a type that owns a CUDA resource, automatic cleanup comes from the
`std::shared_ptr` handle's deleter (see **Resource lifetime management**):
`close()` is `self._handle.reset()` and there is **no** `__dealloc__`/`__del__`
that destroys the resource directly. Reserve `__dealloc__` (or `__del__` where
`__dealloc__` is unavailable) for finalizing non-resource state, such as
decref-ing a Python owner -- not for calling a driver destroy.

### Documentation

The entirety of the public API should be documented in `api.rst` or one of the
subpages linked from it.  Classes that are not directly instantiable but which
may be returned through the public API should be documented in `api_private.rst`
so that they are documented but don't appear in the main index.

### API stability

Reviews should point out where existing public APIs are broken.
