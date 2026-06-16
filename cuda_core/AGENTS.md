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
`__exit__`, not `contextlib.contextmanager`.  For destructors use `__dealloc__`
where possible, otherwise `__del__`.

### Documentation

The entirety of the public API should be documented in `api.rst` or one of the
subpages linked from it.  Classes that are not directly instantiable but which
may be returned through the public API should be documented in `api_private.rst`
so that they are documented but don't appear in the main index.

### API stability

Reviews should point out where existing public APIs are broken.

### Deprecation and API lifecycle

`cuda.core` follows SemVer (see `docs/source/support.rst`):

- **New APIs** may be added at any time (`x.Y.0`).
- **Breaking removals** only happen in **major releases** (`X.0.0`).
- Per the support policy, a deprecation notice must be present for **at least
  one minor release** before the API is actually removed.
- Changes should be notated in the code and also in the release notes in the
  "Deprecated APIs" section.

**Annotating a new API** — Use the `versionadded` decorator from the vendored
`cuda.core._vendored.deprecated.sphinx` module:

```python

from cuda.core._vendored.deprecated.sphinx import versionadded

@versionadded("1.2.0")
def new_feature(...):
    """Short description.
    """
```

Alternatively, if the vagaries of how we implement functions in Cython does not
allow this, you can add the reST `versionadded` directive directly:

```python
def new_feature(...):
    """Short description.

    .. versionadded:: 1.2.0
    """
```

**Annotating a changed API** — Use the `versionchanged` decorator from the
vendored `cuda.core._vendored.deprecated.sphinx` module:

```python

from cuda.core._vendored.deprecated.sphinx import versionchanged

@versionchanged("1.2.0", reason="The old version was broken because...")
def new_feature(...):
    """Short description.
    """
```

Alternatively, if the vagaries of how we implement functions in Cython does not
allow this, you can add the reST `versionchanged` directive directly:

```python
def new_feature(...):
    """Short description.

    .. versionchanged:: 1.2.0
        The old version was broken because...
    """
```

**Deprecating an existing API** — use the `@deprecated` decorator from the
vendored `cuda.core._vendored.deprecated.sphinx` module and add a
`.. deprecated::` directive in the docstring.  The decorator emits a
`DeprecationWarning` at call time; the docstring directive surfaces it in the
generated docs.

```python
from cuda.core._vendored.deprecated.sphinx import deprecated

@deprecated(version="1.2.0", reason="Use `new_feature` instead.")
def old_feature(...):
    """Short description.
    """
```

Rules to follow when deprecating:

- The `version=` argument must be the **first** in which the
  deprecation appears, not the release in which removal is planned.
- The `reason=` string must name the replacement (if one exists) so users
  know what to migrate to.
- Keep the old implementation fully functional — do not change its behavior,
  only add the decorator.
- The deprecated API must remain in the codebase for **at least one full minor
  release cycle** before it can be removed in a subsequent major release.

**Removing a deprecated API** — removals land in a **major release**.  Before
removing, verify that the deprecation has been present since at least the
previous minor release.  Remove the decorator, the implementation, and any
`__all__` entry; update `api.rst` and the release notes accordingly.

At some point in the future, we will provide automation for removal of
deprecated APIs.
