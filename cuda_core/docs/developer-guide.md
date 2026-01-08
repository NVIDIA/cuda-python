# CUDA Core Developer Guide

This guide defines conventions for Python and Cython code in `cuda/core`.

**This project follows [PEP 8](https://peps.python.org/pep-0008/) as the base style guide and [PEP 257](https://peps.python.org/pep-0257/) for docstring conventions.** The guidance in this document extends these with project-specific patterns, particularly for Cython code and the structure of this codebase. Standard conventions are not repeated here.

## Table of Contents

1. [File Structure](#file-structure)
2. [Package Layout](#package-layout)
3. [Import Statements](#import-statements)
4. [Class and Function Definitions](#class-and-function-definitions)
5. [Naming Conventions](#naming-conventions)
6. [Type Annotations and Declarations](#type-annotations-and-declarations)
7. [Docstrings](#docstrings)
8. [Errors and Warnings](#errors-and-warnings)
9. [CUDA-Specific Patterns](#cuda-specific-patterns)
10. [Development Lifecycle](#development-lifecycle)

---

## File Structure

The goal is **readability and maintainability**. A well-organized file lets readers quickly find what they're looking for and understand how the pieces fit together.

To support this, we suggest organizing content from most important to least important: principal classes first, then supporting classes, then implementation details. This way, readers can start at the top and immediately see what matters most. Unlike C/C++ where definitions must precede uses, Python imposes no such constraint—we're free to optimize for the reader.

These are guidelines, not rules. Place helper functions near their call sites if that's clearer. Group related code together if it aids understanding. When in doubt, choose whatever makes the code easiest to read and maintain.

The following is a suggested file organization:

### 1. SPDX Copyright Header

Every file begins with an SPDX copyright header. The pre-commit hook adds or updates these automatically.

### 2. Module Docstring (Optional)

If present, the module docstring comes immediately after the copyright header, before any imports. Per PEP 257, this is the standard location for module-level documentation.

### 3. Import Statements

Imports come next. See [Import Statements](#import-statements) for ordering conventions.

### 4. `__all__` Declaration (Optional)

If present, `__all__` specifies symbols included in star imports.

```python
__all__ = ['DeviceMemoryResource', 'DeviceMemoryResourceOptions']
```

### 5. Type Aliases and Constants (Optional)

Type aliases and module-level constants, if any, come next.

```python
DevicePointerT = driver.CUdeviceptr | int | None
"""Type union for device pointer representations."""

LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
```

### 6. Principal Class or Function

If the file centers on a single class or function (e.g., `_buffer.pyx` defines `Buffer`, `_device.pyx` defines `Device`), that principal element comes first among the definitions.

### 7. Other Public Classes and Functions

Other public classes and functions follow. These might include auxiliary classes (e.g., `DeviceMemoryResourceOptions`), abstract base classes, or additional exports. Organize them logically—by related functionality or typical usage.

### 8. Public Module Functions

Public module-level functions come after classes.

### 9. Private and Implementation Details

Finally, private functions and implementation details: functions prefixed with `_`, `cdef inline` helpers, and any specialized code that would distract from the principal content.

### Example Structure

```python
# <SPDX copyright header>
"""Module for buffer and memory resource management."""

from libc.stdint cimport uintptr_t
from cuda.core._memory._device_memory_resource cimport DeviceMemoryResource
import abc

__all__ = ['Buffer', 'MemoryResource', 'some_public_function']

DevicePointerT = driver.CUdeviceptr | int | None
"""Type union for device pointer representations."""

cdef class Buffer:
    """Principal class for this module."""
    # ...

cdef class MemoryResource:
    """Abstract base class."""
    # ...

def some_public_function():
    """Public API function."""
    # ...

cdef inline void Buffer_close(Buffer self, stream):
    """Private implementation helper."""
    # ...
```

### Notes

- Not every file will have all sections. For example, a utility module may not have a principal class.
- The distinction between "principal" and "other" classes is based on the file's primary purpose. If a file exists primarily to define one class, that class is the principal class.
- Private implementation functions should be placed at the end of the file to keep the public API visible at the top.
- **Within each section**, prefer logical ordering (e.g., by functionality or typical usage). Alphabetical ordering is a reasonable fallback when no clear logical structure exists.

## Package Layout

### File Types

The `cuda/core` package uses three types of files:

1. **`.pyx` files**: Cython implementation files containing the actual code
2. **`.pxd` files**: Cython declaration files containing type definitions and function signatures for C-level access
3. **`.py` files**: Pure Python files for utilities and high-level interfaces

### File Naming Conventions

- **Implementation files**: Use `.pyx` for Cython code, `.py` for pure Python code
- **Declaration files**: Use `.pxd` for Cython type declarations
- **Private modules**: Prefix with underscore (e.g., `_buffer.pyx`, `_device.pyx`)
- **Public modules**: No underscore prefix (e.g., `utils.py`)

### Relationship Between `.pxd` and `.pyx` Files

For each `.pyx` file that defines classes or functions used by other Cython modules, create a corresponding `.pxd` file:

- **`.pxd` file**: Contains `cdef` class declarations, `cdef`/`cpdef` function signatures, and `cdef` attribute declarations
- **`.pyx` file**: Contains the full implementation including Python methods, docstrings, and implementation details

**Example:**

`_buffer.pxd`:
```python
cdef class Buffer:
    cdef:
        uintptr_t      _ptr
        size_t         _size
        MemoryResource _memory_resource
        object         _ipc_data
```

`_buffer.pyx`:
```python
cdef class Buffer:
    """Full implementation with methods and docstrings."""

    def close(self, stream=None):
        """Implementation here."""
        # ...
```

### Module Organization

#### Simple Top-Level Modules

For simple modules at the `cuda/core` level, define classes and functions directly in the module file with an `__all__` list:

```python
# _device.pyx
__all__ = ['Device', 'DeviceProperties']

cdef class Device:
    # ...

cdef class DeviceProperties:
    # ...
```

#### Complex Subpackages

For complex subpackages that require extra structure (like `_memory/`), use the following pattern:

1. **Private submodules**: Each component is implemented in a private submodule (e.g., `_buffer.pyx`, `_device_memory_resource.pyx`)
2. **Submodule `__all__`**: Each submodule defines its own `__all__` list
3. **Subpackage `__init__.py`**: The subpackage `__init__.py` uses `from ._module import *` to assemble the package

**Example structure for `_memory/` subpackage:**

`_memory/_buffer.pyx`:
```python
__all__ = ['Buffer', 'MemoryResource']

cdef class Buffer:
    # ...

cdef class MemoryResource:
    # ...
```

`_memory/_device_memory_resource.pyx`:
```python
__all__ = ['DeviceMemoryResource', 'DeviceMemoryResourceOptions']

cdef class DeviceMemoryResourceOptions:
    # ...

cdef class DeviceMemoryResource:
    # ...
```

`_memory/__init__.py`:
```python
from ._buffer import *  # noqa: F403
from ._device_memory_resource import *  # noqa: F403
from ._graph_memory_resource import *  # noqa: F403
from ._ipc import *  # noqa: F403
from ._legacy import *  # noqa: F403
from ._virtual_memory_resource import *  # noqa: F403
```

This pattern allows:
- **Modular organization**: Each component lives in its own file
- **Clear star-import behavior**: Each submodule explicitly defines what it exports via `__all__`
- **Clean package interface**: The subpackage `__init__.py` assembles all exports into a single namespace
- **Easier refactoring**: Components can be moved or reorganized without changing the public API

**Migration guidance**: Simple top-level modules can be migrated to this subpackage structure when they become sufficiently complex (e.g., when a module grows to multiple related classes or when logical grouping would improve maintainability).

### Guidelines

1. **Always create `.pxd` files for shared Cython types**: If a class or function is `cimport`ed by other modules, provide a `.pxd` declaration file.

2. **Keep `.pxd` files minimal**: Only include declarations needed for Cython compilation. Omit implementation details, docstrings, and Python-only code.

3. **Use `__all__` when helpful**: Define `__all__` to control exported symbols when it simplifies or clarifies the module structure.

4. **Use `from ._module import *` in subpackage `__init__.py`**: This pattern assembles the subpackage API from its submodules. Use `# noqa: F403` to suppress linting warnings about wildcard imports.

5. **Migrate to subpackage structure when complex**: When a top-level module becomes complex (multiple related classes, logical grouping needed), consider refactoring to the subpackage pattern.

6. **Separate concerns**: Use `.py` files for pure Python utilities, `.pyx` files for Cython implementations that need C-level performance.

## Import Statements

Import statements must be organized into five groups, in the following order.

**Note**: Within each group, imports must be sorted alphabetically. This is enforced by pre-commit linters (`ruff`).

### 1. `__future__` Imports

`__future__` imports must come first, before all other imports.


```python
from __future__ import annotations
```

### 2. External `cimport` Statements

External Cython imports from standard libraries and third-party packages. This includes:

- `libc.*` (e.g., `libc.stdint`, `libc.stdlib`, `libc.string`)
- `cpython`
- `cython`
- `cuda.bindings` (CUDA bindings package)

```python
cimport cpython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from cuda.bindings cimport cydriver
```

### 3. cuda-core `cimport` Statements

Cython imports from within the `cuda.core` package.

```python
from cuda.core._memory._buffer cimport Buffer, MemoryResource
from cuda.core._stream cimport Stream_accept, Stream
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    check_or_create_options,
)
```

### 4. External `import` Statements

Regular Python imports from standard libraries and third-party packages. This includes:

- Standard library modules (e.g., `abc`, `typing`, `threading`, `dataclasses`)
- Third-party packages

```python
import abc
import threading
from dataclasses import dataclass
```

### 5. cuda-core `import` Statements

Regular Python imports from within the `cuda.core` package.

```python
from cuda.core._context import Context, ContextOptions
from cuda.core._dlpack import DLDeviceType, make_py_capsule
from cuda.core._utils.cuda_utils import (
    CUDAError,
    driver,
    handle_return,
)
```

### Additional Rules

1. **Alphabetical Ordering**: Within each group, imports must be sorted alphabetically by module name. This is enforced by pre-commit linters.

2. **Multi-line Imports**: When importing multiple items from a single module, use parentheses for multi-line formatting:
   ```python
   from cuda.core._utils.cuda_utils cimport (
       HANDLE_RETURN,
       check_or_create_options,
   )
   ```

3. **Type-only imports**: With `from __future__ import annotations`, types can be imported normally even if only used in annotations. Avoid `TYPE_CHECKING` blocks (see [Type Annotations and Declarations](#type-annotations-and-declarations) for details).

4. **Blank Lines**: Use blank lines to separate the five import groups. Do not use blank lines within a group unless using multi-line import formatting.

5. **`try/except` Blocks**: Import fallbacks (e.g., for optional dependencies) should be placed in the appropriate group (external or cuda-core) using `try/except` blocks.

### Example

```python
# <SPDX copyright header>

from __future__ import annotations

cimport cpython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from cuda.bindings cimport cydriver

from cuda.core._memory._buffer cimport Buffer, MemoryResource
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import abc
from dataclasses import dataclass

from cuda.core._context import Context
from cuda.core._device import Device
from cuda.core._utils.cuda_utils import driver
```

## Class and Function Definitions

### Class Definition Order

Within a class definition, the suggested organization is:

1. **Special (dunder) methods**: Methods with names starting and ending with double underscores. By convention, `__init__` (or `__cinit__` in Cython) should be first among dunder methods, as it defines the class interface.

2. **Methods**: Regular instance methods, class methods (`@classmethod`), and static methods (`@staticmethod`)

3. **Properties**: Properties defined with `@property` decorator

**Note**: Within each section, prefer logical ordering (e.g., grouping related methods). Alphabetical ordering is acceptable when no clear logical structure exists. Developers should use their judgment.

### Example

```python
cdef class Buffer:
    """Example class demonstrating the ordering."""

    # 1. Special (dunder) methods (__cinit__/__init__ first by convention)
    def __cinit__(self):
        """Cython initialization."""
        # ...

    def __init__(self, *args, **kwargs):
        """Python initialization."""
        # ...

    def __buffer__(self, flags: int, /) -> memoryview:
        """Buffer protocol support."""
        # ...

    def __dealloc__(self):
        """Cleanup."""
        # ...

    def __dlpack__(self, *, stream=None):
        """DLPack protocol support."""
        # ...

    def __reduce__(self):
        """Pickle support."""
        # ...

    # 2. Methods
    def close(self, stream=None):
        """Close the buffer."""
        # ...

    def copy_from(self, src, *, stream):
        """Copy data from source buffer."""
        # ...

    def copy_to(self, dst=None, *, stream):
        """Copy data to destination buffer."""
        # ...

    @classmethod
    def from_handle(cls, ptr, size, mr=None):
        """Create buffer from handle."""
        # ...

    def get_ipc_descriptor(self):
        """Get IPC descriptor."""
        # ...

    # 3. Properties
    @property
    def device_id(self) -> int:
        """Device ID property."""
        # ...

    @property
    def handle(self):
        """Handle property."""
        # ...

    @property
    def size(self) -> int:
        """Size property."""
        # ...
```

### Helper Functions

When a class grows long or a method becomes deeply nested, consider extracting implementation details into helper functions. The goal is to keep class definitions easy to navigate—readers shouldn't have to scroll through hundreds of lines to understand a class's interface.

In Cython files, helpers are typically `cdef` or `cdef inline` functions named with the pattern `ClassName_methodname` (e.g., `DMR_close`, `Buffer_close`). Place them at the end of the file or near their call sites, whichever aids readability.

**Example:**

```python
cdef class DeviceMemoryResource:
    def close(self):
        """Close the memory resource."""
        DMR_close(self)

# Helper function (at end of file or nearby)
cdef inline DMR_close(DeviceMemoryResource self):
    if self._handle == NULL:
        return
    # ... implementation ...
```

### Function Definitions

For module-level functions (outside of classes), follow the ordering specified in [File Structure](#file-structure): principal functions first (if applicable), then other public functions, then private functions. Within each group, prefer logical ordering; alphabetical ordering is a reasonable fallback.

## Naming Conventions

Follow PEP 8 naming conventions (CamelCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants, leading underscore for private names).

### Cython `cdef` Variables

Consider prefixing `cdef` variables with `c_` to distinguish them from Python variables. This improves code readability by making it clear which variables are C-level types.

**Preferred:**
```python
def copy_to(self, dst: Buffer = None, *, stream: Stream | GraphBuilder) -> Buffer:
    stream = Stream_accept(stream)
    cdef size_t c_src_size = self._size

    if dst is None:
        dst = self._memory_resource.allocate(c_src_size, stream)

    cdef size_t c_dst_size = dst._size
    if c_dst_size != c_src_size:
        raise ValueError(f"buffer sizes mismatch: src={c_src_size}, dst={c_dst_size}")
    # ...
```

**Also acceptable (if context is clear):**
```python
cdef cydriver.CUdevice get_device_from_ctx(
        cydriver.CUcontext target_ctx, cydriver.CUcontext curr_ctx) except?cydriver.CU_DEVICE_INVALID nogil:
    cdef bint switch_context = (curr_ctx != target_ctx)
    cdef cydriver.CUcontext ctx
    cdef cydriver.CUdevice target_dev
    # ...
```

The `c_` prefix is particularly helpful when mixing Python and Cython variables in the same scope, or when the variable name would otherwise be ambiguous.

## Type Annotations and Declarations

### Python Type Annotations

#### PEP 604 Union Syntax

Use the modern [PEP 604](https://peps.python.org/pep-0604/) union syntax (`X | Y`) instead of `typing.Union` or `typing.Optional`.

**Preferred:**
```python
def allocate(self, size_t size, stream: Stream | GraphBuilder | None = None) -> Buffer:
    # ...

def close(self, stream: Stream | None = None):
    # ...
```

**Avoid:**
```python
from typing import Optional, Union

def allocate(self, size_t size, stream: Optional[Union[Stream, GraphBuilder]] = None) -> Buffer:
    # ...

def close(self, stream: Optional[Stream] = None):
    # ...
```

#### Forward References and `from __future__ import annotations`

Where needed, files should include `from __future__ import annotations` at the top (after the SPDX header). This enables:

1. **Forward references**: Type annotations can reference types that are defined later in the file or in other modules without requiring `TYPE_CHECKING` blocks.

2. **Cleaner syntax**: Annotations are evaluated as strings, avoiding circular import issues.

**Preferred:**
```python
from __future__ import annotations

# Can reference Stream even if it's defined later or in another module
def allocate(self, size_t size, stream: Stream | None = None) -> Buffer:
    # ...
```

**Avoid:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core._stream import Stream

def allocate(self, size_t size, stream: Stream | None = None) -> Buffer:
    # ...
```

#### Guidelines

1. **Use `from __future__ import annotations`**: This should be present in all `.py` and `.pyx` files with type annotations.

2. **Use `|` for unions**: Prefer `X | Y | None` over `Union[X, Y]` or `Optional[X]`.

3. **Avoid `TYPE_CHECKING` blocks**: With `from __future__ import annotations`, forward references work without `TYPE_CHECKING` guards.

4. **Import types normally**: Even if a type is only used in annotations, import it normally (not in a `TYPE_CHECKING` block).

### Cython Type Declarations

Cython uses `cdef` declarations for C-level types. These follow different rules:

```python
cdef class Buffer:
    cdef:
        uintptr_t _ptr
        size_t _size
        MemoryResource _memory_resource
```

For Cython-specific type declarations, see [Cython-Specific Features](#cython-specific-features).

## Docstrings

This project uses the **NumPy docstring style** for all documentation. This format is well-suited for scientific and technical libraries and integrates well with Sphinx documentation generation.

### Format Overview

Docstrings use triple double-quotes (`"""`) and follow this general structure:

```python
"""Summary line.

Extended description (optional).

Parameters
----------
param1 : type
    Description of param1.
param2 : type, optional
    Description of param2. Default is value.

Returns
-------
return_type
    Description of return value.

Raises
------
ExceptionType
    Description of when this exception is raised.

Notes
-----
Additional notes and implementation details.

Examples
--------
>>> example_code()
result
"""
```

### Module Docstrings

Per PEP 257, module docstrings appear at the top of the file, immediately after the copyright header and before any imports. They provide a brief overview of the module's purpose.

```python
# <SPDX copyright header>
"""Module for managing CUDA device memory resources.

This module provides classes and functions for allocating and managing
device memory using CUDA's stream-ordered memory pool API.
"""

from __future__ import annotations
# ... imports ...
```

For simple utility modules, a single-line docstring may suffice:

```python
"""Utility functions for CUDA error handling."""
```

### Class Docstrings

Class docstrings should include:

1. **Summary line**: A one-line description of the class
2. **Extended description** (optional): Additional context about the class
3. **Parameters section**: If the class is callable (has `__init__`), document constructor parameters
4. **Attributes section**: Document public attributes (if any)
5. **Notes section**: Important usage notes, implementation details, or examples
6. **Examples section**: Usage examples (if helpful)

**Example:**

```python
cdef class DeviceMemoryResource(MemoryResource):
    """
    A device memory resource managing a stream-ordered memory pool.

    Parameters
    ----------
    device_id : :class:`Device` | int
        Device or device ordinal for which a memory resource is constructed.
    options : :class:`DeviceMemoryResourceOptions`, optional
        Memory resource creation options. If None, uses the driver's current
        or default memory pool for the specified device.

    Attributes
    ----------
    device_id : int
        The device ID associated with this memory resource.
    is_ipc_enabled : bool
        Whether this memory resource supports IPC.

    Notes
    -----
    To create an IPC-enabled memory resource, specify ``ipc_enabled=True``
    in the options. IPC-enabled resources can share allocations between
    processes.

    Examples
    --------
    >>> dmr = DeviceMemoryResource(0)
    >>> buffer = dmr.allocate(1024)
    """
```

For simple classes, a brief docstring may be sufficient:

```python
@dataclass
cdef class DeviceMemoryResourceOptions:
    """Customizable DeviceMemoryResource options.

    Attributes
    ----------
    ipc_enabled : bool, optional
        Whether to create an IPC-enabled memory pool. Default is False.
    max_size : int, optional
        Maximum pool size. Default is 0 (system-dependent).
    """
```

### Method and Function Docstrings

Method and function docstrings should include:

1. **Summary line**: A one-line description starting with a verb (e.g., "Allocate", "Return", "Create")
2. **Extended description** (optional): Additional details about behavior
3. **Parameters section**: All parameters with types and descriptions
4. **Returns section**: Return type and description
5. **Raises section**: Exceptions that may be raised (if any)
6. **Notes section**: Important implementation details or usage notes (if needed)
7. **Examples section**: Usage examples (if helpful)

**Example:**

```python
def allocate(self, size_t size, stream: Stream | GraphBuilder | None = None) -> Buffer:
    """Allocate a buffer of the requested size.

    Parameters
    ----------
    size : int
        The size of the buffer to allocate, in bytes.
    stream : :class:`Stream` | :class:`GraphBuilder`, optional
        The stream on which to perform the allocation asynchronously.
        If None, an internal stream is used.

    Returns
    -------
    :class:`Buffer`
        The allocated buffer object, which is accessible on the device
        that this memory resource was created for.

    Raises
    ------
    TypeError
        If called on a mapped IPC-enabled memory resource.
    RuntimeError
        If allocation fails.

    Notes
    -----
    The allocated buffer is associated with this memory resource and will
    be deallocated when the buffer is closed or when this resource is closed.
    """
```

For simple functions, a brief docstring may suffice:

```python
def get_ipc_descriptor(self) -> IPCBufferDescriptor:
    """Export a :class:`Buffer` for sharing between processes."""
```

### Property Docstrings

Property docstrings should be concise and focus on what the property represents. For read-write properties, document both getter and setter behavior.

**Read-only property:**

```python
@property
def device_id(self) -> int:
    """Return the device ordinal of this buffer."""
```

**Read-write property:**

```python
@property
def peer_accessible_by(self):
    """
    Get or set the devices that can access allocations from this memory pool.

    Returns
    -------
    tuple of int
        A tuple of sorted device IDs that currently have peer access to
        allocations from this memory pool.

    Notes
    -----
    When setting, accepts a sequence of :class:`Device` objects or device IDs.
    Setting to an empty sequence revokes all peer access.

    Examples
    --------
    >>> dmr.peer_accessible_by = [1]  # Grant access to device 1
    >>> assert dmr.peer_accessible_by == (1,)
    """
```

### Type References in Docstrings

Use Sphinx cross-reference roles to link to other documented objects. Use the most specific role for each type:

| Role | Use for | Example |
|------|---------|---------|
| `:class:` | Classes | `:class:`Buffer`` |
| `:func:` | Functions | `:func:`launch`` |
| `:meth:` | Methods | `:meth:`Device.create_stream`` |
| `:attr:` | Attributes | `:attr:`device_id`` |
| `:mod:` | Modules | `:mod:`multiprocessing`` |
| `:obj:` | Type aliases, other objects | `:obj:`DevicePointerT`` |

The `~` prefix displays only the final component: `:class:`~cuda.core.Buffer`` renders as "Buffer" while still linking to the full path.

For more details, see the [Sphinx Python domain documentation](https://www.sphinx-doc.org/en/master/usage/domains/python.html#cross-referencing-python-objects).

**Example:**

```python
def from_handle(
    ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None
) -> Buffer:
    """Create a new :class:`Buffer` from a pointer.

    Parameters
    ----------
    ptr : :obj:`DevicePointerT`
        Allocated buffer handle object.
    size : int
        Memory size of the buffer.
    mr : :class:`MemoryResource`, optional
        Memory resource associated with the buffer.
    """
```

### Guidelines

1. **Always include docstrings**: All public classes, methods, functions, and properties should have docstrings.

2. **Start with a verb**: Summary lines for methods and functions should start with a verb in imperative mood (e.g., "Allocate", "Return", "Create", not "Allocates", "Returns", "Creates").

3. **Be concise but complete**: Provide enough information for users to understand and use the API, but avoid unnecessary verbosity.

4. **Use proper sections**: Include Parameters, Returns, Raises sections when applicable. Use Notes and Examples sections when they add value.

5. **Document optional parameters**: Clearly indicate optional parameters and their default values.

6. **Use type hints**: Type information in docstrings should complement (not duplicate) type annotations. Use docstrings to provide additional context about types.

7. **Cross-reference related APIs**: Use Sphinx cross-references to link to related classes, methods, and attributes.

8. **Keep private methods brief**: Private methods (starting with `_`) may have minimal docstrings, but should still document non-obvious behavior.

9. **Update docstrings with code changes**: Keep docstrings synchronized with implementation changes.

## Errors and Warnings

### CUDA Exceptions

The project defines custom exceptions for CUDA-specific errors:

- **`CUDAError`**: Base exception for CUDA driver errors
- **`NVRTCError`**: Exception for NVRTC compiler errors (inherits from `CUDAError`)

Use these instead of generic exceptions when reporting CUDA failures.

### CUDA API Error Handling

In `nogil` contexts, use the `HANDLE_RETURN` macro:

```python
with nogil:
    HANDLE_RETURN(cydriver.cuMemAlloc(ptr, size))
```

At the Python level, use `handle_return()` or `raise_if_driver_error()`:

```python
err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
handle_return((err,))
```

### Warnings

When emitting warnings, always specify `stacklevel` so the warning points to the caller:

```python
warnings.warn(message, UserWarning, stacklevel=3)
```

The value depends on call depth—typically `stacklevel=2` for direct calls, `stacklevel=3` when called through a helper.

## CUDA-Specific Patterns

### GIL Management for CUDA Driver API Calls

For optimized Cython code, release the GIL when calling CUDA driver APIs. This improves performance and allows other Python threads to run during CUDA operations.

During initial development, it's fine to use the Python `driver` module without releasing the GIL (see [Development Lifecycle](#development-lifecycle)). GIL release is a performance optimization that can be applied once the implementation is correct.

#### Using `with nogil:` Blocks

Wrap `cydriver` calls in `with nogil:` blocks (or declare entire functions as `nogil`):

```python
cdef int value
with nogil:
    HANDLE_RETURN(cydriver.cuDeviceGetAttribute(&value, attr, device_id))
```

Group multiple driver calls in a single block:

```python
cdef int low, high
with nogil:
    HANDLE_RETURN(cydriver.cuCtxGetStreamPriorityRange(&low, &high))
```

#### Raising Exceptions from `nogil` Context

To raise exceptions from a `nogil` context, acquire the GIL first:

```python
with gil:
    raise CUDAError(f"CUDA operation failed: {error}")
```

## Development Lifecycle

### Two-Phase Development

A common pattern when implementing CUDA functionality is to develop in two phases:

1. **Start with Python**: Use the `driver` module for a straightforward implementation. Write tests to verify correctness. This allows faster iteration and easier debugging.

2. **Optimize with Cython**: Once the implementation is correct, switch to `cydriver` with `nogil` blocks and `HANDLE_RETURN` for better performance.

This approach separates correctness from optimization. Getting the logic right first—with Python's better error messages and stack traces—often saves time overall.

### Python Implementation

Use the `driver` module from `cuda.core._utils.cuda_utils`:

```python
from cuda.core._utils.cuda_utils import driver
from cuda.core._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
)

def get_attribute(self, attr: int) -> int:
    err, value = driver.cuDeviceGetAttribute(attr, self._id)
    raise_if_driver_error(err)
    return value
```

### Cython Optimization

When ready to optimize, switch to `cydriver`:

```python
from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

def get_attribute(self, attr: int) -> int:
    cdef int value
    with nogil:
        HANDLE_RETURN(cydriver.cuDeviceGetAttribute(&value, attr, self._id))
    return value
```

Key changes:
- Replace `driver` with `cydriver`
- Wrap calls in `with nogil:`
- Use `HANDLE_RETURN` instead of `raise_if_driver_error`

Run tests after optimization to verify behavior is unchanged.
