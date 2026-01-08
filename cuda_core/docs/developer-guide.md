# CUDA Core Style Guide

This style guide defines conventions for Python and Cython code in `cuda/core/experimental`.

**This project follows [PEP 8](https://peps.python.org/pep-0008/) as the base style guide.** The rules in this document highlight project-specific conventions and extensions beyond PEP 8, particularly for Cython code and the structure of this codebase.

## Table of Contents

1. [File Structure](#file-structure)
2. [Package Layout](#package-layout)
3. [Import Statements](#import-statements)
4. [Class and Function Definitions](#class-and-function-definitions)
5. [Naming Conventions](#naming-conventions)
6. [Type Annotations and Declarations](#type-annotations-and-declarations)
7. [Docstrings](#docstrings)
8. [Errors and Warnings](#errors-and-warnings)
9. [Memory Management](#memory-management)
10. [Thread Safety and Concurrency](#thread-safety-and-concurrency)
11. [Cython-Specific Features](#cython-specific-features)
12. [Constants and Magic Numbers](#constants-and-magic-numbers)
13. [Comments and Inline Documentation](#comments-and-inline-documentation)
14. [Code Organization Within Files](#code-organization-within-files)
15. [Performance Considerations](#performance-considerations)
16. [API Design Principles](#api-design-principles)
17. [CUDA-Specific Patterns](#cuda-specific-patterns)
18. [Development Lifecycle](#development-lifecycle)
19. [Copyright and Licensing](#copyright-and-licensing)

---

## File Structure

Files in `cuda/core/experimental` must follow a consistent structure. The ordering of elements within a file is as follows:

### 1. SPDX Copyright Header

The file must begin with the SPDX copyright header as specified in [Copyright and Licensing](#copyright-and-licensing).

### 2. Import Statements

Import statements come immediately after the copyright header. Follow the import ordering conventions specified in [Import Statements](#import-statements).

### 3. `__all__` Declaration

If the module exports public API elements, include an `__all__` list after the imports and before any other code. This explicitly defines the public API of the module.

```python
__all__ = ['DeviceMemoryResource', 'DeviceMemoryResourceOptions']
```

### 4. Type Aliases and Constants

Type aliases and module-level constants may immediately follow `__all__` (if present) or come after imports:

```python
DevicePointerT = driver.CUdeviceptr | int | None
"""Type union for device pointer representations."""

LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
```

### 5. Principal Class or Function

If the file principally implements a single class or function (e.g., `_buffer.pyx` defines the `Buffer` class, `_device.pyx` defines the `Device` class), that principal class or function should come next, immediately after `__all__` (if present).

**The principal class or function is an exception to alphabetical ordering** and appears first in its section.

### 6. Other Public Classes and Functions

Following the principal class or function, define other public classes and functions. These include:

- **Auxiliary classes**: Supporting classes that are part of the public API (e.g., `DeviceMemoryResourceOptions` is an auxiliary class used by `DeviceMemoryResource`)
- **Abstract base classes**: ABCs that define interfaces (e.g., `MemoryResource` in `_buffer.pyx`)
- **Other public classes**: Additional classes exported by the module

**All classes and functions in this section should be sorted alphabetically by name**, regardless of their relationship to the principal class. The principal class appears first as an exception to this rule.

**Example:** In `_device_memory_resource.pyx`, `DeviceMemoryResource` is the principal class and appears first. Then `DeviceMemoryResourceOptions` appears after it (alphabetically after the principal class), even though it's an auxiliary/options class.

### 7. Public Module Functions

After all classes, define public module-level functions that are part of the API.

### 8. Private or Implementation Functions

Finally, define private functions and implementation details. These include:

- Functions with names starting with `_` (private)
- `cdef inline` functions used for internal implementation
- Helper functions not part of the public API

### Example Structure

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Imports (cimports first, then regular imports)
from libc.stdint cimport uintptr_t
from cuda.core.experimental._memory._device_memory_resource cimport DeviceMemoryResource
import abc

__all__ = ['Buffer', 'MemoryResource', 'some_public_function']

# Type aliases (if any)
DevicePointerT = driver.CUdeviceptr | int | None
"""Type union for device pointer representations."""

# Principal class
cdef class Buffer:
    """Principal class for this module."""
    # ...

# Other public classes
cdef class MemoryResource:
    """Abstract base class."""
    # ...

# Public module functions
def some_public_function():
    """Public API function."""
    # ...

# Private implementation functions
cdef inline void Buffer_close(Buffer self, stream):
    """Private implementation helper."""
    # ...
```

### Notes

- Not every file will have all sections. For example, a utility module may not have a principal class.
- The distinction between "principal" and "other" classes is based on the file's primary purpose. If a file exists primarily to define one class, that class is the principal class.
- Private implementation functions should be placed at the end of the file to keep the public API visible at the top.
- **Within each section**, classes and functions should be sorted alphabetically by name. The principal class or function is an exception to this rule, as it appears first in its respective section.

## Package Layout

### File Types

The `cuda/core/experimental` package uses three types of files:

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
    cdef:
        uintptr_t      _ptr
        size_t         _size
        MemoryResource _memory_resource
        object         _ipc_data

    def close(self, stream=None):
        """Implementation here."""
        # ...
```

### Module Organization

#### Simple Top-Level Modules

For simple modules at the `cuda/core/experimental` level, define classes and functions directly in the module file with an `__all__` list:

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
2. **Submodule `__all__`**: Each submodule defines its own `__all__` list with the symbols it exports
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
- **Clear public API**: Each submodule explicitly defines what it exports via `__all__`
- **Clean package interface**: The subpackage `__init__.py` assembles all exports into a single namespace
- **Easier refactoring**: Components can be moved or reorganized without changing the public API

**Migration guidance**: Simple top-level modules can be migrated to this subpackage structure when they become sufficiently complex (e.g., when a module grows to multiple related classes or when logical grouping would improve maintainability).

### Guidelines

1. **Always create `.pxd` files for shared Cython types**: If a class or function is `cimport`ed by other modules, provide a `.pxd` declaration file.

2. **Keep `.pxd` files minimal**: Only include declarations needed for Cython compilation. Omit implementation details, docstrings, and Python-only code.

3. **Use `__all__` in submodules**: Each submodule should define `__all__` to explicitly declare its public API.

4. **Use `from ._module import *` in subpackage `__init__.py`**: This pattern assembles the subpackage API from its submodules. Use `# noqa: F403` to suppress linting warnings about wildcard imports.

5. **Migrate to subpackage structure when complex**: When a top-level module becomes complex (multiple related classes, logical grouping needed), consider refactoring to the subpackage pattern.

6. **Separate concerns**: Use `.py` files for pure Python utilities, `.pyx` files for Cython implementations that need C-level performance.

## Import Statements

Import statements must be organized into five groups, in the following order:
**Note**: Within each section, imports should be sorted alphabetically.

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

Cython imports from within the `cuda.core.experimental` package.

```python
from cuda.core.experimental._memory._buffer cimport Buffer, MemoryResource
from cuda.core.experimental._stream cimport Stream_accept, Stream
from cuda.core.experimental._utils.cuda_utils cimport (
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

Regular Python imports from within the `cuda.core.experimental` package.

```python
from cuda.core.experimental._context import Context, ContextOptions
from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._utils.cuda_utils import (
    CUDAError,
    driver,
    handle_return,
)
```

### Additional Rules

1. **Alphabetical Ordering**: Within each group, imports should be sorted alphabetically by module name.

2. **Multi-line Imports**: When importing multiple items from a single module, use parentheses for multi-line formatting:
   ```python
   from cuda.core.experimental._utils.cuda_utils cimport (
       HANDLE_RETURN,
       check_or_create_options,
   )
   ```

3. **Type-only imports**: With `from __future__ import annotations`, types can be imported normally even if only used in annotations. Avoid `TYPE_CHECKING` blocks (see [Type Annotations and Declarations](#type-annotations-and-declarations) for details).

4. **Blank Lines**: Use blank lines to separate the five import groups. Do not use blank lines within a group unless using multi-line import formatting.

5. **`try/except` Blocks**: Import fallbacks (e.g., for optional dependencies) should be placed in the appropriate group (external or cuda-core) using `try/except` blocks.

### Example

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# 1. __future__ imports
from __future__ import annotations

# 2. External cimports
cimport cpython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from cuda.bindings cimport cydriver

# 3. cuda-core cimports
from cuda.core.experimental._memory._buffer cimport Buffer, MemoryResource
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN

# 4. External imports
import abc
from dataclasses import dataclass

# 5. cuda-core imports
from cuda.core.experimental._context import Context
from cuda.core.experimental._device import Device
from cuda.core.experimental._utils.cuda_utils import driver
```

## Class and Function Definitions

### Class Definition Order

Within a class definition, elements must be organized in the following order:

1. **Special (dunder) methods**: Methods with names starting and ending with double underscores (e.g., `__init__`, `__cinit__`, `__dealloc__`, `__reduce__`, `__dlpack__`)

2. **Methods**: Regular instance methods, class methods (`@classmethod`), and static methods (`@staticmethod`)

3. **Properties**: Properties defined with `@property` decorator

**Note**: Within each section (dunder methods, methods, properties), elements should be sorted alphabetically by name.

### Example

```python
cdef class Buffer:
    """Example class demonstrating the ordering."""

    # 1. Special (dunder) methods (alphabetically sorted)
    def __buffer__(self, flags: int, /) -> memoryview:
        """Buffer protocol support."""
        # ...

    def __cinit__(self):
        """Cython initialization."""
        # ...

    def __dealloc__(self):
        """Cleanup."""
        # ...

    def __dlpack__(self, *, stream=None):
        """DLPack protocol support."""
        # ...

    def __init__(self, *args, **kwargs):
        """Python initialization."""
        # ...

    def __reduce__(self):
        """Pickle support."""
        # ...

    # 2. Methods (alphabetically sorted)
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

    # 3. Properties (alphabetically sorted)
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

Sometimes, implementation details are moved outside of the class definition to improve readability. Helper functions should be placed at the end of the file (in the private/implementation section) when:

- The indentation level exceeds 4 levels
- A method definition is long (>20 lines)
- The class definition itself is very long

In Cython files, these are often `cdef` or `cdef inline` functions. The helper function name typically follows the pattern `ClassName_methodname` (e.g., `DMR_close`, `Buffer_close`).

**Example:**

```python
cdef class DeviceMemoryResource:
    def close(self):
        """Close the memory resource."""
        DMR_close(self)  # Calls helper function

# ... other classes and functions ...

# Helper function at end of file
cdef inline DMR_close(DeviceMemoryResource self):
    """Implementation moved outside class for readability."""
    if self._handle == NULL:
        return
    # ... implementation ...
```

### Function Definitions

For module-level functions (outside of classes), follow the ordering specified in [File Structure](#file-structure): principal functions first (if applicable), then other public functions, then private functions. Within each group, sort alphabetically.

## Naming Conventions

### Class Names

Use **PascalCase** (also known as CapWords) for class names.

```python
cdef class Buffer:
    # ...

cdef class DeviceMemoryResource:
    # ...

class CUDAError(Exception):
    # ...
```

### Function and Method Names

Use **snake_case** for function and method names.

```python
def allocate(self, size_t size, stream=None) -> Buffer:
    # ...

def get_ipc_descriptor(self) -> IPCBufferDescriptor:
    # ...

cdef inline void Buffer_close(Buffer self, stream):
    # ...
```

### Variable Names

#### Python Variables

Use **snake_case** for Python variables.

```python
device_id = 0
memory_resource = DeviceMemoryResource(device_id)
buffer_size = 1024
```

#### Private Attributes

Use **snake_case** with a leading underscore for private instance attributes.

```python
cdef class Buffer:
    cdef:
        uintptr_t _ptr
        size_t _size
        MemoryResource _memory_resource
        object _ipc_data
```

#### Cython `cdef` Variables

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

### Constants

Use **UPPER_SNAKE_CASE** for module-level constants.

```python
LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
PER_THREAD_DEFAULT_STREAM = C_PER_THREAD_DEFAULT_STREAM

RUNTIME_CUDA_ERROR_EXPLANATIONS = {
    # ...
}
```

### Private Module-Level Names

Use **snake_case** with a leading underscore for private module-level functions, classes, and variables.

```python
_fork_warning_checked = False

def _reduce_3_tuple(t: tuple):
    # ...

cdef inline void _helper_function():
    # ...
```

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
    from cuda.core.experimental._stream import Stream

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

Module docstrings should appear after imports and `__all__` (if present), before any classes or functions. They should provide a brief overview of the module's purpose.

```python
"""Module for managing CUDA device memory resources.

This module provides classes and functions for allocating and managing
device memory using CUDA's stream-ordered memory pool API.
"""
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
    device_id : Device | int
        Device or Device ordinal for which a memory resource is constructed.
    options : DeviceMemoryResourceOptions, optional
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
    stream : Stream | GraphBuilder, optional
        The stream on which to perform the allocation asynchronously.
        If None, an internal stream is used.

    Returns
    -------
    Buffer
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
    """Export a buffer allocated for sharing between processes."""
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
    When setting, accepts a sequence of Device objects or device IDs.
    Setting to an empty sequence revokes all peer access.

    Examples
    --------
    >>> dmr.peer_accessible_by = [1]  # Grant access to device 1
    >>> assert dmr.peer_accessible_by == (1,)
    """
```

### Type References in Docstrings

Use Sphinx-style cross-references for types:

- **Classes**: ``:class:`Buffer` `` or ``:class:`~_memory.Buffer` `` (with `~` to hide module path)
- **Methods**: ``:meth:`DeviceMemoryResource.allocate` ``
- **Attributes**: ``:attr:`device_id` ``
- **Modules**: ``:mod:`multiprocessing` ``
- **Objects**: ``:obj:`~_memory.DevicePointerT` ``

**Example:**

```python
def from_handle(
    ptr: DevicePointerT, size_t size, mr: MemoryResource | None = None
) -> Buffer:
    """Create a new :class:`Buffer` object from a pointer.

    Parameters
    ----------
    ptr : :obj:`~_memory.DevicePointerT`
        Allocated buffer handle object.
    size : int
        Memory size of the buffer.
    mr : :obj:`~_memory.MemoryResource`, optional
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

### Exception Types

#### Custom Exceptions

The project defines custom exception types for CUDA-specific errors:

- **`CUDAError`**: Base exception for CUDA-related errors
- **`NVRTCError`**: Exception for NVRTC (compiler) errors, inherits from `CUDAError`

```python
from cuda.core.experimental._utils.cuda_utils import CUDAError, NVRTCError

raise CUDAError("CUDA operation failed")
raise NVRTCError("NVRTC compilation error")
```

#### Standard Python Exceptions

Use standard Python exceptions when appropriate:

- **`ValueError`**: Invalid argument values
- **`TypeError`**: Invalid argument types
- **`RuntimeError`**: Runtime errors that don't fit other categories
- **`NotImplementedError`**: Features that are not yet implemented
- **`BufferError`**: Buffer protocol-related errors

```python
if size < 0:
    raise ValueError(f"size must be non-negative, got {size}")

if not isinstance(stream, Stream):
    raise TypeError(f"stream must be a Stream, got {type(stream)}")

if self.is_mapped:
    raise RuntimeError("Memory resource is not IPC-enabled")
```

### Raising Errors

#### Error Messages

Error messages should be clear and include context:

**Preferred:**
```python
if dst_size != src_size:
    raise ValueError(
        f"buffer sizes mismatch between src and dst "
        f"(sizes are: src={src_size}, dst={dst_size})"
    )
```

**Avoid:**
```python
if dst_size != src_size:
    raise ValueError("sizes don't match")
```

#### CUDA API Error Handling

For CUDA Driver API calls, use the `HANDLE_RETURN` macro in `nogil` contexts:

```python
cdef int allocate_buffer(uintptr_t* ptr, size_t size) except?-1 nogil:
    HANDLE_RETURN(cydriver.cuMemAlloc(ptr, size))
    return 0
```

For Python-level CUDA error handling, use `handle_return()`:

```python
from cuda.core.experimental._utils.cuda_utils import handle_return

err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
handle_return((err,))
```

Or use `raise_if_driver_error()` for direct error raising:

```python
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
)

err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
raise_if_driver_error(err)
```

#### Error Explanations

CUDA errors include explanations from dictionaries (`DRIVER_CU_RESULT_EXPLANATIONS`, `RUNTIME_CUDA_ERROR_EXPLANATIONS`) when available. The error checking functions (`_check_driver_error()`, `_check_runtime_error()`) automatically include these explanations in the error message.

### Warnings

#### Warning Categories

Use appropriate warning categories:

- **`UserWarning`**: For user-facing warnings about potentially problematic usage
- **`DeprecationWarning`**: For deprecated features that will be removed in future versions

```python
import warnings

warnings.warn(
    "multiprocessing start method is 'fork', which CUDA does not support. "
    "Forked subprocesses exhibit undefined behavior. "
    "Set the start method to 'spawn' before creating processes that use CUDA.",
    UserWarning,
    stacklevel=3
)

warnings.warn(
    "Implementing __cuda_stream__ as an attribute is deprecated; "
    "it must be implemented as a method",
    DeprecationWarning,
    stacklevel=3
)
```

#### Stack Level

Always specify the `stacklevel` parameter to point to the caller, not the warning location:

```python
warnings.warn(message, UserWarning, stacklevel=3)
```

The `stacklevel` value depends on the call depth. Use `stacklevel=2` for direct function calls, `stacklevel=3` for calls through helper functions.

#### One-Time Warnings

For warnings that should only be emitted once per process, use a module-level flag:

```python
_fork_warning_checked = False

def check_multiprocessing_start_method():
    global _fork_warning_checked
    if _fork_warning_checked:
        return
    _fork_warning_checked = True

    # ... check condition and emit warning ...
    warnings.warn(message, UserWarning, stacklevel=3)
```

#### Deprecation Warnings

For deprecation warnings, use `warnings.simplefilter("once", DeprecationWarning)` to ensure each deprecation message is shown only once:

```python
warnings.simplefilter("once", DeprecationWarning)
warnings.warn(
    "Feature X is deprecated and will be removed in a future version",
    DeprecationWarning,
    stacklevel=3
)
```

### Guidelines

1. **Use specific exception types**: Choose the most appropriate exception type for the error condition.

2. **Include context in error messages**: Error messages should include relevant values and context to help users diagnose issues.

3. **Use custom exceptions for CUDA errors**: Use `CUDAError` or `NVRTCError` for CUDA-specific errors rather than generic exceptions.

4. **Specify stacklevel for warnings**: Always include `stacklevel` parameter in `warnings.warn()` calls to point to the actual caller.

5. **Use one-time warnings for repeated operations**: When a warning could be triggered multiple times, use a flag to ensure it's only shown once.

6. **Prefer warnings over errors for recoverable issues**: Use warnings for issues that don't prevent execution but may cause problems.

## Memory Management

### Resource Lifecycle

CUDA memory resources and buffers follow a clear lifecycle pattern:

1. **Creation**: Resources and buffers are created through factory methods or constructors
2. **Usage**: Objects are used for CUDA operations
3. **Cleanup**: Resources are explicitly closed or automatically cleaned up

### Explicit Cleanup

Always provide explicit cleanup methods for resources that manage CUDA handles:

```python
cdef class DeviceMemoryResource:
    def close(self):
        """Close the memory resource and release CUDA handles."""
        DMR_close(self)

    def __dealloc__(self):
        """Automatic cleanup when object is garbage collected."""
        DMR_close(self)
```

### Buffer Lifecycle

Buffers are associated with memory resources and should be closed when no longer needed:

```python
cdef class Buffer:
    def close(self, stream: Stream | GraphBuilder | None = None):
        """Deallocate this buffer asynchronously on the given stream."""
        Buffer_close(self, stream)

    def __dealloc__(self):
        """Automatic cleanup if not explicitly closed."""
        self.close(self._alloc_stream)
```

### Guidelines

1. **Provide explicit `close()` methods**: All resources managing CUDA handles should have a `close()` method for explicit cleanup.

2. **Implement `__dealloc__` as safety net**: Use `__dealloc__` to ensure cleanup happens even if users forget to call `close()`, but don't rely on it for normal operation.

3. **Document cleanup behavior**: Clearly document when cleanup happens automatically versus when it must be called explicitly.

4. **Handle cleanup errors gracefully**: Cleanup methods should be idempotent (safe to call multiple times) and handle errors without raising exceptions when possible.

5. **Use stream-ordered deallocation**: When deallocating buffers, use the appropriate stream for asynchronous cleanup to avoid blocking operations.

6. **Track resource ownership**: Clearly document which objects own CUDA handles and are responsible for cleanup.

## Thread Safety and Concurrency

### Thread-Local Storage

Use `threading.local()` for thread-local state that needs to persist across function calls:

```python
import threading

_tls = threading.local()

def some_function():
    if not hasattr(_tls, 'devices'):
        _tls.devices = []
    return _tls.devices
```

### Locks for Shared State

Use `threading.Lock()` to protect shared mutable state:

```python
import threading

_lock = threading.Lock()

def thread_safe_operation():
    with _lock:
        # Critical section
        # Modify shared state
        pass
```

### Combining Locks with `nogil`

When protecting CUDA operations, acquire the lock before entering `nogil` context:

```python
def thread_safe_cuda_operation():
    with _lock, nogil:
        HANDLE_RETURN(cydriver.cuSomeOperation())
```

### One-Time Initialization

For one-time initialization that must be thread-safe, use a lock with a flag:

```python
cdef bint _initialized = False
_lock = threading.Lock()

def initialize():
    global _initialized
    with _lock:
        if not _initialized:
            # Perform initialization
            _initialized = True
```

### Guidelines

1. **Use thread-local storage for per-thread state**: When state needs to be isolated per thread, use `threading.local()`.

2. **Protect shared mutable state**: Use locks to protect any shared mutable state that could be accessed from multiple threads.

3. **Minimize lock scope**: Keep critical sections as short as possible to reduce contention.

4. **Document thread safety**: Clearly document which operations are thread-safe and which require external synchronization.

5. **Avoid global mutable state**: Prefer thread-local storage or instance variables over global mutable state when possible.

6. **Combine locks with `nogil` correctly**: Acquire locks before entering `nogil` contexts, not inside them.

## Cython-Specific Features

### Function Declarations

Cython provides three types of function declarations:

1. **`def`**: Python function, callable from Python, slower than C functions
2. **`cdef`**: C function, not callable from Python, fastest
3. **`cpdef`**: Hybrid function, callable from both Python and C, faster than `def` but slower than `cdef`

**Guidelines:**

- Use `cdef` for internal helper functions that are only called from Cython code
- Use `cpdef` when a function needs to be callable from Python but performance is important
- Use `def` for public Python API functions where flexibility is more important than performance

```python
# Internal helper - only used in Cython
cdef inline void Buffer_close(Buffer self, stream):
    # ...

# Public API - callable from Python, performance important
cpdef inline int _check_driver_error(cydriver.CUresult error) except?-1 nogil:
    # ...

# Public API - standard Python function
def allocate(self, size_t size, stream=None) -> Buffer:
    # ...
```

### Class Declarations

Use `cdef class` for Cython extension types:

```python
cdef class Buffer:
    cdef:
        uintptr_t _ptr
        size_t _size
        MemoryResource _memory_resource
```

### The `nogil` Context

Use `nogil` to release the Global Interpreter Lock (GIL) for performance-critical C operations. See [CUDA-Specific Patterns](#cuda-specific-patterns) for detailed guidelines.

### Exception Handling

Use `except?` or `except` clauses to propagate exceptions from `nogil` functions:

```python
cdef int get_device_from_ctx(...) except?cydriver.CU_DEVICE_INVALID nogil:
    # Returns CU_DEVICE_INVALID on error, otherwise raises exception
```

### Type Declarations

Declare C types explicitly for performance:

```python
cdef:
    int device_id
    size_t buffer_size
    cydriver.CUdeviceptr ptr
```

### Inline Functions

Use `inline` for small, frequently-called functions:

```python
cdef inline void Buffer_close(Buffer self, stream):
    # ...
```

### Guidelines

1. **Choose the right function type**: Use `cdef` for internal code, `cpdef` for performance-critical public APIs, `def` for standard public APIs.

2. **Declare types explicitly**: Use `cdef` declarations for C-level types to enable optimizations.

3. **Use `inline` judiciously**: Mark small, frequently-called functions as `inline`, but avoid overuse.

4. **Handle exceptions properly**: Use appropriate exception clauses (`except`, `except?`) for `nogil` functions.

5. **Document Cython-specific behavior**: When using Cython features that affect the Python API, document them clearly.

## Constants and Magic Numbers

### Naming Constants

Use **UPPER_SNAKE_CASE** for module-level constants:

```python
LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
PER_THREAD_DEFAULT_STREAM = C_PER_THREAD_DEFAULT_STREAM

RUNTIME_CUDA_ERROR_EXPLANATIONS = {
    # ...
}
```

### CUDA Constants

For CUDA API constants, use the bindings directly or create aliases with descriptive names:

```python
from cuda.bindings cimport cydriver

# Use CUDA constants directly
cdef cydriver.CUdevice device_id = cydriver.CU_DEVICE_INVALID

# Or create descriptive aliases
cdef object CU_DEVICE_INVALID = cydriver.CU_DEVICE_INVALID
```

### Avoid Magic Numbers

Replace magic numbers with named constants:

**Avoid:**
```python
if flags & 1:  # What does 1 mean?
    # ...
```

**Preferred:**
```python
if flags & cydriver.CUstream_flags.CU_STREAM_NON_BLOCKING:
    # ...
```

### Dictionary Mappings

Use dictionaries to map between string representations and constants:

```python
_access_flags = {
    "rw": cydriver.CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    "r": cydriver.CU_MEM_ACCESS_FLAGS_PROT_READ,
    None: 0
}
```

### Guidelines

1. **Name all constants**: Avoid magic numbers and strings. Use descriptive constant names.

2. **Use UPPER_SNAKE_CASE**: Follow Python convention for module-level constants.

3. **Group related constants**: Define related constants together, optionally in a dictionary or class.

4. **Document non-obvious constants**: If a constant's purpose isn't immediately clear, add a comment explaining it.

5. **Prefer CUDA bindings**: Use constants from `cuda.bindings` directly when possible rather than redefining them.

## Comments and Inline Documentation

### TODO Comments

Use `TODO` comments to mark incomplete work or future improvements:

```python
# TODO: It is better to take a stream for latter deallocation
return Buffer._init(ptr, size, mr=mr)

# TODO: consider lower this to Cython
expl = DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
```

### NOTE Comments

Use `NOTE` comments to explain non-obvious implementation details:

```python
# NOTE: match this behavior to DeviceMemoryResource.allocate()
stream = default_stream()

# NOTE: this is referenced in instructions to debug nvbug 5698116
cpdef DMR_mempool_get_access(DeviceMemoryResource dmr, int device_id):
```

### Implementation Comments

Add comments to explain complex logic or non-obvious behavior:

```python
# Must not serialize the parent's stream!
return Buffer.from_ipc_descriptor, (self.memory_resource, self.get_ipc_descriptor())

# This works around nvbug 5698116. When a memory pool handle is recycled
# the new handle inherits the peer access state of the previous handle.
if self._peer_accessible_by:
    self.peer_accessible_by = []
```

### Inline Type Comments

Use type comments sparingly, only when type annotations aren't sufficient:

```python
import platform  # no-cython-lint
```

### Guidelines

1. **Use TODO for incomplete work**: Mark known limitations, future improvements, or incomplete features with `TODO` comments.

2. **Use NOTE for important context**: Add `NOTE` comments to explain non-obvious implementation decisions or workarounds.

3. **Explain complex logic**: Add comments to explain why code is written a certain way, not what it does (the code should be self-explanatory).

4. **Keep comments up-to-date**: Update or remove comments when code changes.

5. **Avoid obvious comments**: Don't comment what the code clearly shows. Focus on the "why" rather than the "what".

6. **Document workarounds**: Always document workarounds for bugs (include bug numbers when available) and explain why they're necessary.

## Code Organization Within Files

### Overall Structure

Follow the ordering specified in [File Structure](#file-structure):

1. SPDX copyright header
2. Import statements
3. `__all__` declaration
4. Type aliases and constants (optional)
5. Principal class/function
6. Other public classes and functions
7. Public module functions
8. Private/implementation functions

### Within Classes

Follow the ordering specified in [Class and Function Definitions](#class-and-function-definitions):

1. Special (dunder) methods (alphabetically sorted)
2. Methods (alphabetically sorted)
3. Properties (alphabetically sorted)

### Helper Functions

Move complex implementation details to helper functions at the end of the file. See [Class and Function Definitions - Helper Functions](#helper-functions) for details.

### Type Aliases and Constants

Type aliases and module-level constants should be defined after `__all__` (if present) or after imports, before classes. See [File Structure](#file-structure) for the complete ordering.

```python
DevicePointerT = driver.CUdeviceptr | int | None
"""Type union for device pointer representations."""

LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
```

### Guidelines

1. **Follow the established ordering**: Maintain consistency with the file structure and class definition ordering rules.

2. **Group related code**: Keep related functions and classes together.

3. **Separate public and private**: Clearly separate public API from implementation details.

4. **Use helper functions**: Extract complex logic into helper functions to improve readability.

5. **Keep related code close**: Place helper functions near the code that uses them, or group all helpers at the end of the file.

## Performance Considerations

### Use Cython Types

Declare C types explicitly for performance-critical code:

```python
cdef:
    int device_id
    size_t buffer_size
    cydriver.CUdeviceptr ptr
```

### Prefer `cdef` for Internal Functions

Use `cdef` functions for internal operations that don't need to be callable from Python:

```python
cdef inline void Buffer_close(Buffer self, stream):
    # Fast C-level function
```

### Release GIL for CUDA Operations

Always release the GIL when calling CUDA driver APIs. See [CUDA-Specific Patterns](#cuda-specific-patterns) for details.

### Minimize Python Object Creation

Avoid creating Python objects in hot paths:

```python
# Avoid: Creates Python list
result = []
for i in range(n):
    result.append(i)

# Preferred: Use C array or pre-allocate
cdef int* c_result = <int*>malloc(n * sizeof(int))
```

### Use `inline` for Small Functions

Mark small, frequently-called functions as `inline`:

```python
cdef inline int get_device_id(DeviceMemoryResource mr):
    return mr._dev_id
```

### Avoid Unnecessary Type Conversions

Minimize conversions between C and Python types:

```python
# Avoid: Unnecessary conversion
cdef int size = int(self._size)

# Preferred: Use C type directly
cdef size_t size = self._size
```

### Guidelines

1. **Profile before optimizing**: Don't optimize prematurely. Use profiling to identify actual bottlenecks.

2. **Use C types in hot paths**: Declare C types (`cdef`) for variables used in performance-critical loops.

3. **Release GIL appropriately**: Always release GIL for CUDA operations, but be careful about Python object access.

4. **Minimize Python overhead**: Avoid Python object creation, method calls, and attribute access in hot paths.

5. **Use `inline` judiciously**: Mark small, frequently-called functions as `inline`, but don't overuse (compiler may ignore if function is too large).

6. **Cache expensive lookups**: Cache results of expensive operations (e.g., dictionary lookups, attribute access) when used repeatedly.

## API Design Principles

### Public vs Private API

Use naming conventions to distinguish public and private APIs:

- **Public API**: No leading underscore, documented in docstrings, included in `__all__`
- **Private API**: Leading underscore (`_`), may have minimal documentation, not in `__all__`

```python
__all__ = ['Buffer', 'MemoryResource']  # Public API

# Public API
cdef class Buffer:
    def allocate(self):  # Public method
        # ...

# Private API
cdef inline void Buffer_close(Buffer self, stream):  # Private helper
    # ...
```

### Backward Compatibility

Maintain backward compatibility when possible:

- **Deprecation warnings**: Use `DeprecationWarning` for APIs that will be removed
- **Gradual migration**: Provide both old and new APIs during transition periods
- **Version documentation**: Document when APIs were introduced or deprecated

### Consistency

Maintain consistency across the API:

- **Naming patterns**: Use consistent naming patterns (e.g., `from_*` for factory methods)
- **Parameter ordering**: Use consistent parameter ordering across similar functions
- **Return types**: Use consistent return types for similar operations

### Factory Methods

Use class methods or static methods for factory functions:

```python
@classmethod
def from_ipc_descriptor(cls, mr, ipc_descriptor, stream=None) -> Buffer:
    """Factory method to create Buffer from IPC descriptor."""
    # ...

@staticmethod
def from_handle(ptr, size, mr=None) -> Buffer:
    """Factory method to create Buffer from handle."""
    # ...
```

### Error Handling

Design APIs to fail fast with clear error messages:

- **Validate inputs early**: Check parameters at the start of functions
- **Use appropriate exceptions**: Raise specific exception types for different error conditions
- **Provide context**: Include relevant values and context in error messages

### Guidelines

1. **Minimize public API surface**: Keep the public API small and focused. Use private helpers for implementation details.

2. **Document public APIs**: All public APIs must have complete docstrings following the [Docstrings](#docstrings) guidelines.

3. **Use `__all__` explicitly**: List all public symbols in `__all__` to clearly define the module's public API.

4. **Design for extensibility**: Consider future needs when designing APIs, but don't over-engineer.

5. **Follow Python conventions**: Adhere to Python naming and design conventions (PEP 8, PEP 20).

6. **Provide clear error messages**: When APIs fail, provide error messages that help users understand and fix the problem.

7. **Use type hints**: Provide type annotations for all public APIs to improve IDE support and documentation.

## CUDA-Specific Patterns

### GIL Management for CUDA Driver API Calls

**Always release the Global Interpreter Lock (GIL) when calling CUDA driver API functions.** This is critical for performance and thread safety.

#### Using `with nogil:` Blocks

Wrap CUDA driver API calls in `with nogil:` blocks:

```python
cdef cydriver.CUstream s
with nogil:
    HANDLE_RETURN(cydriver.cuStreamCreateWithPriority(&s, flags, prio))
self._handle = s
```

For multiple driver calls, group them in a single `with nogil:` block:

```python
cdef int high, low
with nogil:
    HANDLE_RETURN(cydriver.cuCtxGetStreamPriorityRange(&high, &low))
    HANDLE_RETURN(cydriver.cuStreamCreateWithPriority(&s, flags, prio))
```

#### Function-Level `nogil` Declaration

For functions that primarily call CUDA driver APIs, declare the function `nogil`:

```python
cdef int get_device_from_ctx(
        cydriver.CUcontext target_ctx, cydriver.CUcontext curr_ctx) except?cydriver.CU_DEVICE_INVALID nogil:
    """Get device ID from the given ctx."""
    cdef bint switch_context = (curr_ctx != target_ctx)
    cdef cydriver.CUcontext ctx
    cdef cydriver.CUdevice target_dev
    with nogil:
        if switch_context:
            HANDLE_RETURN(cydriver.cuCtxPopCurrent(&ctx))
            HANDLE_RETURN(cydriver.cuCtxPushCurrent(target_ctx))
        HANDLE_RETURN(cydriver.cuCtxGetDevice(&target_dev))
        if switch_context:
            HANDLE_RETURN(cydriver.cuCtxPopCurrent(&ctx))
            HANDLE_RETURN(cydriver.cuCtxPushCurrent(curr_ctx))
    return target_dev
```

#### Raising Exceptions from `nogil` Context

When raising exceptions from a `nogil` context, acquire the GIL first using `with gil:`:

```python
cpdef inline int _check_driver_error(cydriver.CUresult error) except?-1 nogil:
    if error == cydriver.CUresult.CUDA_SUCCESS:
        return 0
    cdef const char* name
    name_err = cydriver.cuGetErrorName(error, &name)
    if name_err != cydriver.CUresult.CUDA_SUCCESS:
        with gil:
            raise CUDAError(f"UNEXPECTED ERROR CODE: {error}")
    with gil:
        expl = DRIVER_CU_RESULT_EXPLANATIONS.get(int(error))
        if expl is not None:
            raise CUDAError(f"{name.decode()}: {expl}")
    # ... rest of error handling ...
```

#### Guidelines

1. **Always use `with nogil:` for CUDA driver calls**: Every call to `cydriver.*` functions should be within a `with nogil:` block.

2. **Use `HANDLE_RETURN` within `nogil` blocks**: The `HANDLE_RETURN` macro is designed to work in `nogil` contexts.

3. **Acquire GIL before raising exceptions**: When raising Python exceptions from a `nogil` context, use `with gil:` to acquire the GIL first.

4. **Group related driver calls**: If multiple driver calls are made sequentially, group them in a single `with nogil:` block for efficiency.

5. **Declare functions `nogil` when appropriate**: Functions that primarily call CUDA driver APIs and don't need Python object access should be declared `nogil` at the function level.

### Example

```python
cdef inline void DMR_close(DeviceMemoryResource self):
    if self._handle == NULL:
        return

    try:
        if self._mempool_owned:
            with nogil:
                HANDLE_RETURN(cydriver.cuMemPoolDestroy(self._handle))
    finally:
        self._dev_id = cydriver.CU_DEVICE_INVALID
        self._handle = NULL
        # ... cleanup ...
```

## Development Lifecycle

### Two-Phase Development Approach

When implementing new CUDA functionality, follow a two-phase development approach:

1. **Phase 1: Python Implementation with Tests**
   - Start with a pure Python implementation using the CUDA driver module
   - Write comprehensive tests to verify correctness
   - Ensure all tests pass before proceeding to Phase 2

2. **Phase 2: Cythonization for Performance**
   - After tests are passing, optimize by switching to `cydriver`
   - Add `with nogil:` blocks around CUDA driver API calls
   - Use `HANDLE_RETURN` macro for error handling
   - Verify tests still pass after optimization

### Phase 1: Initial Python Implementation

Begin with a straightforward Python implementation using the `driver` module from `cuda.core.experimental._utils.cuda_utils`:

```python
from cuda.core.experimental._utils.cuda_utils import driver
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
)

def copy_to(self, dst: Buffer = None, *, stream: Stream | GraphBuilder) -> Buffer:
    stream = Stream_accept(stream)
    cdef size_t src_size = self._size
    
    # ... validation logic ...
    
    err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
    raise_if_driver_error(err)
    return dst
```

**Benefits of starting with Python:**
- Faster iteration during development
- Easier debugging with Python stack traces
- Simpler error handling
- Focus on correctness before optimization

### Phase 2: Cythonization Process

Once tests are passing, optimize the implementation by:

1. **Switching to `cydriver`**: Replace `driver` module calls with direct `cydriver` calls
2. **Adding `with nogil:` blocks**: Wrap CUDA driver API calls to release the GIL
3. **Using `HANDLE_RETURN`**: Replace `raise_if_driver_error()` with the `HANDLE_RETURN` macro
4. **Casting stream handles**: Access the C-level stream handle for `cydriver` calls

#### Step-by-Step Conversion

**Step 1: Update imports**

```python
# Remove Python driver import
# from cuda.core.experimental._utils.cuda_utils import driver

# Add cydriver cimport
from cuda.bindings cimport cydriver

# Add HANDLE_RETURN
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN
```

**Step 2: Cast stream and extract C-level handle**

```python
stream = Stream_accept(stream)
cdef Stream s_stream = <Stream>stream
cdef cydriver.CUstream s = s_stream._handle
```

**Step 3: Wrap driver calls in `with nogil:` and use `HANDLE_RETURN`**

```python
# Before (Python driver):
err, = driver.cuMemcpyAsync(dst._ptr, self._ptr, src_size, stream.handle)
raise_if_driver_error(err)

# After (cydriver):
with nogil:
    HANDLE_RETURN(cydriver.cuMemcpyAsync(
        <cydriver.CUdeviceptr>dst._ptr,
        <cydriver.CUdeviceptr>self._ptr,
        src_size,
        s
    ))
```

**Step 4: Cast pointers to `cydriver.CUdeviceptr`**

All device pointers passed to `cydriver` functions must be cast to `cydriver.CUdeviceptr`:

```python
<cydriver.CUdeviceptr>self._ptr
```

### Complete Example: Before and After

**Before (Python driver implementation):**

```python
from cuda.core.experimental._utils.cuda_utils import driver
from cuda.core.experimental._utils.cuda_utils cimport (
    _check_driver_error as raise_if_driver_error,
)

def fill(self, value: int, width: int, *, stream: Stream | GraphBuilder):
    stream = Stream_accept(stream)
    cdef size_t buffer_size = self._size
    cdef unsigned char c_value8
    
    # Validation...
    if width == 1:
        c_value8 = <unsigned char>value
        N = buffer_size
        err, = driver.cuMemsetD8Async(self._ptr, c_value8, N, stream.handle)
        raise_if_driver_error(err)
```

**After (Cythonized with cydriver):**

```python
from cuda.bindings cimport cydriver
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN

def fill(self, value: int, width: int, *, stream: Stream | GraphBuilder):
    stream = Stream_accept(stream)
    cdef Stream s_stream = <Stream>stream
    cdef cydriver.CUstream s = s_stream._handle
    cdef size_t buffer_size = self._size
    cdef unsigned char c_value8
    
    # Validation...
    if width == 1:
        c_value8 = <unsigned char>value
        N = buffer_size
        with nogil:
            HANDLE_RETURN(cydriver.cuMemsetD8Async(
                <cydriver.CUdeviceptr>self._ptr, c_value8, N, s
            ))
```

### Guidelines

1. **Always write tests first**: Implement comprehensive tests before optimizing. This ensures correctness is established before performance improvements.

2. **Verify tests after optimization**: After converting to `cydriver`, run all tests to ensure behavior is unchanged.

3. **Don't skip Phase 1**: Even if you're confident about the implementation, starting with Python helps catch logic errors early.

4. **Performance benefits**: The Cythonized version eliminates Python overhead and releases the GIL, providing significant performance improvements for CUDA operations.

5. **Consistent pattern**: Follow this pattern for all new CUDA driver API wrappers to maintain consistency across the codebase.

6. **Error handling**: The `HANDLE_RETURN` macro is designed to work in `nogil` contexts and will automatically raise appropriate exceptions when needed.

## Copyright and Licensing

All source files in `cuda/core/experimental` must include a copyright header at the top of the file using the SPDX format.

### Required Header Format

Every `.py`, `.pyx`, and `.pxd` file must begin with the following header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
```

### Guidelines

1. **Placement**: The copyright header must be the first lines of the file, before any imports or other code.

2. **Blank Lines**: Include a blank line between the copyright notice and the license identifier, and another blank line after the license identifier before the code begins.

3. **Year Range**:
   - The beginning year reflects the year the file was first added to the repository.
   - The end year reflects the most recent year in which the file was modified.
   - For new files, use a single year (e.g., `2025`) or the current year range if created mid-year.
   - Update the end year when making modifications to existing files.

4. **Consistency**: All files must use the same copyright text and license identifier (`Apache-2.0`).

5. **SPDX Format**: The header uses the SPDX (Software Package Data Exchange) format, which is a standard way to communicate license and copyright information.

### Example

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ... rest of the file ...
```
