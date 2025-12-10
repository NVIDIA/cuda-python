# Buffer.fill() Redesign Proposal

**Issue**: [#1345 - Revisit `Buffer.fill()`](https://github.com/NVIDIA/cuda-python/issues/1345)  
**Author**: Andy Jost  
**Date**: December 10, 2025

## Background

PR #1318 implemented `Buffer.fill(value, width, *, stream)` but was merged before review feedback was addressed. This document proposes a simplified API based on that feedback.

## Proposed API

```python
def fill(self, value, *, stream: Stream | GraphBuilder):
    """Fill buffer with a repeating byte pattern.
    
    Parameters
    ----------
    value : int or buffer-protocol object
        - int: Must be in range [0, 256). Converted to 1 byte.
        - buffer-protocol object: Must be 1, 2, or 4 bytes.
    stream : Stream | GraphBuilder
        Stream for the asynchronous fill operation.
    
    Raises
    ------
    TypeError
        If value is not an int and does not support the buffer protocol.
    ValueError
        If value byte length is not 1, 2, or 4.
        If buffer size is not divisible by value byte length.
    OverflowError
        If int value is outside [0, 256).
    """
```

## Implementation

```python
def get_fill_pattern(value):
    if isinstance(value, int):
        return value.to_bytes(1, 'little')  # Raises OverflowError if not in [0, 256)
    mv = memoryview(value)
    return mv.tobytes()

pattern = get_fill_pattern(value)
L = len(pattern)
if L not in (1, 2, 4):
    raise ValueError(f"value must be 1, 2, or 4 bytes, got {L}")
if buffer_size % L != 0:
    raise ValueError(f"buffer size ({buffer_size}) must be divisible by {L}")

# Call appropriate cuMemsetD{8,16,32}Async based on L
```

## Examples

```python
# Byte fill (1 byte)
buffer.fill(0, stream=stream)              # Zero memory
buffer.fill(0xFF, stream=stream)           # Fill with 0xFF

# Multi-byte fill via numpy scalars
buffer.fill(np.uint16(0x1234), stream=stream)      # 2-byte pattern
buffer.fill(np.float32(1.0), stream=stream)        # 4-byte pattern

# Raw bytes
buffer.fill(b'\xDE\xAD\xBE\xEF', stream=stream)    # 4-byte pattern
```

## Changes from Current API

| Current | Proposed |
|---------|----------|
| `fill(value, width, *, stream)` | `fill(value, *, stream)` |
| Explicit `width` parameter | Width inferred from value |
| Only accepts `int` | Accepts `int` or buffer-protocol objects |
