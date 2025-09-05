# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""
Tests whether importing a specific module leads to cyclical imports.

See https://github.com/NVIDIA/cuda-python/issues/789 for more info.
"""

import argparse

orig_import = __builtins__.__import__

import_stack = []


def import_hook(name, globals=None, locals=None, fromlist=(), *args, **kwargs):
    """Approximate a custom import system that does not allow import cycles."""

    stack_entry = (tuple(fromlist) if fromlist is not None else None, name)
    if stack_entry in import_stack and name.startswith("cuda.bindings."):
        raise ImportError(f"Import cycle detected: {stack_entry}, stack: {import_stack}")
    import_stack.append(stack_entry)
    try:
        res = orig_import(name, globals, locals, fromlist, *args, **kwargs)
    finally:
        import_stack.pop()
    return res


__builtins__.__import__ = import_hook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "module",
        type=str,
    )
    args = parser.parse_args()

    __import__(args.module)
