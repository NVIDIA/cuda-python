# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Marks `tests/memory` as a package. Under pytest's default "prepend" import
# mode, a test file's sys.path entry is the first parent directory *without* an
# __init__.py. Adding this file moves that entry up from `tests/memory` to
# `tests/`, which is what we want for two reasons:
#
#   * `tests/memory` is no longer on sys.path, so the `from conftest import ...`
#     in test_managed_ops.py resolves to the root tests/conftest.py. Without it
#     the local memory/conftest.py wins and the import fails with ImportError.
#   * Modules are named `memory.test_x` rather than `test_x`, so a file basename
#     reused under another directory cannot collide.
#
# Both follow from module identity tracking the directory layout instead of
# whichever directory happens to land on sys.path. `tests/memory_ipc/` carries
# an __init__.py for the same reasons.
