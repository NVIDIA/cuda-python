# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated shim package.

These helpers moved to ``cuda_python_test_helpers`` in #2384 (2026-08-04).
This package is kept for one cuda-core release cycle so that already-released
cuda-core test trees (<=1.1.x), which still import from here, keep working
against newer cuda-bindings wheels. See #2725.

.. deprecated:: cuda-core-1.2
    Remove once cuda-core >= 1.2 is the oldest supported release.
"""

import importlib
import importlib.util
import pathlib
import sys

# Private alias under which a modern (post-#2384) cuda_python_test_helpers
# source checkout is loaded, when the real `cuda_python_test_helpers` name is
# already claimed by an older, incompatible copy. See _import_current below.
_CURRENT_ALIAS = "_cuda_bindings_test_helpers_current"


def _find_current_test_helpers_package():
    """Locate and load a post-#2384 cuda_python_test_helpers source checkout
    (one that actually contains arch_check.py) under a private alias name,
    bypassing sys.modules entirely.

    Don't call .resolve(): resolving symlinks can make a parent point
    somewhere other than the monorepo root if a sub-directory is symlinked.

    This module's own __file__ is not a reliable anchor either: this shim is
    imported from an installed cuda-bindings wheel (e.g. site-packages), not
    necessarily from a monorepo checkout. So, in addition to walking up from
    __file__ (dev/editable checkouts), we also walk up from the current
    working directory -- this matters for the released-cuda-core compat job,
    where pytest runs from a checkout of the released cuda-core tag (e.g.
    ``.../cuda-core-released/cuda_core``), which has its own sibling
    ``cuda_python_test_helpers`` one level up, separate from the one checked
    out alongside cuda-bindings itself.
    """
    if _CURRENT_ALIAS in sys.modules:
        return sys.modules[_CURRENT_ALIAS]

    candidates = [*pathlib.Path(__file__).parents, pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        pkg_dir = candidate / "cuda_python_test_helpers" / "cuda_python_test_helpers"
        init_file = pkg_dir / "__init__.py"
        if init_file.is_file() and (pkg_dir / "arch_check.py").is_file():
            spec = importlib.util.spec_from_file_location(
                _CURRENT_ALIAS, init_file, submodule_search_locations=[str(pkg_dir)]
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[_CURRENT_ALIAS] = module
            spec.loader.exec_module(module)
            return module

    return None


def _import_current(submodule_name: str):
    """Import cuda_python_test_helpers.<submodule_name>.

    Prefers whatever is already importable under the real
    ``cuda_python_test_helpers`` name. Falls back to a modern (post-#2384)
    copy loaded under a private alias if the real name resolves to an older,
    incompatible copy -- e.g. released cuda-core <= 1.1.1 (tagged 2026-07-29,
    six days before #2384 landed) ships its own ``tests/conftest.py`` that
    puts a pre-#2384 ``cuda_python_test_helpers`` (predating arch_check.py /
    mempool.py / pep723.py living there at all) on sys.path and caches it in
    sys.modules under the real name before this shim ever runs.
    """
    try:
        return importlib.import_module(f"cuda_python_test_helpers.{submodule_name}")
    except ImportError:
        pass

    module = _find_current_test_helpers_package()
    if module is None:
        raise ModuleNotFoundError(
            f"cuda_python_test_helpers.{submodule_name} is required by cuda.bindings._test_helpers "
            "but is not installed, and no source checkout containing it was found near this file "
            "or the current working directory."
        )
    return importlib.import_module(f"{module.__name__}.{submodule_name}")
