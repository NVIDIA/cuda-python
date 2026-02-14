# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib.util
import sys
from pathlib import Path

import pytest


class SampleTestError(Exception):
    pass


def run_example(parent_dir: str, rel_path_to_example: str, env=None) -> None:
    fullpath = Path(parent_dir) / rel_path_to_example
    module_name = fullpath.stem

    old_sys_path = sys.path.copy()
    old_argv = sys.argv

    try:
        sys.path.append(parent_dir)
        sys.argv = [str(fullpath)]

        # Collect metadata for file 'module_name' located at 'fullpath'.
        # CASE: file does not exist -> spec is none.
        # CASE: file is not .py -> spec is none.
        # CASE: file does not have proper loader (module.spec.__loader__) -> spec.loader is none.
        spec = importlib.util.spec_from_file_location(module_name, fullpath)

        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {rel_path_to_example}")

        # Otherwise convert the spec to a module, then run the module.
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # This runs top-level code.
        # CASE: exec() -> top-level code is implicitly run.
        spec.loader.exec_module(module)

        # CASE: main() -> we find main and call it below.
        if hasattr(module, "main"):
            module.main()

    except ImportError as e:
        # for samples requiring any of optional dependencies
        for m in ("cupy", "torch"):
            if f"No module named '{m}'" in str(e):
                pytest.skip(f"{m} not installed, skipping related tests")
                break
        else:
            raise
    except SystemExit:
        # for samples that early return due to any missing requirements
        pytest.skip(f"skip {rel_path_to_example}")
    except Exception as e:
        msg = "\n"
        msg += f"Got error ({rel_path_to_example}):\n"
        msg += str(e)
        raise SampleTestError(msg) from e
    finally:
        sys.path = old_sys_path
        sys.argv = old_argv

        # further reduce the memory watermark
        sys.modules.pop(module_name, None)
        gc.collect()
