# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import sys

import pytest


class SampleTestError(Exception):
    pass


def parse_python_script(filepath):
    if not filepath.endswith(".py"):
        raise ValueError(f"{filepath} not supported")
    with open(filepath, encoding="utf-8") as f:
        script = f.read()
    return script


def run_example(samples_path, filename, env=None):
    fullpath = os.path.join(samples_path, filename)
    script = parse_python_script(fullpath)
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        old_sys_path = sys.path.copy()
        sys.path.append(samples_path)
        # TODO: Refactor the examples to give them a common callable `main()` to avoid needing to use exec here?
        exec(script, env if env else {})  # noqa: S102
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
        pytest.skip(f"skip {filename}")
    except Exception as e:
        msg = "\n"
        msg += f"Got error ({filename}):\n"
        msg += str(e)
        raise SampleTestError(msg) from e
    finally:
        sys.path = old_sys_path
        sys.argv = old_argv
        # further reduce the memory watermark
        gc.collect()
