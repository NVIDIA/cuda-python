# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import gc
import os
import sys
import pytest
from cuda import cuda
import cupy as cp

class SampleTestError(Exception):
    pass

def parse_python_script(filepath):
    if filepath.endswith('.py'):
        with open(filepath, "r", encoding='utf-8') as f:
            script = f.read()
    else:
        raise ValueError(f"{filepath} not supported")
    return script


def run_example(samples_path, filename, env=None):
    fullpath = os.path.join(samples_path, filename)
    script = parse_python_script(fullpath)
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        SYS_PATH_BACKUP = sys.path.copy()
        sys.path.append(samples_path)
        exec(script, env if env is not None else {})
    except ImportError as e:
        # for samples requiring any of optional dependencies
        for m in ('cupy',):
            if f"No module named '{m}'" in str(e):
                pytest.skip(f'{m} not installed, skipping related tests')
                break
        else:
            raise
    except Exception as e:
            msg = "\n"
            msg += f'Got error ({filename}):\n'
            msg += str(e)
            raise SampleTestError(msg) from e
    finally:
        sys.path = SYS_PATH_BACKUP
        sys.argv = old_argv
        # further reduce the memory watermark
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()