# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host allocation failure in the C++ handle layer must raise, never terminate.

``std::bad_alloc`` that escapes a ``noexcept`` function calls ``std::terminate``
and aborts the interpreter. These tests make the Nth C++ allocation on a thread
throw ``std::bad_alloc`` by preloading a tiny ``operator new`` replacement into a
child process, so an abort shows up as a non-zero exit status of the child
instead of taking down the test run.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the allocation shim is preloaded with LD_PRELOAD"
)

# The Nth C++ allocation made by the arming thread throws std::bad_alloc. Python
# allocates through its own allocators, so only C++ allocations are affected.
SHIM_SOURCE = textwrap.dedent(
    """
    #include <cstdlib>
    #include <new>

    static thread_local long countdown = 0;

    extern "C" void arm_bad_alloc(long n) { countdown = n; }

    void* operator new(std::size_t size) {
        if (countdown > 0 && --countdown == 0) {
            throw std::bad_alloc();
        }
        if (void* p = std::malloc(size ? size : 1)) {
            return p;
        }
        throw std::bad_alloc();
    }
    void operator delete(void* p) noexcept { std::free(p); }
    void operator delete(void* p, std::size_t) noexcept { std::free(p); }
    """
)

# Runs in the child. argv[1] is the shim path, argv[2] the scenario.
CHILD = textwrap.dedent(
    """
    import ctypes
    import sys

    shim = ctypes.CDLL(sys.argv[1])
    scenario = sys.argv[2]

    from cuda.core import Buffer, Device, MemoryResource
    from cuda.core._stream import default_stream
    from cuda.core._utils.cuda_utils import driver, handle_return


    class DeviceMR(MemoryResource):
        def allocate(self, size, *, stream=None):
            raise NotImplementedError

        def deallocate(self, ptr, size, *, stream=None):
            handle_return(driver.cuMemFree(ptr))

        is_device_accessible = True
        is_host_accessible = False
        device_id = 0


    def arm(n):
        shim.arm_bad_alloc(ctypes.c_long(n))


    Device(0).set_current()
    mr = DeviceMR()
    outcomes = set()

    if scenario == "set_deallocation_stream":
        buf = Buffer.from_handle(handle_return(driver.cuMemAlloc(64)), 64, mr=mr)
        token = default_stream()
        arm(1)
        try:
            buf.set_deallocation_stream(token)
        except MemoryError:
            outcomes.add("MemoryError")
        else:
            outcomes.add("ok")
        finally:
            arm(0)
        buf.close()
    elif scenario == "from_handle":
        # Fail each successive allocation on the way through Buffer.from_handle.
        for n in range(1, 9):
            ptr = handle_return(driver.cuMemAlloc(64))
            arm(n)
            try:
                buf = Buffer.from_handle(ptr, 64, mr=mr)
            except MemoryError:
                outcomes.add("MemoryError")
            else:
                outcomes.add("ok")
                buf.close()
            finally:
                arm(0)
    else:
        raise SystemExit("unknown scenario " + scenario)

    print(",".join(sorted(outcomes)))
    """
)


@pytest.fixture(scope="module")
def alloc_shim(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler available to build the allocation shim")
    workdir = tmp_path_factory.mktemp("alloc_shim")
    source = workdir / "shim.cpp"
    source.write_text(SHIM_SOURCE)
    shim = workdir / "libshim.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O1", "-o", str(shim), str(source)], check=True)  # noqa: S603
    return shim


def run_child(shim, scenario):
    env = dict(os.environ)
    preload = [str(shim)]
    if env.get("LD_PRELOAD"):
        preload.append(env["LD_PRELOAD"])
    env["LD_PRELOAD"] = " ".join(preload)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD, str(shim), scenario],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_set_deallocation_stream_raises_memory_error_instead_of_terminating(init_cuda, alloc_shim):
    """Binding a default-stream token allocates; failure there must be a MemoryError.

    Before the fix, ``set_deallocation_stream`` and ``make_deallocation_stream``
    were ``noexcept``, so the ``std::bad_alloc`` called ``std::terminate`` and the
    child died with SIGABRT.
    """
    result = run_child(alloc_shim, "set_deallocation_stream")
    assert result.returncode == 0, f"child exited with {result.returncode}\n{result.stderr}"
    assert result.stdout.strip() == "MemoryError"


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_buffer_from_handle_survives_allocation_failure(init_cuda, alloc_shim):
    """Failing each allocation in ``Buffer.from_handle`` raises MemoryError or succeeds, never aborts."""
    result = run_child(alloc_shim, "from_handle")
    assert result.returncode == 0, f"child exited with {result.returncode}\n{result.stderr}"
    assert "MemoryError" in result.stdout.strip().split(",")
