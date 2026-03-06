# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

"""Integration tests for explicit CUDA graph construction.

Three test scenarios exercise complementary subsets of node types:

test_heat_diffusion
    1D heat bar evolving toward steady state via finite differences.
    Exercises: AllocNode, FreeNode, MemsetNode, ChildGraphNode,
    EmptyNode, EventRecordNode, EventWaitNode, WhileNode, KernelNode,
    MemcpyNode, HostCallbackNode.

test_bisection_root
    Find sqrt(2) by bisecting f(x) = x^2 - 2 on [0, 2], with an
    optional Newton polish step.
    Exercises: IfElseNode (interval halving), IfNode (refinement
    guard), WhileNode, KernelNode, AllocNode, MemsetNode, MemcpyNode,
    HostCallbackNode, FreeNode, EmptyNode.

test_switch_dispatch
    Apply one of four element-wise transforms selected at graph
    creation time via a switch condition.
    Exercises: SwitchNode, KernelNode, AllocNode, MemsetNode,
    MemcpyNode, FreeNode.

Together the three tests cover all 14 explicit-graph node types.
"""

import ctypes

import numpy as np
import pytest

from cuda.core import Device, EventOptions, LaunchConfig, Program, ProgramOptions
from cuda.core._graph._graphdef import GraphDef
from cuda.core._utils.cuda_utils import driver, handle_return

SIZEOF_FLOAT = 4
SIZEOF_INT = 4

# ===================================================================
# Kernel sources
# ===================================================================

_COND_PREAMBLE = r"""
extern "C" __device__ __cudart_builtin__ void CUDARTAPI
cudaGraphSetConditional(cudaGraphConditionalHandle handle,
                        unsigned int value);
"""

_HEAT_KERNEL_SOURCE = (
    _COND_PREAMBLE
    + r"""
extern "C" __global__
void heat_step(float* u_next, const float* u_curr, int N, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1)
        u_next[i] = u_curr[i];
    else
        u_next[i] = u_curr[i]
                   + alpha * (u_curr[i-1] - 2.0f * u_curr[i] + u_curr[i+1]);
}

extern "C" __global__
void countdown(cudaGraphConditionalHandle handle, int* counter) {
    int c = atomicSub(counter, 1);
    cudaGraphSetConditional(handle, (c > 1) ? 1u : 0u);
}
"""
)

_BISECT_KERNEL_SOURCE = (
    _COND_PREAMBLE
    + r"""
extern "C" __global__
void bisect_eval(float* a, float* b,
                 cudaGraphConditionalHandle ie_cond) {
    float mid = (*a + *b) * 0.5f;
    float fm = mid * mid - 2.0f;
    cudaGraphSetConditional(ie_cond, (fm > 0.0f) ? 1u : 0u);
}

extern "C" __global__
void update_hi(float* a, float* b) {
    *b = (*a + *b) * 0.5f;
}

extern "C" __global__
void update_lo(float* a, float* b) {
    *a = (*a + *b) * 0.5f;
}

extern "C" __global__
void countdown(cudaGraphConditionalHandle handle, int* counter) {
    int c = atomicSub(counter, 1);
    cudaGraphSetConditional(handle, (c > 1) ? 1u : 0u);
}

extern "C" __global__
void check_refine(float* a, float* b,
                  cudaGraphConditionalHandle if_cond) {
    float mid = (*a + *b) * 0.5f;
    float fm = mid * mid - 2.0f;
    float abs_fm = fm < 0.0f ? -fm : fm;
    cudaGraphSetConditional(if_cond, (abs_fm > 1e-10f) ? 1u : 0u);
}

extern "C" __global__
void newton_refine(float* a, float* b) {
    float mid = (*a + *b) * 0.5f;
    float refined = mid - (mid * mid - 2.0f) / (2.0f * mid);
    *a = refined;
    *b = refined;
}
"""
)

_SWITCH_KERNEL_SOURCE = r"""
extern "C" __global__
void negate_it(int* x) { *x = -(*x); }

extern "C" __global__
void double_it(int* x) { *x = 2 * (*x); }

extern "C" __global__
void square_it(int* x) { *x = (*x) * (*x); }
"""

# ===================================================================
# Compilation helpers
# ===================================================================


def _nvrtc_opts():
    arch = "".join(f"{i}" for i in Device().compute_capability)
    return ProgramOptions(std="c++17", arch=f"sm_{arch}")


def _compile_heat_kernels():
    prog = Program(_HEAT_KERNEL_SOURCE, code_type="c++", options=_nvrtc_opts())
    try:
        mod = prog.compile(
            "cubin",
            name_expressions=("heat_step", "countdown"),
        )
    except Exception:
        pytest.skip("NVRTC does not support cudaGraphConditionalHandle")
    return mod.get_kernel("heat_step"), mod.get_kernel("countdown")


def _compile_bisect_kernels():
    names = (
        "bisect_eval",
        "update_hi",
        "update_lo",
        "countdown",
        "check_refine",
        "newton_refine",
    )
    prog = Program(_BISECT_KERNEL_SOURCE, code_type="c++", options=_nvrtc_opts())
    try:
        mod = prog.compile("cubin", name_expressions=names)
    except Exception:
        pytest.skip("NVRTC does not support cudaGraphConditionalHandle")
    return tuple(mod.get_kernel(n) for n in names)


def _compile_switch_kernels():
    names = ("negate_it", "double_it", "square_it")
    prog = Program(_SWITCH_KERNEL_SOURCE, code_type="c++", options=_nvrtc_opts())
    mod = prog.compile("cubin", name_expressions=names)
    return tuple(mod.get_kernel(n) for n in names)


# ===================================================================
# Test 1 — Heat diffusion (WhileNode, ChildGraphNode, EventNodes, …)
#
#   alloc(curr) ─ memset(0) ──┐
#   alloc(next) ─ memset(0) ──┼─ join ─ embed(bc) ─ rec(start) ─ WHILE ──┐
#   alloc(ctr)  ─ memset(50) ─┘                                          │
#   ┌─────────────────────────────────────────────────────────────────────┘
#   └─ wait(start) ─ rec(end) ─ memcpy(→host) ─ callback
#      ─ free(curr) ─ free(next) ─ free(ctr)
#
#   bc graph:     memset(T_LEFT) ─ memset(T_RIGHT)
#   while body:   heat_step ─ memcpy(curr ← next) ─ countdown
# ===================================================================

_HEAT_N = 32
_HEAT_T_LEFT = np.float32(100.0)
_HEAT_T_RIGHT = np.float32(0.0)
_HEAT_ALPHA = np.float32(0.4)
_HEAT_ITERS = 50


def _heat_reference():
    """Compute the reference heat solution on the host (NumPy)."""
    u = np.zeros(_HEAT_N, dtype=np.float32)
    u[0] = _HEAT_T_LEFT
    u[-1] = _HEAT_T_RIGHT
    u_next = np.empty_like(u)
    for _ in range(_HEAT_ITERS):
        u_next[0] = u[0]
        u_next[-1] = u[-1]
        u_next[1:-1] = u[1:-1] + _HEAT_ALPHA * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        u, u_next = u_next, u
    return u


def test_heat_diffusion(init_cuda):
    """1D heat-bar simulation exercising most explicit-graph node types."""
    dev = Device()

    if dev.compute_capability < (9, 0):
        pytest.skip("Conditional nodes require compute capability >= 9.0")

    k_heat, k_countdown = _compile_heat_kernels()

    host_ptr = handle_return(driver.cuMemAllocHost(_HEAT_N * SIZEOF_FLOAT))

    try:
        _run_heat_graph(dev, k_heat, k_countdown, host_ptr)
    finally:
        handle_return(driver.cuMemFreeHost(host_ptr))


def _run_heat_graph(dev, k_heat, k_countdown, host_ptr):
    """Build, instantiate, launch, and verify the heat-diffusion graph."""

    # Definitions
    g = GraphDef()
    condition = g.create_condition(default_value=1)
    event_start = dev.create_event(EventOptions(enable_timing=True))
    event_end = dev.create_event(EventOptions(enable_timing=True))
    results = {}

    def capture_result():
        arr = (ctypes.c_float * _HEAT_N).from_address(host_ptr)
        results["data"] = np.array(arr, copy=True)

    block = min(_HEAT_N, 256)
    grid = (_HEAT_N + block - 1) // block
    heat_cfg = LaunchConfig(grid=grid, block=block)
    tick_cfg = LaunchConfig(grid=1, block=1)

    # fmt: off
    # Phase 1 — Allocate device memory
    a_curr = g.alloc(_HEAT_N * SIZEOF_FLOAT)
    a_next = g.alloc(_HEAT_N * SIZEOF_FLOAT)
    a_ctr  = g.alloc(SIZEOF_INT)

    # Phase 2 — Initialise buffers
    m_curr = a_curr.memset(a_curr.dptr, 0, _HEAT_N * SIZEOF_FLOAT)
    m_next = a_next.memset(a_next.dptr, 0, _HEAT_N * SIZEOF_FLOAT)
    m_ctr  = a_ctr.memset(a_ctr.dptr, np.int32(_HEAT_ITERS), 1)

    # Phase 3 — Boundary conditions (child graph)
    bc = GraphDef() \
         .memset(a_curr.dptr, np.float32(_HEAT_T_LEFT), 1) \
         .memset(a_curr.dptr + (_HEAT_N - 1) * SIZEOF_FLOAT,
                 np.float32(_HEAT_T_RIGHT), 1) \
         .graph
    p = g.join(m_curr, m_next, m_ctr) \
         .embed(bc) \
         .record_event(event_start)

    # Phase 4 — Iterate
    loop = p.while_loop(condition)
    loop.body.launch(heat_cfg, k_heat, a_next.dptr, a_curr.dptr,
                     np.int32(_HEAT_N), _HEAT_ALPHA) \
             .memcpy(a_curr.dptr, a_next.dptr, _HEAT_N * SIZEOF_FLOAT) \
             .launch(tick_cfg, k_countdown, condition.handle, a_ctr.dptr)

    # Phase 5 — After loop: timing end, readback, verify, free memory
    loop.wait_event(event_start) \
        .record_event(event_end) \
        .memcpy(host_ptr, a_curr.dptr, _HEAT_N * SIZEOF_FLOAT) \
        .callback(capture_result) \
        .free(a_curr.dptr) \
        .free(a_next.dptr) \
        .free(a_ctr.dptr)
    # fmt: on

    # Phase 6 — Instantiate, launch, verify
    graph = g.instantiate()
    stream = dev.create_stream()
    graph.launch(stream)
    stream.sync()

    assert "data" in results, "Host callback did not execute"
    np.testing.assert_allclose(results["data"], _heat_reference(), rtol=1e-5)


# ===================================================================
# Test 2 — Bisection root finder (IfElseNode, IfNode)
#
#   Find sqrt(2) by bisecting f(x) = x^2 - 2 on [0, 2].
#
#   alloc(a) ─ memset(0.0) ──┐
#   alloc(b) ─ memset(2.0) ──┼─ join ─ WHILE(while_cond) ──────────────────┐
#   alloc(ctr) ─ memset(20) ─┘                                             │
#   ┌───────────────────────────────────────────────────────────────────────┘
#   └─ check_refine ─ IF(if_cond) ─ memcpy(→host) ─ callback
#                      └─ body: newton_refine
#      ─ free(a) ─ free(b) ─ free(ctr)
#
#   while body:
#     bisect_eval ─ IF_ELSE(ie_cond) ─ countdown
#                   ├─ then: update_hi (b = mid)    [f(mid) > 0]
#                   └─ else: update_lo (a = mid)    [f(mid) ≤ 0]
# ===================================================================

_BISECT_ITERS = 20


def test_bisection_root(init_cuda):
    """Bisection search for sqrt(2) with optional Newton refinement.

    Exercises IfElseNode (interval halving) and IfNode (refinement guard).
    """
    dev = Device()

    if dev.compute_capability < (9, 0):
        pytest.skip("Conditional nodes require compute capability >= 9.0")

    k_eval, k_hi, k_lo, k_cd, k_check, k_newton = _compile_bisect_kernels()

    host_ptr = handle_return(driver.cuMemAllocHost(SIZEOF_FLOAT))

    try:
        _run_bisection_graph(dev, k_eval, k_hi, k_lo, k_cd, k_check, k_newton, host_ptr)
    finally:
        handle_return(driver.cuMemFreeHost(host_ptr))


def _run_bisection_graph(dev, k_eval, k_hi, k_lo, k_cd, k_check, k_newton, host_ptr):
    """Build, instantiate, launch, and verify the bisection graph."""

    # Definitions
    g = GraphDef()
    cfg = LaunchConfig(grid=1, block=1)
    results = {}

    def capture_result():
        results["root"] = ctypes.c_float.from_address(host_ptr).value

    # fmt: off
    # Allocate and initialise: a = 0.0, b = 2.0, counter = ITERS
    a   = g.alloc(SIZEOF_FLOAT)
    b   = g.alloc(SIZEOF_FLOAT)
    ctr = g.alloc(SIZEOF_INT)

    p = g.join(a.memset(a.dptr, np.float32(0.0), 1),
               b.memset(b.dptr, np.float32(2.0), 1),
               ctr.memset(ctr.dptr, np.int32(_BISECT_ITERS), 1))

    # While loop: bisection iterations
    while_cond = g.create_condition(default_value=1)
    ie_cond    = g.create_condition(default_value=0)
    loop = p.while_loop(while_cond)

    ie = loop.body.launch(cfg, k_eval, a.dptr, b.dptr, ie_cond.handle) \
                  .if_else(ie_cond)
    ie.then.launch(cfg, k_hi, a.dptr, b.dptr)
    ie.else_.launch(cfg, k_lo, a.dptr, b.dptr)
    ie.launch(cfg, k_cd, while_cond.handle, ctr.dptr)

    # Post-loop: Newton refinement (IfNode), readback, free
    if_cond = g.create_condition(default_value=0)
    if_node = loop.launch(cfg, k_check, a.dptr, b.dptr, if_cond.handle) \
                  .if_cond(if_cond)
    if_node.then.launch(cfg, k_newton, a.dptr, b.dptr)

    if_node.memcpy(host_ptr, a.dptr, SIZEOF_FLOAT) \
           .callback(capture_result) \
           .free(a.dptr) \
           .free(b.dptr) \
           .free(ctr.dptr)
    # fmt: on

    # Instantiate, launch, verify
    graph = g.instantiate()
    stream = dev.create_stream()
    graph.launch(stream)
    stream.sync()

    assert "root" in results, "Host callback did not execute"
    np.testing.assert_allclose(
        results["root"],
        np.sqrt(np.float32(2.0)),
        rtol=1e-6,
    )


# ===================================================================
# Test 3 — Switch dispatch (SwitchNode)
#
#   A mode value (0-3) selects one of four transforms on a scalar:
#
#   alloc(x) ─ memset(42) ─ SWITCH(mode, 4)
#                            ├─ 0: negate(x)
#                            ├─ 1: double(x)
#                            ├─ 2: square(x)
#                            └─ 3: (identity)
#                           ─ memcpy(→host) ─ free(x)
# ===================================================================

_SWITCH_VALUE = 42


@pytest.mark.parametrize(
    "mode, expected",
    [
        (0, -_SWITCH_VALUE),
        (1, 2 * _SWITCH_VALUE),
        (2, _SWITCH_VALUE * _SWITCH_VALUE),
        (3, _SWITCH_VALUE),
    ],
)
def test_switch_dispatch(init_cuda, mode, expected):
    """Runtime kernel selection via SwitchNode."""
    dev = Device()

    if dev.compute_capability < (9, 0):
        pytest.skip("Conditional nodes require compute capability >= 9.0")

    k_negate, k_double, k_square = _compile_switch_kernels()

    host_ptr = handle_return(driver.cuMemAllocHost(SIZEOF_INT))

    try:
        _run_switch_graph(dev, mode, k_negate, k_double, k_square, host_ptr)

        result = ctypes.c_int.from_address(host_ptr).value
        assert result == expected
    finally:
        handle_return(driver.cuMemFreeHost(host_ptr))


def _run_switch_graph(dev, mode, k_negate, k_double, k_square, host_ptr):
    """Build, instantiate, launch, and verify the switch-dispatch graph."""
    g = GraphDef()
    cfg = LaunchConfig(grid=1, block=1)

    # fmt: off
    x = g.alloc(SIZEOF_INT)
    sw_cond = g.create_condition(default_value=mode)
    sw = x.memset(x.dptr, np.int32(_SWITCH_VALUE), 1) \
          .switch(sw_cond, 4)

    sw.branches[0].launch(cfg, k_negate, x.dptr)
    sw.branches[1].launch(cfg, k_double, x.dptr)
    sw.branches[2].launch(cfg, k_square, x.dptr)
    # branch 3: identity (no kernel — value unchanged)

    sw.memcpy(host_ptr, x.dptr, SIZEOF_INT) \
      .free(x.dptr)
    # fmt: on

    graph = g.instantiate()
    stream = dev.create_stream()
    graph.launch(stream)
    stream.sync()
