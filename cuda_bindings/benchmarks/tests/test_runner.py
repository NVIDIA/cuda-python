# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_MAIN_PATH = REPO_ROOT / "cuda_bindings/benchmarks/runner/main.py"
BENCH_LAUNCH_PATH = REPO_ROOT / "cuda_bindings/benchmarks/benchmarks/bench_launch.py"


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load test module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner_main(monkeypatch):
    pyperf_module = types.ModuleType("pyperf")

    class FakeRunner:
        def bench_time_func(self, *_args, **_kwargs) -> None:
            raise AssertionError("FakeRunner should not be used in these tests")

    pyperf_module.Runner = FakeRunner
    monkeypatch.setitem(sys.modules, "pyperf", pyperf_module)
    return load_module_from_path("test_cuda_bindings_bench_runner_main", RUNNER_MAIN_PATH)


def load_bench_launch(monkeypatch, calls: list[tuple]):
    pointer_values = itertools.count(1000)

    runtime_module = types.ModuleType("runner.runtime")

    def alloc_persistent(size: int) -> int:
        calls.append(("alloc_persistent", size))
        return next(pointer_values)

    def assert_drv(err) -> None:
        calls.append(("assert_drv", err))
        assert err == 0

    def compile_and_load(source: str) -> str:
        calls.append(("compile_and_load", source))
        return "module"

    runtime_module.alloc_persistent = alloc_persistent
    runtime_module.assert_drv = assert_drv
    runtime_module.compile_and_load = compile_and_load

    runner_module = types.ModuleType("runner")
    runner_module.runtime = runtime_module

    driver_module = types.ModuleType("cuda.bindings.driver")

    class FakeCUresult:
        CUDA_SUCCESS = 0

    class FakeCUstreamFlags:
        CU_STREAM_NON_BLOCKING = types.SimpleNamespace(value=1)

    def cuModuleGetFunction(module, name):
        calls.append(("cuModuleGetFunction", module, name))
        return 0, name

    def cuStreamCreate(flags):
        calls.append(("cuStreamCreate", flags))
        return 0, "stream"

    def cuLaunchKernel(*args):
        calls.append(("cuLaunchKernel", args))
        return 0

    driver_module.CUresult = FakeCUresult
    driver_module.CUstream_flags = FakeCUstreamFlags
    driver_module.cuModuleGetFunction = cuModuleGetFunction
    driver_module.cuStreamCreate = cuStreamCreate
    driver_module.cuLaunchKernel = cuLaunchKernel

    cuda_module = types.ModuleType("cuda")
    bindings_module = types.ModuleType("cuda.bindings")
    bindings_module.driver = driver_module
    cuda_module.bindings = bindings_module

    monkeypatch.setitem(sys.modules, "runner", runner_module)
    monkeypatch.setitem(sys.modules, "runner.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver_module)

    return load_module_from_path("test_cuda_bindings_bench_launch", BENCH_LAUNCH_PATH)


def test_discover_benchmarks_is_lazy(monkeypatch, tmp_path):
    runner_main = load_runner_main(monkeypatch)

    marker_path = tmp_path / "imported.txt"
    bench_path = tmp_path / "bench_lazy.py"
    bench_path.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                f"Path({str(marker_path)!r}).write_text('imported')",
                "",
                "def helper() -> float:",
                "    return 0.0",
                "",
                "def bench_visible(loops: int) -> float:",
                "    return loops + 0.5",
                "",
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner_main, "BENCH_DIR", tmp_path)
    runner_main._MODULE_CACHE.clear()

    registry = runner_main.discover_benchmarks()

    assert sorted(registry) == ["lazy.visible"]
    assert not marker_path.exists()
    assert registry["lazy.visible"](3) == 3.5
    assert marker_path.read_text(encoding="utf-8") == "imported"


def test_ensure_pyperf_worker_env_preserves_existing_args(monkeypatch):
    runner_main = load_runner_main(monkeypatch)

    for env_var in runner_main.PYPERF_INHERITED_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64")

    argv = runner_main.ensure_pyperf_worker_env(["--fast", "--inherit-environ=FOO,BAR"])

    assert argv == ["--fast", "--inherit-environ", "FOO,BAR,CUDA_PATH,LD_LIBRARY_PATH"]


def test_bench_launch_initializes_on_first_use(monkeypatch):
    calls: list[tuple] = []
    bench_launch = load_bench_launch(monkeypatch, calls)

    assert calls == []

    bench_launch.bench_launch_empty_kernel(1)
    compile_calls = [call for call in calls if call[0] == "compile_and_load"]
    launch_calls = [call for call in calls if call[0] == "cuLaunchKernel"]

    assert len(compile_calls) == 1
    assert len(launch_calls) == 1

    bench_launch.bench_launch_16_args_pre_packed(1)
    compile_calls = [call for call in calls if call[0] == "compile_and_load"]
    launch_calls = [call for call in calls if call[0] == "cuLaunchKernel"]

    assert len(compile_calls) == 1
    assert len(launch_calls) == 2
