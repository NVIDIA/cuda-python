# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from run_samples import RunPlan, run_sample

from cuda.bindings import driver

REPO_ROOT = Path(__file__).resolve().parents[3]
UTILITIES_DIR = REPO_ROOT / "samples" / "cuda_core" / "Utilities"
WAIVER_SAMPLE_NAMES = (
    "blurImageUnifiedMemory",
    "copyImageArraytoGPU",
    "glInteropFluid",
    "glInteropFluidNumbaCudaMlir",
    "glInteropMipmapLod",
    "glInteropPlasma",
    "ipcMemoryPool",
    "launchConfigTuning",
    "memoryResources",
    "pageRank",
    "processCheckpoint",
    "simpleMultiGpu",
    "simpleP2P",
    "stridedMemoryViewConstructors",
    "threadBlockCluster",
    "tmaTensorMap",
)


def _load_sample_function(sample_name: str, function_name: str, **namespace):
    """Exercise sample functions without importing their optional GPU packages.

    Callers supply the globals needed by the exercised paths through ``namespace``;
    changes to those dependencies may require updating the test doubles.
    """
    sample_path = REPO_ROOT / "samples" / "cuda_core" / sample_name / f"{sample_name}.py"
    tree = ast.parse(sample_path.read_text(encoding="utf-8"), filename=str(sample_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(sample_path), "exec"), namespace)  # noqa: S102 - trusted sample source
    return namespace[function_name]


@pytest.mark.parametrize("host_pools_supported", [False, True])
@pytest.mark.agent_authored(model="GPT-6")
def test_image_sample_requires_host_memory_pools(host_pools_supported: bool) -> None:
    def create_stream():
        raise RuntimeError("stream setup failed")

    device = SimpleNamespace(
        properties=SimpleNamespace(memory_pools_supported=True, host_memory_pools_supported=host_pools_supported),
        set_current=lambda: None,
        create_stream=create_stream,
    )
    main = _load_sample_function("copyImageArraytoGPU", "main", Device=lambda: device, np=np, sys=sys, EXIT_WAIVED=77)

    if host_pools_supported:
        with pytest.raises(RuntimeError, match="stream setup failed"):
            main()
    else:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 77


@pytest.mark.parametrize(
    ("handle_types", "supported"),
    [
        (int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32), False),
        (
            int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32)
            | int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            True,
        ),
    ],
)
@pytest.mark.agent_authored(model="GPT-6")
def test_ipc_sample_checks_memory_pool_handle_support(handle_types: int, supported: bool) -> None:
    check_ipc_support = _load_sample_function(
        "ipcMemoryPool",
        "check_ipc_support",
        platform=SimpleNamespace(system=lambda: "Linux"),
        open=lambda _path: io.StringIO("6.1.0"),
        driver=driver,
    )
    device = SimpleNamespace(
        properties=SimpleNamespace(
            memory_pools_supported=True,
            mempool_supported_handle_types=handle_types,
        )
    )

    assert check_ipc_support(device) is supported


@pytest.mark.parametrize("display", [None, ":0"])
@pytest.mark.agent_authored(model="GPT-6")
def test_mlir_sample_waives_missing_display_before_cuda_setup(display: str | None) -> None:
    def is_available():
        raise RuntimeError("CUDA query failed")

    main = _load_sample_function(
        "glInteropFluidNumbaCudaMlir",
        "main",
        os=SimpleNamespace(environ={} if display is None else {"DISPLAY": display}),
        platform=SimpleNamespace(system=lambda: "Linux"),
        sys=sys,
        cuda=SimpleNamespace(is_available=is_available),
        EXIT_WAIVED=77,
    )

    if display is None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 77
    else:
        with pytest.raises(RuntimeError, match="CUDA query failed"):
            main()


@pytest.mark.parametrize("sample_name", WAIVER_SAMPLE_NAMES)
@pytest.mark.agent_authored(model="gpt-5")
def test_core_sample_waivers_use_negotiated_exit_code(
    sample_name: str,
) -> None:
    sample_path = REPO_ROOT / "samples" / "cuda_core" / sample_name / f"{sample_name}.py"
    tree = ast.parse(sample_path.read_text(encoding="utf-8"), filename=str(sample_path))

    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXIT_WAIVED" for target in node.targets)
    ]
    assert len(assignments) == 1
    expected_assignment = ast.parse(
        'EXIT_WAIVED = int(os.environ.get("CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE", "2"))'
    ).body[0]
    assert isinstance(expected_assignment, ast.Assign)
    assert ast.dump(assignments[0].value) == ast.dump(expected_assignment.value)

    legacy_waivers = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value == 2) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "exit"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 2
        ):
            legacy_waivers.append(node.lineno)
    assert not legacy_waivers, f"literal exit-2 waivers remain at lines {legacy_waivers}"


@pytest.mark.parametrize(
    ("sample_name", "unsupported_condition"),
    [
        ("ipcMemoryPool", "not check_ipc_support(device)"),
        ("processCheckpoint", 'sys.platform != "linux"'),
        ("processCheckpoint", "device.properties.integrated"),
        ("tmaTensorMap", "arch < (9, 0)"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5")
def test_unsupported_core_sample_paths_return_waiver(
    sample_name: str,
    unsupported_condition: str,
) -> None:
    sample_path = REPO_ROOT / "samples" / "cuda_core" / sample_name / f"{sample_name}.py"
    tree = ast.parse(sample_path.read_text(encoding="utf-8"), filename=str(sample_path))
    expected_condition = ast.parse(unsupported_condition, mode="eval").body

    matching_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(expected_condition)
    ]
    assert len(matching_branches) == 1
    waiver_returns = [
        node
        for node in matching_branches[0].body
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "EXIT_WAIVED"
    ]
    assert len(waiver_returns) == 1


def _load_simple_p2p(monkeypatch: pytest.MonkeyPatch):
    sample_path = REPO_ROOT / "samples" / "cuda_core" / "simpleP2P" / "simpleP2P.py"
    spec = importlib.util.spec_from_file_location("_test_simple_p2p", sample_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as import_ctx:
        import_ctx.setattr(sys, "path", list(sys.path))
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("unsupported_gpu", "unsupported_capability"),
    [
        (0, "memory_pools_supported"),
        (1, "memory_pools_supported"),
        (0, "host_memory_pools_supported"),
        (1, "host_memory_pools_supported"),
    ],
)
@pytest.mark.agent_authored(model="gemini-4-carbon-high")
def test_simple_p2p_waives_before_allocation_without_memory_pool_support(
    monkeypatch: pytest.MonkeyPatch,
    unsupported_gpu: int,
    unsupported_capability: str,
) -> None:
    """Missing device or host mempool support on any selected GPU must waive before allocating (NVBug 7081727)."""
    module = _load_simple_p2p(monkeypatch)

    # GPU 0 is non-P2P so the sample selects the (1, 2) pair rather than hardcoded ordinals (0, 1).
    p2p_pair = (1, 2)
    target_device_id = p2p_pair[unsupported_gpu]
    devices = []
    for device_id in range(3):
        props = {
            "memory_pools_supported": True,
            "host_memory_pools_supported": True,
        }
        if device_id == target_device_id:
            props[unsupported_capability] = False
        devices.append(
            SimpleNamespace(
                device_id=device_id,
                name=f"FakeGPU{device_id}",
                properties=SimpleNamespace(**props),
                can_access_peer=lambda other, dev_id=device_id: (
                    dev_id in p2p_pair and other.device_id in p2p_pair and dev_id != other.device_id
                ),
                set_current=lambda: None,
            )
        )

    class _FakeDeviceFactory:
        def __new__(cls, device_id: int):
            return devices[int(device_id)]

        @staticmethod
        def get_all_devices():
            return list(devices)

    constructed_resources: list[str] = []

    def _record_construction(kind: str):
        def _fail(*args, **kwargs):
            constructed_resources.append(kind)
            raise AssertionError(f"{kind} must not be constructed when memory pools are unsupported")

        return _fail

    monkeypatch.setattr(module, "Device", _FakeDeviceFactory)
    monkeypatch.setattr(module, "DeviceMemoryResource", _record_construction("DeviceMemoryResource"))
    monkeypatch.setattr(module, "PinnedMemoryResource", _record_construction("PinnedMemoryResource"))

    assert module.run(num_elements=16) == module.EXIT_WAIVED
    assert constructed_resources == []


@pytest.mark.agent_authored(model="gpt-5")
def test_verify_array_result_or_raise_enforces_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(UTILITIES_DIR))
    from cuda_samples_utils import verify_array_result_or_raise

    verify_array_result_or_raise(
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="forced mismatch"):
        verify_array_result_or_raise(
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            verbose=False,
            error_message="forced mismatch",
        )


@pytest.mark.agent_authored(model="gpt-5")
def test_verification_exception_is_reported_as_sample_failure(tmp_path: Path) -> None:
    sample_dir = tmp_path / "verificationFailure"
    sample_dir.mkdir()
    sample = sample_dir / "verificationFailure.py"
    sample.write_text(
        "\n".join(
            (
                "import sys",
                f"sys.path.insert(0, {str(UTILITIES_DIR)!r})",
                "import numpy as np",
                "from cuda_samples_utils import verify_array_result_or_raise",
                "verify_array_result_or_raise(np.array([1.0]), np.array([2.0]), verbose=False)",
            )
        ),
        encoding="utf-8",
    )

    result = run_sample(RunPlan(sample, [], [], timeout=10))

    assert result.status == "FAIL"
    assert result.return_code != 0
