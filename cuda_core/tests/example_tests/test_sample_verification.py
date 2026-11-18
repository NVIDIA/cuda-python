# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from run_samples import RunPlan, run_sample

REPO_ROOT = Path(__file__).resolve().parents[3]
UTILITIES_DIR = REPO_ROOT / "samples" / "cuda_core" / "Utilities"
WAIVER_SAMPLE_NAMES = (
    "blurImageUnifiedMemory",
    "copyImageArraytoGPU",
    "glInteropFluid",
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
    """Exercise sample functions without importing their optional GPU packages."""
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


@pytest.mark.parametrize(("handle_types", "supported"), [(2, False), (3, True)])
@pytest.mark.agent_authored(model="GPT-6")
def test_ipc_sample_checks_memory_pool_handle_support(handle_types: int, supported: bool) -> None:
    from cuda.bindings import driver

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
            handle_type_posix_file_descriptor_supported=True,
            mempool_supported_handle_types=handle_types,
        )
    )

    assert check_ipc_support(device) is supported


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
