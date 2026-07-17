# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .run_samples import RunPlan, run_sample

REPO_ROOT = Path(__file__).resolve().parents[3]
UTILITIES_DIR = REPO_ROOT / "samples" / "cuda_core" / "Utilities"


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
