# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _install_pathspec_stub():
    if "pathspec" in sys.modules:
        return

    class _StubSpec:
        def match_file(self, _filepath):
            return False

    class _StubPathSpec:
        @staticmethod
        def from_lines(_pattern_type, _lines):
            return _StubSpec()

    module = ModuleType("pathspec")
    module.PathSpec = _StubPathSpec
    sys.modules["pathspec"] = module


def _load_check_spdx():
    check_spdx_path = Path(__file__).resolve().with_name("check_spdx.py")
    spec = importlib.util.spec_from_file_location("check_spdx", check_spdx_path)
    assert spec is not None
    assert spec.loader is not None
    _install_pathspec_stub()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_spdx = _load_check_spdx()


def _write_spdx_file(root, relative_path, license_identifier, *, years="2025-2026"):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"# SPDX-FileCopyrightText: Copyright (c) {years} NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved.\n"
            f"# SPDX-License-Identifier: {license_identifier}\n"
            "\n"
            "print('hello')\n"
        ),
        encoding="ascii",
    )
    return path


def test_get_expected_license_identifier_normalizes_windows_paths():
    assert check_spdx.get_expected_license_identifier(r".\cuda_core\example.py") == "Apache-2.0"


def test_find_or_fix_spdx_rejects_non_apache_license_under_cuda_core(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_spdx, "is_staged", lambda _: False)
    _write_spdx_file(tmp_path, "cuda_core/example.py", "LicenseRef-NVIDIA-SOFTWARE-LICENSE")

    assert not check_spdx.find_or_fix_spdx("cuda_core/example.py", fix=False)

    assert "expected 'Apache-2.0'" in capsys.readouterr().out


def test_find_or_fix_spdx_allows_non_apache_license_outside_cuda_core(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_spdx, "is_staged", lambda _: False)
    _write_spdx_file(tmp_path, "cuda_bindings/example.py", "LicenseRef-NVIDIA-SOFTWARE-LICENSE")

    assert check_spdx.find_or_fix_spdx("cuda_bindings/example.py", fix=False)


def test_find_or_fix_spdx_updates_outdated_copyright_when_fix_requested(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_spdx, "CURRENT_YEAR", "2026")
    monkeypatch.setattr(check_spdx, "is_staged", lambda _: True)
    path = _write_spdx_file(tmp_path, "cuda_core/example.py", "Apache-2.0", years="2024")

    assert not check_spdx.find_or_fix_spdx("cuda_core/example.py", fix=True)

    assert "OUTDATED copyright '2024' (expected '2026')" in capsys.readouterr().out
    assert "Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved." in path.read_text(
        encoding="ascii"
    )
