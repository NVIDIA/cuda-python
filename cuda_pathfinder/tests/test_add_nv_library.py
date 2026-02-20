# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_wizard_module():
    script_path = Path(__file__).resolve().parents[2] / "toolshed" / "add-nv-library.py"
    spec = importlib.util.spec_from_file_location("add_nv_library", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_descriptor(mod):
    return mod.DescriptorInput(
        name="foo_lib",
        strategy="other",
        linux_sonames=("libfoo.so.1",),
        windows_dlls=("foo64_1.dll",),
        site_packages_linux=("nvidia/foo/lib",),
        site_packages_windows=("nvidia/foo/bin",),
        dependencies=("bar",),
        anchor_rel_dirs_linux=("lib64", "lib"),
        anchor_rel_dirs_windows=("bin/x64", "bin"),
        requires_add_dll_directory=True,
        requires_rtld_deepbind=False,
    )


def test_all_form_fields_supplied_detects_complete_inputs():
    mod = _load_wizard_module()
    args = mod._build_arg_parser().parse_args([
        "--name",
        "foo_lib",
        "--strategy",
        "other",
        "--linux-sonames",
        "libfoo.so.1",
        "--windows-dlls",
        "foo64_1.dll",
        "--site-packages-linux",
        "nvidia/foo/lib",
        "--site-packages-windows",
        "nvidia/foo/bin",
        "--dependencies",
        "bar",
        "--anchor-rel-dirs-linux",
        "lib64,lib",
        "--anchor-rel-dirs-windows",
        "bin/x64,bin",
        "--requires-add-dll-directory",
        "true",
        "--requires-rtld-deepbind",
        "false",
    ])
    assert mod._all_form_fields_supplied(args)


def test_build_input_from_args_parses_values():
    mod = _load_wizard_module()
    args = mod._build_arg_parser().parse_args([
        "--name",
        "foo_lib",
        "--strategy",
        "other",
        "--linux-sonames",
        "libfoo.so.1",
        "--windows-dlls",
        "foo64_1.dll",
        "--site-packages-linux",
        "nvidia/foo/lib",
        "--site-packages-windows",
        "nvidia/foo/bin",
        "--dependencies",
        "bar,baz",
        "--anchor-rel-dirs-linux",
        "lib64,lib",
        "--anchor-rel-dirs-windows",
        "bin/x64,bin",
        "--requires-add-dll-directory",
        "true",
        "--requires-rtld-deepbind",
        "false",
    ])
    spec = mod._build_input_from_args(args)
    assert spec.name == "foo_lib"
    assert spec.dependencies == ("bar", "baz")
    assert spec.requires_add_dll_directory is True
    assert spec.requires_rtld_deepbind is False


def test_render_descriptor_block_includes_expected_lines():
    mod = _load_wizard_module()
    block = mod.render_descriptor_block(_sample_descriptor(mod))
    assert 'name="foo_lib"' in block
    assert 'strategy="other"' in block
    assert 'linux_sonames=("libfoo.so.1",)' in block
    assert "requires_add_dll_directory=True" in block
    assert block.strip().endswith("),")


def test_merge_descriptor_block_inserts_before_catalog_closing():
    mod = _load_wizard_module()
    original = """from x import y

DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
    DescriptorSpec(
        name="existing",
        strategy="ctk",
    ),
)  # END DESCRIPTOR_CATALOG
"""
    updated = mod.merge_descriptor_block(original, _sample_descriptor(mod))
    assert 'name="foo_lib"' in updated
    assert updated.rfind('name="foo_lib"') < updated.rfind("# END DESCRIPTOR_CATALOG")


def test_merge_descriptor_block_rejects_duplicate_names():
    mod = _load_wizard_module()
    original = """DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
    DescriptorSpec(
        name="foo_lib",
        strategy="other",
    ),
)  # END DESCRIPTOR_CATALOG
"""
    with pytest.raises(ValueError, match="already exists"):
        mod.merge_descriptor_block(original, _sample_descriptor(mod))


def test_apply_descriptor_dry_run_does_not_modify_file(tmp_path):
    mod = _load_wizard_module()
    catalog = tmp_path / "descriptor_catalog.py"
    original = """DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
)  # END DESCRIPTOR_CATALOG
"""
    catalog.write_text(original, encoding="utf-8")
    updated = mod.apply_descriptor(catalog, _sample_descriptor(mod), dry_run=True)
    assert 'name="foo_lib"' in updated
    assert catalog.read_text(encoding="utf-8") == original


def test_apply_descriptor_writes_file_when_not_dry_run(tmp_path):
    mod = _load_wizard_module()
    catalog = tmp_path / "descriptor_catalog.py"
    catalog.write_text(
        "DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (\n)  # END DESCRIPTOR_CATALOG\n", encoding="utf-8"
    )
    mod.apply_descriptor(catalog, _sample_descriptor(mod), dry_run=False)
    assert 'name="foo_lib"' in catalog.read_text(encoding="utf-8")
