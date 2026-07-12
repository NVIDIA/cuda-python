# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consistency check between ``cuda.core.__all__`` and the public API docs.

Compares the public symbols exported by ``cuda.core.__all__`` against the
public entries in ``docs/source/api.rst``. Symbols documented in
``docs/source/api_private.rst`` (returned helpers users cannot instantiate)
are accepted as documented. Dotted entries such as ``graph.Graph`` or
``checkpoint.Process`` describe submodule namespaces, not the flat
``cuda.core`` namespace, and are excluded from the comparison.
"""

import pathlib

import pytest

import cuda.core

DOCS_SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "source"


def _iter_directive_blocks(text):
    """Yield (module, directive, argument, body_lines) for each RST directive.

    ``module`` is the module active at the directive, tracked from
    ``.. module::`` and ``.. currentmodule::`` directives. ``body_lines`` is
    the indented block that follows the directive.
    """
    module = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.startswith(".. "):
            continue
        head, _, argument = line[3:].partition("::")
        directive = head.strip()
        argument = argument.strip()
        if directive in ("module", "currentmodule"):
            module = argument
            continue
        body = []
        while i < len(lines):
            body_line = lines[i]
            if body_line.strip() and not body_line.startswith(" "):
                break
            body.append(body_line)
            i += 1
        yield module, directive, argument, body


def _documented_names(rst_path, module="cuda.core"):
    """Collect names documented for ``module`` in an RST file.

    Returns the entries of ``autosummary`` blocks plus ``.. data::``
    arguments that appear while ``module`` is the active module.
    """
    names = set()
    for active_module, directive, argument, body in _iter_directive_blocks(rst_path.read_text()):
        if active_module != module:
            continue
        if directive == "data":
            names.add(argument)
        elif directive == "autosummary":
            for line in body:
                entry = line.strip()
                if not entry or entry.startswith(":"):
                    continue
                names.add(entry)
    return names


def _flat_names(names):
    """Filter out dotted (submodule-namespace) entries."""
    return {name for name in names if "." not in name}


@pytest.fixture(scope="module")
def exported():
    if not hasattr(cuda.core, "__all__"):
        pytest.skip("cuda.core does not define __all__")
    return set(cuda.core.__all__)


@pytest.fixture(scope="module")
def docs_dir():
    if not (DOCS_SOURCE_DIR / "api.rst").is_file():
        pytest.skip("docs sources not available (not running from a source checkout)")
    return DOCS_SOURCE_DIR


def test_all_exports_resolve():
    if not hasattr(cuda.core, "__all__"):
        pytest.skip("cuda.core does not define __all__")
    missing = [name for name in cuda.core.__all__ if not hasattr(cuda.core, name)]
    assert missing == [], f"cuda.core.__all__ lists names that do not resolve: {missing}"


def test_public_symbols_are_documented(exported, docs_dir):
    documented = _flat_names(_documented_names(docs_dir / "api.rst"))
    # Returned helpers are deliberately documented in api_private.rst; accept
    # them by their trailing name (e.g. _device_resources.DeviceResources).
    private_documented = {name.rsplit(".", 1)[-1] for name in _documented_names(docs_dir / "api_private.rst")}
    undocumented = exported - documented - private_documented
    assert not undocumented, (
        f"public by cuda.core.__all__ but missing from public docs (api.rst): {sorted(undocumented)}"
    )


def test_documented_symbols_are_exported(exported, docs_dir):
    documented = _flat_names(_documented_names(docs_dir / "api.rst"))
    unexported = documented - exported
    assert not unexported, (
        f"documented as public in api.rst but not exported by cuda.core.__all__: {sorted(unexported)}"
    )
