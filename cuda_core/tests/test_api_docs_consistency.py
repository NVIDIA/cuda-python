# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consistency checks between the public ``__all__`` surface and the API docs.

Covers the flat ``cuda.core`` namespace and every public subpackage
(``graph``, ``system``, ``texture``, ``utils``, and any added later) discovered
automatically from ``cuda.core.__path__``. For each, the exported ``__all__``
is compared against the public entries in ``docs/source/api.rst``. Symbols
documented in ``docs/source/api_private.rst`` (returned helpers users cannot
instantiate) are accepted as documented.

Docs reference submodule symbols two ways: dotted entries such as
``graph.Graph`` under the ``cuda.core`` module, and flat entries under a
``.. currentmodule:: cuda.core.<sub>`` block. Both forms are collected. A
subpackage documented outside ``api.rst`` (for example ``system``, whose
reference lives in ``api_nvml.rst``) is still checked for a well-formed
``__all__``, but its doc cross-check is skipped here.
"""

import importlib
import pathlib
import pkgutil
import re

import pytest

import cuda.core

DOCS_SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "source"

# ``cuda.core`` ships a versioned wheel shim as ``cu12`` / ``cu13`` subpackages;
# those are an internal packaging mechanism, not public API.
_VERSIONED_SUBPACKAGE = re.compile(r"^cu\d+$")

PUBLIC_SUBPACKAGES = sorted(
    name
    for _, name, ispkg in pkgutil.iter_modules(cuda.core.__path__)
    if ispkg and not name.startswith("_") and not _VERSIONED_SUBPACKAGE.match(name)
)


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


def _documented_subpackage_names(rst_path, sub):
    """Collect the public names documented for subpackage ``sub`` in ``rst_path``.

    Combines the two doc conventions: flat entries under
    ``.. currentmodule:: cuda.core.<sub>`` and dotted ``<sub>.Name`` entries
    written under the ``cuda.core`` module.
    """
    flat = _documented_names(rst_path, module=f"cuda.core.{sub}")
    dotted = {
        name.split(".", 1)[1]
        for name in _documented_names(rst_path, module="cuda.core")
        if name.startswith(f"{sub}.") and name.count(".") == 1
    }
    return flat | dotted


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


def test_public_subpackages_discovered():
    # Guards against a broken __path__ walk silently turning every
    # parametrized subpackage check into a no-op.
    assert PUBLIC_SUBPACKAGES, "no public cuda.core subpackages were discovered"


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


@pytest.mark.parametrize("sub", PUBLIC_SUBPACKAGES)
def test_subpackage_defines_all(sub):
    module = importlib.import_module(f"cuda.core.{sub}")
    assert hasattr(module, "__all__"), f"cuda.core.{sub} does not define __all__"
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"cuda.core.{sub}.__all__ lists names that do not resolve: {missing}"


@pytest.mark.parametrize("sub", PUBLIC_SUBPACKAGES)
def test_subpackage_exports_match_docs(sub, docs_dir):
    documented = _documented_subpackage_names(docs_dir / "api.rst", sub)
    if not documented:
        pytest.skip(f"cuda.core.{sub} is not documented in api.rst (its reference lives elsewhere)")
    module = importlib.import_module(f"cuda.core.{sub}")
    exported = set(module.__all__)
    private_documented = {name.rsplit(".", 1)[-1] for name in _documented_names(docs_dir / "api_private.rst")}
    undocumented = exported - documented - private_documented
    assert not undocumented, (
        f"public by cuda.core.{sub}.__all__ but missing from public docs (api.rst): {sorted(undocumented)}"
    )
    unexported = documented - exported
    assert not unexported, (
        f"documented as public in api.rst under {sub} but not exported by cuda.core.{sub}.__all__: {sorted(unexported)}"
    )
