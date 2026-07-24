# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consistency checks between the public ``__all__`` surface and the API docs.

Covers the flat ``cuda.core`` namespace and every public submodule
(``checkpoint``, ``graph``, ``system``, ``texture``, ``typing``, ``utils``,
and any added later) discovered automatically from ``cuda.core.__path__``.
For each public namespace, exported ``__all__`` names must appear somewhere
in ``cuda_core/docs/source``.

The enforced direction is deliberately one-way (public export -> documented).
This is intentionally a *name-presence* check, and it does not verify:

- the reverse direction (documented -> exported): documenting a private or
  internal symbol on any page is allowed, so a documented name is never
  required to be public;
- signatures, docstrings, parameter lists, or rendered output: only that each
  exported name appears as a documented entry;
- whether an entry is marked ``:no-index:`` or deprecated: such entries still
  count as documented;
- class members or attributes nested below the namespace level: only top-level
  names of each namespace are matched (entries deeper than
  ``<subpackage>.<name>`` are ignored);
- docs outside the top-level ``docs/source/*.rst`` files: nested pages are not
  scanned.
"""

import collections
import importlib
import io
import pathlib
import pkgutil
import re

import pytest
from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive, directives

import cuda.core

DOCS_SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "source"

# ``cuda.core`` ships a versioned wheel shim as ``cu12`` / ``cu13`` subpackages;
# those are an internal packaging mechanism, not public API.
_VERSIONED_SUBPACKAGE = re.compile(r"^cu\d+$")

PUBLIC_SUBMODULES = sorted(
    name
    for _, name, ispkg in pkgutil.iter_modules(cuda.core.__path__)
    if not name.startswith("_") and not _VERSIONED_SUBPACKAGE.match(name)
)


class _ModuleNode(nodes.Element):
    pass


class _AutosummaryNode(nodes.Element):
    pass


class _DataNode(nodes.Element):
    pass


class _ModuleDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = False
    has_content = True
    option_spec = {
        "deprecated": directives.unchanged,
        "no-index": directives.flag,
        "platform": directives.unchanged,
        "synopsis": directives.unchanged,
    }

    def run(self):
        node = _ModuleNode()
        node["module"] = self.arguments[0].strip()
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class _AutosummaryDirective(Directive):
    has_content = True
    option_spec = {
        "caption": directives.unchanged,
        "nosignatures": directives.flag,
        "recursive": directives.flag,
        "template": directives.unchanged,
        "toctree": directives.unchanged,
    }

    def run(self):
        node = _AutosummaryNode()
        node["entries"] = [entry for line in self.content if (entry := line.strip()) and not entry.startswith(":")]
        return [node]


class _DataDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = {
        "annotation": directives.unchanged,
        "no-index": directives.flag,
        "type": directives.unchanged,
        "value": directives.unchanged,
    }

    def run(self):
        node = _DataNode()
        node["name"] = self.arguments[0].strip()
        return [node]


# These patch the global docutils directive registry for the process lifetime.
# Safe as long as no other test module in the same session uses docutils or
# Sphinx with the real autosummary/module/data directives. If that ever changes,
# move these calls into a session-scoped autouse fixture that saves and restores
# the previous mapping.
directives.register_directive("autosummary", _AutosummaryDirective)
directives.register_directive("currentmodule", _ModuleDirective)
directives.register_directive("data", _DataDirective)
directives.register_directive("module", _ModuleDirective)


def _iter_documented_entries(rst_path):
    """Yield (module, entry) pairs from Sphinx directives in an RST file."""
    doctree = publish_doctree(
        rst_path.read_text(),
        source_path=str(rst_path),
        settings_overrides={
            "halt_level": 6,
            "report_level": 5,
            "warning_stream": io.StringIO(),
        },
    )
    module = None
    for node in doctree.findall():
        if isinstance(node, _ModuleNode):
            module = node["module"]
        elif isinstance(node, _AutosummaryNode):
            for entry in node["entries"]:
                yield module, entry
        elif isinstance(node, _DataNode):
            yield module, node["name"]


def _add_documented_name(documented, module, entry):
    if not module or not module.startswith("cuda.core"):
        return
    if module == "cuda.core":
        if "." not in entry:
            documented[module].add(entry)
            return
        sub, name = entry.split(".", 1)
        if sub in PUBLIC_SUBMODULES and "." not in name:
            documented[f"cuda.core.{sub}"].add(name)
        return
    if module.startswith("cuda.core."):
        namespace = module
        if namespace in PUBLIC_NAMESPACES and "." not in entry:
            documented[namespace].add(entry)


def _documented_names(docs_dir, *, exclude=frozenset()):
    documented = collections.defaultdict(set)
    for rst_path in docs_dir.glob("*.rst"):
        if rst_path.name in exclude:
            continue
        for module, entry in _iter_documented_entries(rst_path):
            _add_documented_name(documented, module, entry)
    return documented


PUBLIC_NAMESPACES = ("cuda.core", *(f"cuda.core.{sub}" for sub in PUBLIC_SUBMODULES))


@pytest.fixture(scope="module")
def exported():
    if not hasattr(cuda.core, "__all__"):
        pytest.skip("cuda.core does not define __all__")
    return set(cuda.core.__all__)


@pytest.fixture(scope="module")
def docs_dir():
    if not DOCS_SOURCE_DIR.is_dir():
        pytest.skip("docs sources not available (not running from a source checkout)")
    return DOCS_SOURCE_DIR


@pytest.fixture(scope="module")
def documented(docs_dir):
    return _documented_names(docs_dir)


@pytest.mark.human_authored
def test_public_submodules_discovered():
    # Guards against a broken __path__ walk silently turning every
    # parametrized submodule check into a no-op.
    assert PUBLIC_SUBMODULES, "no public cuda.core submodules were discovered"


@pytest.mark.human_authored
def test_main_package_all_exports_resolve():
    assert hasattr(cuda.core, "__all__"), "cuda.core does not define __all__"
    missing = [name for name in cuda.core.__all__ if not hasattr(cuda.core, name)]
    assert missing == [], f"cuda.core.__all__ lists names that do not resolve: {missing}"


@pytest.mark.human_authored
def test_main_package_symbols_are_documented(exported, documented):
    documented = documented["cuda.core"]
    undocumented = exported - documented
    assert not undocumented, f"public by cuda.core.__all__ but missing from docs/source/*.rst: {sorted(undocumented)}"


@pytest.mark.parametrize("sub", PUBLIC_SUBMODULES)
def test_subpackage_symbols_define_all(sub):
    module = importlib.import_module(f"cuda.core.{sub}")
    assert hasattr(module, "__all__"), f"cuda.core.{sub} does not define __all__"
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"cuda.core.{sub}.__all__ lists names that do not resolve: {missing}"


@pytest.mark.human_authored
@pytest.mark.parametrize("sub", PUBLIC_SUBMODULES)
def test_subpackage_exports_are_documented(sub, documented):
    documented = documented[f"cuda.core.{sub}"]
    module = importlib.import_module(f"cuda.core.{sub}")
    exported = set(module.__all__)
    undocumented = exported - documented
    assert not undocumented, (
        f"public by cuda.core.{sub}.__all__ but missing from docs/source/*.rst: {sorted(undocumented)}"
    )
