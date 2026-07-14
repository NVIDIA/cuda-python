# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consistency checks between the public ``__all__`` surface and the API docs.

Covers the flat ``cuda.core`` namespace and every public subpackage
(``graph``, ``system``, ``texture``, ``utils``, and any added later) discovered
automatically from ``cuda.core.__path__``. For each public namespace, exported
``__all__`` names must appear somewhere in ``cuda_core/docs/source``.
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

PUBLIC_SUBPACKAGES = sorted(
    name
    for _, name, ispkg in pkgutil.iter_modules(cuda.core.__path__)
    if ispkg and not name.startswith("_") and not _VERSIONED_SUBPACKAGE.match(name)
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


def _public_doc_paths(docs_dir):
    return sorted(path for path in docs_dir.glob("*.rst") if path.name != "api_private.rst")


def _add_documented_name(documented, module, entry):
    if not module or not module.startswith("cuda.core"):
        return
    if module == "cuda.core":
        if "." not in entry:
            documented[module].add(entry)
            return
        sub, name = entry.split(".", 1)
        if sub in PUBLIC_SUBPACKAGES and "." not in name:
            documented[f"cuda.core.{sub}"].add(name)
        return
    if module.startswith("cuda.core."):
        namespace = module
        if namespace in PUBLIC_NAMESPACES and "." not in entry:
            documented[namespace].add(entry)


def _documented_names(paths):
    documented = collections.defaultdict(set)
    for rst_path in paths:
        for module, entry in _iter_documented_entries(rst_path):
            _add_documented_name(documented, module, entry)
    return documented


def _private_documented_names(docs_dir):
    names = collections.defaultdict(set)
    private_path = docs_dir / "api_private.rst"
    if not private_path.is_file():
        return names
    for module, entry in _iter_documented_entries(private_path):
        if module == "cuda.core":
            if "." in entry:
                sub, name = entry.split(".", 1)
                if sub in PUBLIC_SUBPACKAGES:
                    names[f"cuda.core.{sub}"].add(name.rsplit(".", 1)[-1])
                else:
                    names[module].add(entry.rsplit(".", 1)[-1])
            else:
                names[module].add(entry)
        elif module in PUBLIC_NAMESPACES:
            names[module].add(entry.rsplit(".", 1)[-1])
    return names


PUBLIC_NAMESPACES = ("cuda.core", *(f"cuda.core.{sub}" for sub in PUBLIC_SUBPACKAGES))


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
def public_documented(docs_dir):
    return _documented_names(_public_doc_paths(docs_dir))


@pytest.fixture(scope="module")
def private_documented(docs_dir):
    return _private_documented_names(docs_dir)


def test_public_subpackages_discovered():
    # Guards against a broken __path__ walk silently turning every
    # parametrized subpackage check into a no-op.
    assert PUBLIC_SUBPACKAGES, "no public cuda.core subpackages were discovered"


def test_all_exports_resolve():
    if not hasattr(cuda.core, "__all__"):
        pytest.skip("cuda.core does not define __all__")
    missing = [name for name in cuda.core.__all__ if not hasattr(cuda.core, name)]
    assert missing == [], f"cuda.core.__all__ lists names that do not resolve: {missing}"


def test_public_symbols_are_documented(exported, public_documented, private_documented):
    documented = public_documented["cuda.core"]
    # Returned helpers are deliberately documented in api_private.rst; accept
    # them by their trailing name.
    private = private_documented["cuda.core"]
    undocumented = exported - documented - private
    assert not undocumented, f"public by cuda.core.__all__ but missing from docs/source/*.rst: {sorted(undocumented)}"


def test_documented_symbols_are_exported(exported, public_documented):
    documented = public_documented["cuda.core"]
    unexported = documented - exported
    assert not unexported, (
        f"documented as public in docs/source/*.rst but not exported by cuda.core.__all__: {sorted(unexported)}"
    )


@pytest.mark.parametrize("sub", PUBLIC_SUBPACKAGES)
def test_subpackage_defines_all(sub):
    module = importlib.import_module(f"cuda.core.{sub}")
    assert hasattr(module, "__all__"), f"cuda.core.{sub} does not define __all__"
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"cuda.core.{sub}.__all__ lists names that do not resolve: {missing}"


@pytest.mark.parametrize("sub", PUBLIC_SUBPACKAGES)
def test_subpackage_exports_match_docs(sub, public_documented, private_documented):
    documented = public_documented[f"cuda.core.{sub}"]
    module = importlib.import_module(f"cuda.core.{sub}")
    exported = set(module.__all__)
    private = private_documented[f"cuda.core.{sub}"]
    undocumented = exported - documented - private
    assert not undocumented, (
        f"public by cuda.core.{sub}.__all__ but missing from docs/source/*.rst: {sorted(undocumented)}"
    )
    unexported = documented - exported
    assert not unexported, (
        f"documented as public in docs/source/*.rst under {sub} but not exported by cuda.core.{sub}.__all__: "
        f"{sorted(unexported)}"
    )
